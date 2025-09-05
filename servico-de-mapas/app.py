# app.py
import os, io, json, base64, math, logging, random
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union
from PIL import Image
from flask import Flask, request, send_file, jsonify

# --- Config ---
DEFAULT_PROVIDER = os.getenv("TILE_PROVIDER", "google_hybrid")  # google_hybrid | mapbox_hybrid | osm
GOOGLE_TEMPLATES = [
    "https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
]
MAPBOX_TOKEN     = os.getenv("MAPBOX_TOKEN", "")
USER_AGENT       = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0 (contato@smartfazendas.com.br)")
DEFAULT_LOGO_URL = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_DARKEN_ALPHA = float(os.getenv("DARKEN_ALPHA", "0.45"))
DEFAULT_JPG_QUALITY  = int(os.getenv("JPG_QUALITY", "85"))
BRAND_COLOR = os.getenv("BRAND_COLOR", "#635AFF")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate-map")

# --- Dependências opcionais (para logo) ---
try:
    import requests
except Exception:
    requests = None

# --- Compat: ImageTiles pode não existir em algumas versões do Cartopy ---
HAS_IMGTILES = hasattr(cimgt, "ImageTiles")

if HAS_IMGTILES:
    class XYZTiles(cimgt.ImageTiles):
        def __init__(self, url_template: str, cache=True, user_agent=None):
            self.url_template = url_template
            super().__init__(cache=cache, user_agent=user_agent)
        def _image_url(self, tile):
            x, y, z = tile
            return self.url_template.format(x=x, y=y, z=z)
else:
    XYZTiles = None  # sem suporte a XYZ genérico nesta versão do cartopy

# --- Funções auxiliares ---
def _make_tile_source(provider: str):
    # Preferir Google/Mapbox se houver suporte a XYZ genérico
    if provider == "google_hybrid" and XYZTiles is not None:
        template = random.choice(GOOGLE_TEMPLATES)
        return XYZTiles(template, cache=True, user_agent=USER_AGENT), "Google Hybrid"
    if provider in ("google_hybrid", "mapbox_hybrid") and XYZTiles is not None and MAPBOX_TOKEN:
        mb_url = ("https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/"
                  "{z}/{x}/{y}@2x?access_token=" + MAPBOX_TOKEN)
        return XYZTiles(mb_url, cache=True, user_agent=USER_AGENT), "Mapbox Satellite-Streets"
    # Fallback universal (sempre existe)
    return cimgt.OSM(cache=True, user_agent=USER_AGENT), "OpenStreetMap"

def _extract_kml_bytes(payload):
    if payload is None:
        raise ValueError("JSON ausente ou malformado (Content-Type deve ser application/json).")
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        kml_raw = payload[0].get("data")
    elif isinstance(payload, dict):
        kml_raw = payload.get("data")
    else:
        kml_raw = None
    if not kml_raw or not isinstance(kml_raw, str):
        raise ValueError("Campo 'data' (string) não encontrado no JSON.")
    s = kml_raw.strip()
    if s.lower().startswith("data:") and ";base64," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return s.encode("utf-8")

def _read_kml_gdf(kml_bytes):
    import io
    bio = io.BytesIO(kml_bytes)
    try:
        return gpd.read_file(bio, driver="LIBKML")
    except Exception:
        bio.seek(0)
        return gpd.read_file(bio, driver="KML")

def _extent_with_padding(geom, pad_ratio=0.10):
    minx, miny, maxx, maxy = geom.bounds
    dx = (maxx - minx) or 1e-6
    dy = (maxy - miny) or 1e-6
    px, py = dx * pad_ratio, dy * pad_ratio
    return [minx - px, maxx + px, miny - py, maxy + py]

def _zoom_from_lon_span(minx, maxx):
    span = max(1e-9, (maxx - minx))
    z = math.log2(360.0 / span)
    return int(max(1, min(18, round(z))))

def _extract_cod_imovel(gdf):
    for col in gdf.columns:
        if "cod_imovel" in col.lower():
            series = gdf[col].dropna()
            if not series.empty:
                return str(series.iloc[0]).strip()
    return None

def _principal_orientation(geom):
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        max_len, best_seg = -1, None
        for i in range(4):
            x1,y1 = coords[i]; x2,y2 = coords[(i+1)%4]
            L = math.hypot(x2-x1, y2-y1)
            if L > max_len: max_len, best_seg = L, ((x1,y1),(x2,y2))
        (x1,y1),(x2,y2) = best_seg
        return math.degrees(math.atan2(y2-y1, x2-x1))
    except Exception:
        return 0.0

_LOGO_IMG = None
def _load_logo(logo_url: str):
    global _LOGO_IMG, requests
    if _LOGO_IMG is not None:
        return _LOGO_IMG
    try:
        if requests is None:
            import urllib.request
            with urllib.request.urlopen(logo_url, timeout=10) as resp:
                data = resp.read()
        else:
            r = requests.get(logo_url, timeout=10)
            r.raise_for_status()
            data = r.content
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        _LOGO_IMG = img
        return img
    except Exception as e:
        log.warning(f"Falha ao carregar logo: {e}")
        return None

def _add_logo(ax, pil_img, width_px=220):
    if pil_img is None: return
    w, h = pil_img.size
    zoom = width_px / float(w)
    imagebox = OffsetImage(pil_img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0.985, 0.985), xycoords='axes fraction',
                        frameon=False, box_alignment=(1,1), pad=0)
    ax.add_artist(ab)

# --- Endpoints ---
@app.route("/generate-map", methods=["POST"])
def generate_map():
    try:
        provider     = request.args.get("provider", DEFAULT_PROVIDER)
        darken_alpha = float(request.args.get("darken", DEFAULT_DARKEN_ALPHA))
        jpg_quality  = int(request.args.get("jpg_quality", DEFAULT_JPG_QUALITY))
        logo_url     = request.args.get("logo_url", DEFAULT_LOGO_URL)
        show_coords  = request.args.get("coords", "1") in ("1","true","True")

        payload = request.get_json(silent=True)
        kml_bytes = _extract_kml_bytes(payload)

        gdf = _read_kml_gdf(kml_bytes).to_crs(4326)
        if gdf.empty or gdf.geometry.isna().all():
            return jsonify({"error":"KML sem geometria válida."}), 400

        geom = unary_union(gdf.geometry)
        if isinstance(geom, GeometryCollection) and geom.is_empty:
            return jsonify({"error":"Geometria vazia."}), 400

        cod = _extract_cod_imovel(gdf)
        label_text = f"CAR: {cod}" if cod else "CAR"

        tile_src, provider_name = _make_tile_source(provider)

        fig = Figure(figsize=(10,8), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1,1,1, projection=getattr(tile_src, "crs", ccrs.PlateCarree()))

        extent = _extent_with_padding(geom, pad_ratio=0.10)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        zoom = _zoom_from_lon_span(extent[0], extent[1])
        try:
            ax.add_image(tile_src, zoom, interpolation="spline36")
        except Exception as e_tiles:
            log.warning(f"Falha ao adicionar tiles ({provider_name}): {e_tiles}")

        # Máscara fora
        view_rect = box(extent[0], extent[2], extent[1], extent[3])
        try:
            mask_geom = view_rect.difference(geom.buffer(0))
            ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                              facecolor=(0,0,0,max(0,min(1,darken_alpha))),
                              edgecolor='none', zorder=6)
        except Exception as e_mask:
            log.warning(f"Máscara falhou: {e_mask}")

        # Polígono com halo
        coll = ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                 facecolor=(1,1,0,0.25),
                                 edgecolor=(1,1,0,0.95), linewidth=2.2, zorder=7)
        for c in ax.collections[-1:]:
            c.set_path_effects([pe.Stroke(linewidth=3.8, foreground='black'), pe.Normal()])

        # Pino + rótulo
        try:
            pt = geom.representative_point()
            ax.scatter([pt.x],[pt.y], transform=ccrs.PlateCarree(),
                       s=70, zorder=8, marker='o', facecolor=BRAND_COLOR,
                       edgecolor='white', linewidth=1.5)
        except Exception:
            pt = None

        angle = _principal_orientation(geom)
        cx, cy = geom.centroid.x, geom.centroid.y
        txt = ax.text(cx, cy, label_text, transform=ccrs.PlateCarree(),
                      fontsize=10, color='white', rotation=angle, ha='center', va='center', zorder=9)
        txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground='black')])

        # Coordenadas
        if show_coords:
            if pt is not None: lat, lon = pt.y, pt.x
            else:              lat, lon = geom.centroid.y, geom.centroid.x
            fig.text(0.02, 0.06, f"{lat:.5f}, {lon:.5f}", fontsize=8, color='white',
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

        # Atribuição + logo
        ax.set_axis_off()
        fig.text(0.01, 0.01, f"© {provider_name}", fontsize=6, color='white',
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])
        _add_logo(ax, _load_logo(logo_url), width_px=220)

        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=150, bbox_inches="tight", pad_inches=0,
                    quality=max(60, min(95, jpg_quality)), optimize=True, progressive=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg", download_name=(cod or "mapa") + ".jpg")

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log.exception("Erro inesperado")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return {"ok": True}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")))
