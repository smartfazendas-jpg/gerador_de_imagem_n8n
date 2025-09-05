# app.py
import os, io, base64, math, logging, random, json
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects as pe

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from shapely.geometry import GeometryCollection, box, Polygon
from shapely.ops import unary_union
from PIL import Image
from flask import Flask, request, send_file, jsonify

# --- extras opcionais (logo/tiles via HTTP) ---
try:
    import requests
except Exception:
    requests = None

# fastkml (fallback se GDAL não tiver KML/LIBKML)
try:
    from fastkml import kml as fastkml_mod
except Exception:
    fastkml_mod = None

# ------------------ Config ------------------
DEFAULT_PROVIDER = os.getenv("TILE_PROVIDER", "google_hybrid")  # google_hybrid | mapbox_hybrid | osm
GOOGLE_TEMPLATES = [
    "https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
]
MAPBOX_TOKEN_ENV = os.getenv("MAPBOX_TOKEN", "")
USER_AGENT       = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0 (contato@smartfazendas.com.br)")
DEFAULT_LOGO_URL = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_DARKEN_ALPHA = float(os.getenv("DARKEN_ALPHA", "0.45"))
DEFAULT_JPG_QUALITY  = int(os.getenv("JPG_QUALITY", "85"))
BRAND_COLOR = os.getenv("BRAND_COLOR", "#635AFF")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("map-croqui-py")

# ------------------ Compat: ImageTiles pode não existir ------------------
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
    XYZTiles = None

def _make_tile_source(provider: str, mapbox_token: str):
    """Retorna (tile_src, provider_name) quando ImageTiles está disponível; senão, chamador usa o modo XYZ manual."""
    if provider == "google_hybrid" and XYZTiles is not None:
        template = random.choice(GOOGLE_TEMPLATES)
        return XYZTiles(template, cache=True, user_agent=USER_AGENT), "Google Hybrid"
    if provider in ("google_hybrid", "mapbox_hybrid") and XYZTiles is not None and mapbox_token:
        mb_url = ("https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/"
                  "{z}/{x}/{y}@2x?access_token=" + mapbox_token)
        return XYZTiles(mb_url, cache=True, user_agent=USER_AGENT), "Mapbox Satellite-Streets"
    return cimgt.OSM(cache=True, user_agent=USER_AGENT), "OpenStreetMap"

# ------------------ KML loaders ------------------
def _extract_kml_bytes(payload):
    if payload is None:
        raise ValueError("JSON ausente ou malformado (Content-Type: application/json).")
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
    import io, xml.etree.ElementTree as ET
    bio = io.BytesIO(kml_bytes)

    # 1) GDAL drivers (preferível se disponíveis)
    for drv in ("LIBKML", "KML"):
        try:
            bio.seek(0)
            return gpd.read_file(bio, driver=drv)
        except Exception:
            pass

    # 2) fastkml (fallback)
    if fastkml_mod is not None:
        try:
            k = fastkml_mod.KML()
            k.from_string(kml_bytes)

            from shapely.geometry import shape
            geoms_fk, props_fk = [], []

            def iter_children(node):
                feats = getattr(node, "features", None)
                if feats is None:
                    return []
                return list(feats() if callable(feats) else feats)

            def walk(node):
                children = iter_children(node)
                if not children:
                    yield node
                else:
                    for ch in children:
                        yield from walk(ch)

            for f in walk(k):
                geom = getattr(f, "geometry", None)
                if geom:
                    geoms_fk.append(shape(geom.__geo_interface__))
                    p = {}
                    ed = getattr(f, "extended_data", None)
                    if ed:
                        for el in getattr(ed, "elements", []):
                            p[el.name] = el.value
                    props_fk.append(p)

            if geoms_fk:
                return gpd.GeoDataFrame(props_fk, geometry=geoms_fk, crs="EPSG:4326")
        except Exception:
            pass  # cai para XML

    # 3) XML puro (Polygon/LinearRing/coordinates)
    try:
        root = ET.fromstring(kml_bytes)
    except Exception as e:
        raise RuntimeError(f"Falha ao parsear KML (XML): {e}")

    K = "{http://www.opengis.net/kml/2.2}"
    geoms, props = [], []

    for pm in root.findall(".//" + K + "Placemark"):
        # ExtendedData -> SimpleData
        p = {}
        for sd in pm.findall(".//" + K + "SimpleData"):
            name = sd.attrib.get("name")
            if name:
                p[name] = (sd.text or "").strip()

        for poly in pm.findall(".//" + K + "Polygon"):
            coords_el = poly.find(".//" + K + "coordinates")
            if coords_el is None:
                continue
            coords_txt = "".join(coords_el.itertext()).replace("\n", " ").strip()
            if not coords_txt:
                continue
            pts = []
            for token in coords_txt.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0]); lat = float(parts[1])
                        pts.append((lon, lat))
                    except ValueError:
                        continue
            if len(pts) >= 3:
                try:
                    geoms.append(Polygon(pts))
                    props.append(dict(p))
                except Exception:
                    continue

    if not geoms:
        raise RuntimeError("KML sem geometrias (parser XML).")

    return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")

# ------------------ Geometria/estética ------------------
def _extent_with_padding(geom, pad_ratio=0.10):
    minx, miny, maxx, maxy = geom.bounds
    dx = (maxx - minx) or 1e-6
    dy = (maxy - miny) or 1e-6
    px, py = dx * pad_ratio, dy * pad_ratio
    return [minx - px, maxx + px, miny - py, maxy + py]  # [minx,maxx,miny,maxy] em WGS84

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
            req = urllib.request.Request(logo_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
        else:
            r = requests.get(logo_url, headers={"User-Agent": USER_AGENT}, timeout=10)
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

# ------------------ XYZ manual (sem ImageTiles) ------------------
def _lonlat_to_pixel(lon, lat, z, tile_size=256):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * tile_size
    lat = max(-85.05112878, min(85.05112878, lat))
    y = (1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n * tile_size
    return x, y

def _tile_url(provider, x, y, z, mapbox_token):
    if provider == "google_hybrid":
        sub = random.choice(["mt0","mt1","mt2","mt3"])
        return f"https://{sub}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    elif provider == "mapbox_hybrid" and mapbox_token:
        return ("https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/"
                f"{z}/{x}/{y}@2x?access_token={mapbox_token}")
    else:
        sub = random.choice(["a","b","c"])
        return f"https://{sub}.tile.openstreetmap.org/{z}/{x}/{y}.png"

def _fetch_tile(url, timeout=8):
    hdrs = {"User-Agent": USER_AGENT}
    if requests is None:
        import urllib.request
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    r = requests.get(url, headers=hdrs, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def _build_xyz_mosaic(extent_wgs84, z, provider, mapbox_token, tile_size=256):
    """
    extent_wgs84 = [minx_lon, maxx_lon, miny_lat, maxy_lat] (WGS84)
    retorna: (PIL.Image RGBA, extent3857[xmin,xmax,ymin,ymax], provider_name)
    """
    minlon, maxlon, minlat, maxlat = extent_wgs84

    # pixels no nível z
    px_min, py_max = _lonlat_to_pixel(minlon, minlat, z, tile_size)
    px_max, py_min = _lonlat_to_pixel(maxlon, maxlat, z, tile_size)

    # tiles a cobrir
    tmin_x, tmax_x = int(px_min // tile_size), int(px_max // tile_size)
    tmin_y, tmax_y = int(py_min // tile_size), int(py_max // tile_size)

    cols = tmax_x - tmin_x + 1
    rows = tmax_y - tmin_y + 1
    mosaic = Image.new("RGBA", (cols * tile_size, rows * tile_size), (0,0,0,0))

    actual_provider = provider
    for ty in range(tmin_y, tmax_y + 1):
        for tx in range(tmin_x, tmax_x + 1):
            url = _tile_url(provider, tx, ty, z, mapbox_token)
            try:
                tile = _fetch_tile(url)
            except Exception:
                # fallback para OSM se falhar
                url = _tile_url("osm", tx, ty, z, mapbox_token)
                tile = _fetch_tile(url)
                actual_provider = "osm"
            mosaic.paste(tile, ((tx - tmin_x) * tile_size, (ty - tmin_y) * tile_size))

    # recorta para o retângulo de interesse
    crop_left   = int(px_min - tmin_x * tile_size)
    crop_top    = int(py_min - tmin_y * tile_size)
    crop_right  = int(px_max - tmin_x * tile_size)
    crop_bottom = int(py_max - tmin_y * tile_size)
    crop = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    # extent em EPSG:3857 equivalente
    def _merc_x(lon):
        R = 6378137.0
        return math.radians(lon) * R
    def _merc_y(lat):
        R = 6378137.0
        lat = max(-85.05112878, min(85.05112878, lat))
        return R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))

    xmin_merc, ymax_merc = _merc_x(minlon), _merc_y(maxlat)
    xmax_merc, ymin_merc = _merc_x(maxlon), _merc_y(minlat)
    extent3857 = [xmin_merc, xmax_merc, ymin_merc, ymax_merc]

    prov_name = {"google_hybrid":"Google Hybrid",
                 "mapbox_hybrid":"Mapbox Satellite-Streets",
                 "osm":"OpenStreetMap"}.get(actual_provider, actual_provider)
    return crop, extent3857, prov_name

# ------------------ Endpoints ------------------
@app.post("/generate-map")
def generate_map():
    try:
        provider     = request.args.get("provider", DEFAULT_PROVIDER).strip()
        darken_alpha = float(request.args.get("darken", DEFAULT_DARKEN_ALPHA))
        jpg_quality  = int(request.args.get("jpg_quality", DEFAULT_JPG_QUALITY))
        logo_url     = request.args.get("logo_url", DEFAULT_LOGO_URL)
        show_coords  = request.args.get("coords", "1").lower() in ("1","true","yes")
        mapbox_token = request.args.get("mapbox_token", MAPBOX_TOKEN_ENV)

        payload  = request.get_json(silent=True)
        kml_byt  = _extract_kml_bytes(payload)
        gdf      = _read_kml_gdf(kml_byt).to_crs(4326)

        if gdf.empty or gdf.geometry.isna().all():
            return jsonify({"error":"KML sem geometria válida."}), 400

        geom = unary_union(gdf.geometry)
        if isinstance(geom, GeometryCollection) and geom.is_empty:
            return jsonify({"error":"Geometria vazia."}), 400

        cod = _extract_cod_imovel(gdf)
        label_text = f"CAR: {cod}" if cod else "CAR"

        # Figura
        fig = Figure(figsize=(10,8), dpi=150)
        canvas = FigureCanvas(fig)

        # Extent/zoom
        extent84 = _extent_with_padding(geom, pad_ratio=0.10)  # [minx,maxx,miny,maxy]
        zoom = _zoom_from_lon_span(extent84[0], extent84[1])

        # Fundo (ImageTiles se disponível; caso contrário, XYZ manual)
        if HAS_IMGTILES:
            tile_src, provider_name = _make_tile_source(provider, mapbox_token)
            # use EPSG:3857 para encaixar melhor tiles webmercator
            ax = fig.add_subplot(1,1,1, projection=getattr(tile_src, "crs", ccrs.epsg(3857)))
            ax.set_extent(extent84, crs=ccrs.PlateCarree())
            try:
                ax.add_image(tile_src, zoom, interpolation="spline36")
            except Exception as e_tiles:
                log.warning(f"Falha add_image ({provider_name}): {e_tiles}. Usando XYZ manual.")
                ax = fig.add_subplot(1,1,1, projection=ccrs.epsg(3857))
                img_bg, wm_extent, provider_name = _build_xyz_mosaic(extent84, zoom, provider, mapbox_token)
                ax.set_extent(wm_extent, crs=ccrs.epsg(3857))
                ax.imshow(img_bg, extent=wm_extent, transform=ccrs.epsg(3857), origin="upper")
        else:
            ax = fig.add_subplot(1,1,1, projection=ccrs.epsg(3857))
            img_bg, wm_extent, provider_name = _build_xyz_mosaic(extent84, zoom, provider, mapbox_token)
            ax.set_extent(wm_extent, crs=ccrs.epsg(3857))
            ax.imshow(img_bg, extent=wm_extent, transform=ccrs.epsg(3857), origin="upper")

        # Máscara fora do polígono (em WGS84)
        try:
            view_rect = box(extent84[0], extent84[2], extent84[1], extent84[3])
            mask_geom = view_rect.difference(geom.buffer(0))
            ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                              facecolor=(0,0,0,max(0,min(1,darken_alpha))),
                              edgecolor='none', zorder=6)
        except Exception as e_mask:
            log.warning(f"Máscara falhou: {e_mask}")

        # Polígono com halo
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
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

        # --- Render -> PNG -> JPEG (compat universal) ---
        fig.tight_layout(pad=0)

        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf_png.seek(0)

        img = Image.open(buf_png).convert("RGB")
        buf_jpg = io.BytesIO()
        q = max(60, min(95, jpg_quality))
        img.save(buf_jpg, format="JPEG", quality=q, optimize=True, progressive=True)
        buf_jpg.seek(0)

        return send_file(buf_jpg,
                         mimetype="image/jpeg",
                         download_name=(cod or "mapa") + ".jpg")

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log.exception("Erro inesperado")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    import cartopy, matplotlib as mpl, shapely
    diag = request.args.get("diag", "0").lower() in ("1","true","yes")
    mapbox_token = request.args.get("mapbox_token") or MAPBOX_TOKEN_ENV
    info = {
        "ok": True,
        "versions": {
            "cartopy": getattr(cartopy, "__version__", "unknown"),
            "matplotlib": getattr(mpl, "__version__", "unknown"),
            "geopandas": getattr(gpd, "__version__", "unknown"),
            "shapely": getattr(shapely, "__version__", "unknown"),
        },
        "has_imgtiles": hasattr(cimgt, "ImageTiles"),
        "tile_default": DEFAULT_PROVIDER,
    }
    if diag:
        if HAS_IMGTILES:
            src, name = _make_tile_source(DEFAULT_PROVIDER, mapbox_token)
            info["tile_provider_selected"] = name
            info["tile_src_type"] = src.__class__.__name__
        else:
            info["tile_provider_selected"] = "XYZ-manual"
            info["tile_src_type"] = "PIL-mosaic"
        info["mapbox_token_present"] = bool(mapbox_token)
    return info, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")))
