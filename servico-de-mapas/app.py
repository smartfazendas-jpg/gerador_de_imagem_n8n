# app.py
import io, os, math, logging
from datetime import datetime

import xml.etree.ElementTree as ET

import requests
from PIL import Image

import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from flask import Flask, request, jsonify, send_file

# Tenta carregar fastkml para fallback de parsing KML
try:
    from fastkml import kml as fastkml
    HAS_FASTKML = True
except Exception:
    HAS_FASTKML = False

# --------------------
# Configurações globais
# --------------------
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("map-croqui")

DEFAULT_FIG_ASPECT = 1.33  # 4:3 padrão (paisagem)
DEFAULT_DPI = 150

DEFAULT_POLY_ALPHA = 0.28
DEFAULT_POLY_EDGE = "#ffd54f"   # amarelo borda
DEFAULT_POLY_FACE = "#ffeb3b"   # amarelo face
DEFAULT_EDGE_WIDTH = 3.0

DEFAULT_DARKEN = 0.35
DEFAULT_JPG_QUALITY = 82

LOGO_URL_DEFAULT = "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png"

# Token do Mapbox (para Google Hybrid fallback via estilo satellite-streets)
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "").strip()

# ------------
# Tiles Mapbox
# ------------
class MapboxTiles(cimgt.GoogleTiles):
    """
    Usa a base do GoogleTiles para gerenciar quadros, mas com URL XYZ customizada (Mapbox).
    """
    def __init__(self, url_template, cache=False):
        self._url_template = url_template
        super().__init__()
        self.desired_tile_form = "RGB"
        self.cache = cache

    def _image_url(self, tile):
        x, y, z = tile
        return self._url_template.format(x=x, y=y, z=z)

def get_tile_provider(name: str):
    name = (name or "").lower()
    if name in ("google_hybrid", "mapbox_satellite_streets"):
        if not MAPBOX_TOKEN:
            LOGGER.warning("MAPBOX_TOKEN ausente -> fallback para OSM.")
            return cimgt.OSM(), "OpenStreetMap"
        url = (
            "https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/"
            "tiles/256/{z}/{x}/{y}?access_token=" + MAPBOX_TOKEN
        )
        return MapboxTiles(url), "Google Hybrid"
    # fallback
    return cimgt.OSM(), "OpenStreetMap"

# -------------------------
# Utilitários de geometria
# -------------------------
def safe_unary_union(gdf):
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    merged = unary_union(geoms)
    try:
        return merged.buffer(0)
    except Exception:
        return merged

def pad_extent(bounds, pad_frac=0.10):
    # bounds shapely: (minx, miny, maxx, maxy) -> retorna [minx, maxx, miny, maxy]
    minx, miny, maxx, maxy = bounds
    dx, dy = (maxx - minx), (maxy - miny)
    if dx <= 0 or dy <= 0:
        return [minx, maxx, miny, maxy]
    return [minx - dx*pad_frac, maxx + dx*pad_frac, miny - dy*pad_frac, maxy + dy*pad_frac]

# --- Conversões lon/lat <-> WebMercator (m) para forçar aspecto corretamente ---
def _mx(lon):  # lon -> x
    return 6378137.0 * math.radians(lon)

def _my(lat):  # lat -> y
    lat = max(-85.05112878, min(85.05112878, lat))
    return 6378137.0 * math.log(math.tan(math.pi/4 + math.radians(lat)/2))

def _lon(mx):  # x -> lon
    return math.degrees(mx / 6378137.0)

def _lat(my):  # y -> lat
    return math.degrees(2*math.atan(math.exp(my/6378137.0)) - math.pi/2)

def enforce_extent_aspect(ext84, target_aspect):
    """
    Expande simetricamente o extent WGS84 para atingir L/H = target_aspect em 3857.
    ext84 = [minx, maxx, miny, maxy] (lon/lat)
    """
    minlon, maxlon, minlat, maxlat = ext84
    xmin, xmax = _mx(minlon), _mx(maxlon)
    ymin, ymax = _my(minlat), _my(maxlat)
    w, h = (xmax - xmin), (ymax - ymin)
    if w <= 0 or h <= 0:
        return ext84
    cur = w / h
    cx, cy = (xmin + xmax)/2.0, (ymin + ymax)/2.0
    if cur < target_aspect:
        # muito alto -> alarga largura
        new_w = target_aspect * h
        xmin, xmax = cx - new_w/2.0, cx + new_w/2.0
    elif cur > target_aspect:
        # muito largo -> aumenta altura
        new_h = w / target_aspect
        ymin, ymax = cy - new_h/2.0, cy + new_h/2.0
    return [_lon(xmin), _lon(xmax), _lat(ymin), _lat(ymax)]

def set_figure_aspect(fig: Figure, aspect: float):
    """Define o tamanho da figura em polegadas para manter aspecto alvo (paisagem)."""
    H = 9.0
    W = max(8.0, min(20.0, H * aspect))
    fig.set_size_inches(W, H, forward=True)

def estimate_zoom(extent84, base_width_px=1600):
    """Heurística simples para zoom WebMercator a partir do span em longitude."""
    minx, maxx, _, _ = extent84
    span_deg = max(0.0005, maxx - minx)
    z = math.log2((360.0 * base_width_px) / (256.0 * span_deg))
    return int(max(7, min(18, round(z))))

# -------------------------
# KML -> GeoDataFrame (robusto)
# -------------------------
def gdf_from_kml_string(kml_str: str) -> gpd.GeoDataFrame:
    """
    Lê KML e devolve GeoDataFrame (EPSG:4326).
    1) GDAL/Fiona (driver KML)
    2) fastkml (robusto, lida com .features método ou lista)
    3) XML manual AGNÓSTICO de namespace (varre por localname: Polygon/LinearRing/coordinates)
    """
    # 1) GDAL/Fiona
    try:
        gdf = gpd.read_file(io.BytesIO(kml_str.encode("utf-8")), driver="KML")
        if not gdf.empty:
            return gdf.to_crs(4326)
    except Exception as e:
        msg = str(e).lower()
        if "unsupported driver" not in msg and "kml" not in msg:
            LOGGER.info(f"read_file(driver='KML') falhou: {e}")

    # 2) fastkml
    if HAS_FASTKML:
        try:
            from shapely.geometry import Polygon, MultiPolygon
            k = fastkml.KML()
            k.from_string(kml_str.encode("utf-8"))

            def iter_children(obj):
                feats = getattr(obj, "features", None)
                if callable(feats):
                    try:
                        for f in feats():
                            yield f
                    except TypeError:
                        pass
                if isinstance(feats, (list, tuple)):
                    for f in feats:
                        yield f
                feats2 = getattr(obj, "_features", None)
                if isinstance(feats2, (list, tuple)):
                    for f in feats2:
                        yield f

            geoms = []
            stack = list(iter_children(k))  # nada de k.features() direto
            while stack:
                f = stack.pop()
                geom = getattr(f, "geometry", None)
                if geom is not None:
                    geoms.append(geom)
                for ch in iter_children(f):
                    stack.append(ch)

            polys = []
            for g in geoms:
                if isinstance(g, (Polygon, MultiPolygon)):
                    polys.append(g)
                else:
                    geoms_attr = getattr(g, "geoms", None)
                    if geoms_attr:
                        for sg in geoms_attr:
                            if isinstance(sg, (Polygon, MultiPolygon)):
                                polys.append(sg)

            if polys:
                return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
        except Exception as e:
            LOGGER.info(f"fastkml fallback falhou: {e}")

    # 3) XML manual agnóstico de namespace
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon, MultiPolygon

    def strip_ns(tag):
        return tag.split('}')[-1] if isinstance(tag, str) else tag

    def parse_coords(text):
        coords = []
        if not text:
            return coords
        for token in text.replace("\n", " ").replace("\t", " ").split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    lon = float(parts[0]); lat = float(parts[1])
                    coords.append((lon, lat))
                except ValueError:
                    continue
        return coords

    def find_first_local(parent, localname):
        for el in parent.iter():
            if strip_ns(el.tag) == localname:
                return el
        return None

    def find_all_local(parent, localname):
        for el in parent.iter():
            if strip_ns(el.tag) == localname:
                yield el

    try:
        root = ET.fromstring(kml_str.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"KML inválido (XML). {e}")

    polys = []
    # Varre TODOS os <Polygon> independente de prefixo/namespace
    for poly in root.iter():
        if strip_ns(poly.tag) != "Polygon":
            continue

        # OUTER
        outer = find_first_local(poly, "outerBoundaryIs")
        outer_coords = []
        if outer is not None:
            lr = find_first_local(outer, "LinearRing")
            if lr is not None:
                coords_el = find_first_local(lr, "coordinates")
                if coords_el is not None:
                    outer_coords = parse_coords(coords_el.text)

        # INNERS
        holes = []
        for ib in find_all_local(poly, "innerBoundaryIs"):
            lr = find_first_local(ib, "LinearRing")
            if lr is not None:
                coords_el = find_first_local(lr, "coordinates")
                if coords_el is not None:
                    ic = parse_coords(coords_el.text)
                    if len(ic) >= 3:
                        # fecha o anel, se necessário
                        if ic[0] != ic[-1]:
                            ic.append(ic[0])
                        holes.append(ic)

        if len(outer_coords) >= 3:
            if outer_coords[0] != outer_coords[-1]:
                outer_coords.append(outer_coords[0])
            try:
                p = Polygon(outer_coords, holes)
                if p.is_valid and not p.is_empty and p.area > 0:
                    polys.append(p)
            except Exception:
                # Ignora outer inválido
                pass

    if not polys:
        # Tenta caso de MultiGeometry contendo <coordinates> direto (raríssimo)
        # Como fallback extremo: pegar qualquer <coordinates> e formar um polígono único
        coords_texts = [el.text for el in root.iter() if strip_ns(el.tag) == "coordinates" and el.text]
        for txt in coords_texts:
            pts = parse_coords(txt)
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                try:
                    p = Polygon(pts)
                    if p.is_valid and not p.is_empty and p.area > 0:
                        polys.append(p)
                        break
                except Exception:
                    pass

    if not polys:
        raise RuntimeError("KML sem geometrias poligonais (parser XML agnóstico).")

    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")





# --------------------
# Render helpers (plot)
# --------------------
def draw_logo_figimage(fig: Figure, logo_url: str, scale: float = 0.18, margin_px: int = 30):
    """
    Desenha a logo no canto superior direito em coordenadas da FIGURA (fora do eixo),
    então não sofre o escurecimento aplicado no eixo/cartopy.
    """
    try:
        resp = requests.get(logo_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        LOGGER.warning(f"Logo falhou: {e}")
        return

    # dimensiona pela largura da figura
    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height()
    target_w = int(fw * scale)
    ratio = target_w / img.width
    target_h = int(img.height * ratio)
    img = img.resize((target_w, target_h), Image.LANCZOS)

    # topo-direito
    x = fw - target_w - margin_px
    y = fh - target_h - margin_px
    fig.figimage(img, xo=x, yo=y, zorder=50)

def draw_corner_label(ax, text, xy_data, crs, fontsize=10):
    ax.text(
        xy_data[0], xy_data[1], text,
        transform=crs,
        color="#111",
        fontsize=fontsize, fontweight="bold",
        ha="left", va="bottom",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)]
    )

# ----------
# Flask app
# ----------
app = Flask(__name__)

@app.get("/health")
def health():
    diag = request.args.get("diag", "0") == "1"
    provider, prov_name = get_tile_provider("google_hybrid")
    res = {
        "ok": True,
        "tile_default": "google_hybrid",
        "tile_src_type": prov_name
    }
    if diag:
        res.update({
            "versions": {
                "geopandas": gpd.__version__,
                "matplotlib": matplotlib.__version__,
                "cartopy": getattr(cimgt, "__version__", "unknown"),
                "fastkml": "ok" if HAS_FASTKML else "missing"
            },
            "has_imgtiles": hasattr(cimgt, "OSM"),
            "tile_provider_selected": prov_name,
        })
    return jsonify(res)

@app.post("/generate-map")
def generate_map():
    try:
        # -------------------
        # Entrada: JSON com KML em "data" (string), ou lista [{data: "..."}]
        # -------------------
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "JSON ausente"}), 400

        if isinstance(payload, list):
            kml_str = (payload[0] or {}).get("data")
        else:
            kml_str = payload.get("data")

        if not kml_str:
            return jsonify({"error": "Campo 'data' (KML) não encontrado"}), 400

        # ---- robusto: lê KML com fallback fastkml
        gdf = gdf_from_kml_string(kml_str)
        if gdf.empty or gdf.geometry.is_empty.all():
            return jsonify({"error": "KML sem geometrias"}), 400

        gdf = gdf.to_crs(4326)
        geom = safe_unary_union(gdf)
        if geom is None or geom.is_empty:
            return jsonify({"error": "Geometria inválida"}), 400

        # -------------------
        # Query params (opcionais)
        # -------------------
        q = request.args
        provider_name = q.get("provider", "google_hybrid")

        darken = max(0.0, min(0.9, float(q.get("darken", DEFAULT_DARKEN))))
        poly_alpha = max(0.0, min(1.0, float(q.get("poly_alpha", DEFAULT_POLY_ALPHA))))
        edge_w = max(0.5, min(8.0, float(q.get("edge_width", DEFAULT_EDGE_WIDTH))))

        label_text = q.get("label", "").strip()

        jpg_quality = int(q.get("jpg_quality", DEFAULT_JPG_QUALITY))
        jpg_quality = min(95, max(50, jpg_quality))

        # pino por coordenadas
        lat = q.get("lat", None)
        lon = q.get("lon", None)
        pin = None
        if lat is not None and lon is not None:
            try:
                lat = float(lat); lon = float(lon)
                pin = Point(lon, lat)
            except Exception:
                pin = None

        # logo
        logo_url = q.get("logo_url", LOGO_URL_DEFAULT)
        logo_scale = float(q.get("logo_scale", 0.16))

        # -------------------
        # Extent alvo + aspecto 4:3 fixo (paisagem)
        # -------------------
        b = geom.bounds  # (minx, miny, maxx, maxy)
        extent84 = pad_extent(b, pad_frac=0.10)  # [minx,maxx,miny,maxy]
        extent84 = [extent84[0], extent84[1], extent84[2], extent84[3]]
        extent84 = enforce_extent_aspect(extent84, DEFAULT_FIG_ASPECT)

        # -------------------
        # Figura + eixo sem margens/brancos
        # -------------------
        fig = Figure(dpi=DEFAULT_DPI)
        set_figure_aspect(fig, DEFAULT_FIG_ASPECT)

        proj = ccrs.WebMercator()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_position([0, 0, 1, 1])

        # tiles
        tiles, prov_label = get_tile_provider(provider_name)
        ax.add_image(tiles, estimate_zoom(extent84))
        ax.set_extent(extent84, crs=ccrs.PlateCarree())

        # -------------------
        # Escurecer APENAS fora do polígono
        # -------------------
        full_rect = box(extent84[0], extent84[2], extent84[1], extent84[3])
        outside = full_rect.difference(geom)
        if not outside.is_empty and darken > 0:
            ax.add_geometries(
                [outside],
                crs=ccrs.PlateCarree(),
                facecolor="black",
                edgecolor="none",
                alpha=darken,
                zorder=5
            )

        # -------------------
        # Imóvel (amarelo ajustável)
        # -------------------
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor=DEFAULT_POLY_FACE,
            edgecolor=DEFAULT_POLY_EDGE,
            linewidth=edge_w,
            alpha=poly_alpha,
            zorder=10
        )

        # etiqueta do imóvel (fallback por colunas conhecidas)
        if not label_text:
            for col in ["cod_imovel", "CAR", "car", "cod", "name", "Nome"]:
                if col in gdf.columns and isinstance(gdf.iloc[0][col], str):
                    label_text = gdf.iloc[0][col]
                    break

        if label_text:
            c = geom.centroid
            ax.text(
                c.x, c.y, label_text,
                transform=ccrs.PlateCarree(),
                color="#212121",
                fontsize=16, fontweight="bold",
                ha="center", va="center",
                zorder=15,
                path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.9)]
            )

        # pino e rótulo da coordenada
        if pin is not None:
            ax.plot(pin.x, pin.y, marker="o", markersize=7,
                    markerfacecolor="#2e7dff", markeredgecolor="white",
                    linewidth=0.8, transform=ccrs.PlateCarree(), zorder=20)
            draw_corner_label(
                ax,
                f"{lat:.5f}, {lon:.5f}",
                (pin.x, pin.y),
                ccrs.PlateCarree(),
                fontsize=12
            )

        # rodapé com crédito do provedor
        ll = (extent84[0] + 0.005*(extent84[1]-extent84[0]),
              extent84[2] + 0.008*(extent84[3]-extent84[2]))
        draw_corner_label(ax, f"© {prov_label}", ll, ccrs.PlateCarree(), fontsize=8)

        # Logo no topo direito (fora do eixo -> não escurece)
        draw_logo_figimage(fig, logo_url=logo_url, scale=logo_scale)

        # -------------------
        # Exportar JPEG
        # -------------------
        # 1) salva PNG no buffer
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", facecolor="white")
        plt.close(fig)
        png_buf.seek(0)

        # 2) converte para JPEG
        im = Image.open(png_buf).convert("RGB")
        jpg_buf = io.BytesIO()
        im.save(jpg_buf, format="JPEG", quality=jpg_quality, optimize=True, progressive=True)
        jpg_buf.seek(0)

        filename = f"mapa_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        return send_file(jpg_buf, mimetype="image/jpeg", as_attachment=False, download_name=filename)

    except Exception as e:
        LOGGER.exception("Erro no processamento")
        # Devolve msg curta, mas com detalhe útil
        return jsonify({"error": str(e)}), 500

# -------------
# Main (local)
# -------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
