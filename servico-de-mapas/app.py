# app.py
import os, io, base64, math, logging, random, json
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from shapely.geometry import GeometryCollection, box, Polygon, Point
from shapely.ops import unary_union
from PIL import Image
from flask import Flask, request, send_file, jsonify

# HTTP opcional (logo/tiles)
try:
    import requests
except Exception:
    requests = None

# fastkml opcional
try:
    from fastkml import kml as fastkml_mod
except Exception:
    fastkml_mod = None

# ------------------ Config ------------------
DEFAULT_PROVIDER = os.getenv("TILE_PROVIDER", "google_hybrid")
GOOGLE_TEMPLATES = [
    "https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
]
MAPBOX_TOKEN_ENV = os.getenv("MAPBOX_TOKEN", "")
USER_AGENT       = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0 (contato@smartfazendas.com.br)")
DEFAULT_LOGO_URL = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_DARKEN_ALPHA = float(os.getenv("DARKEN_ALPHA", "0.55"))  # máscara fora do polígono (escurecer só o fundo)
DEFAULT_POLY_ALPHA   = float(os.getenv("POLY_ALPHA", "0.28"))    # amarelo dentro do polígono
DEFAULT_JPG_QUALITY  = int(os.getenv("JPG_QUALITY", "82"))
BRAND_COLOR = os.getenv("BRAND_COLOR", "#346DFF")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("map-croqui-py")

# ------------------ Cartopy compat ------------------
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
    if provider == "google_hybrid" and XYZTiles is not None:
        template = random.choice(GOOGLE_TEMPLATES)
        return XYZTiles(template, cache=True, user_agent=USER_AGENT), "Google Hybrid"
    if provider in ("mapbox_hybrid", "google_hybrid") and XYZTiles is not None and mapbox_token:
        mb_url = ("https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/"
                  "{z}/{x}/{y}@2x?access_token=" + mapbox_token)
        return XYZTiles(mb_url, cache=True, user_agent=USER_AGENT), "Mapbox Satellite-Streets"
    return cimgt.OSM(cache=True, user_agent=USER_AGENT), "OpenStreetMap"

# ------------------ KML ------------------
def _extract_kml_bytes(payload):
    if payload is None:
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        kml_raw = payload[0].get("data")
    elif isinstance(payload, dict):
        kml_raw = payload.get("data")
    else:
        kml_raw = None
    if not kml_raw or not isinstance(kml_raw, str):
        return None
    s = kml_raw.strip()
    if s.lower().startswith("data:") and ";base64," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return s.encode("utf-8")

def _read_kml_gdf(kml_bytes):
    import io, xml.etree.ElementTree as ET
    if kml_bytes is None:
        return None
    bio = io.BytesIO(kml_bytes)

    for drv in ("LIBKML", "KML"):
        try:
            bio.seek(0)
            gdf = gpd.read_file(bio, driver=drv)
            return gdf
        except Exception:
            pass

    if fastkml_mod is not None:
        try:
            k = fastkml_mod.KML(); k.from_string(kml_bytes)
            from shapely.geometry import shape
            geoms, props = [], []
            def children(n):
                f = getattr(n, "features", None)
                return list(f() if callable(f) else (f or []))
            def walk(n):
                ch = children(n)
                if not ch: yield n
                else:
                    for c in ch: yield from walk(c)
            for f in walk(k):
                geom = getattr(f, "geometry", None)
                if geom:
                    geoms.append(shape(geom.__geo_interface__))
                    p = {}
                    ed = getattr(f, "extended_data", None)
                    if ed:
                        for el in getattr(ed, "elements", []):
                            p[el.name] = el.value
                    props.append(p)
            if geoms:
                return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
        except Exception:
            pass

    try:
        root = ET.fromstring(kml_bytes)
    except Exception:
        return None
    K = "{http://www.opengis.net/kml/2.2}"
    geoms, props = [], []
    for pm in root.findall(".//" + K + "Placemark"):
        p = {}
        for sd in pm.findall(".//" + K + "SimpleData"):
            name = sd.attrib.get("name")
            if name: p[name] = (sd.text or "").strip()
        for poly in pm.findall(".//" + K + "Polygon"):
            coords_el = poly.find(".//" + K + "coordinates")
            if coords_el is None: continue
            coords_txt = "".join(coords_el.itertext()).replace("\n"," ").strip()
            pts = []
            for tok in coords_txt.split():
                pr = tok.split(",")
                if len(pr)>=2:
                    try: pts.append((float(pr[0]), float(pr[1])))
                    except: pass
            if len(pts)>=3:
                try:
                    geoms.append(Polygon(pts)); props.append(dict(p))
                except: pass
    if not geoms: return None
    return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")

# ------------------ Geometria/estética ------------------
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

def _extent_from_center_zoom(lon, lat, zoom):
    lon_span = 360.0 / (2 ** zoom)
    lat_span = lon_span / max(0.2, math.cos(math.radians(lat)))
    return [lon - lon_span/2, lon + lon_span/2, lat - lat_span/2, lat + lat_span/2]

def _extract_cod_imovel(gdf):
    if gdf is None: return None
    for col in gdf.columns:
        if "cod_imovel" in col.lower():
            s = gdf[col].dropna()
            if not s.empty: return str(s.iloc[0]).strip()
    return None

def _principal_orientation(geom):
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        max_len, best = -1, None
        for i in range(4):
            x1,y1 = coords[i]; x2,y2 = coords[(i+1)%4]
            L = math.hypot(x2-x1, y2-y1)
            if L>max_len: max_len, best = L, ((x1,y1),(x2,y2))
        (x1,y1),(x2,y2) = best
        return math.degrees(math.atan2(y2-y1, x2-x1))
    except Exception:
        return 0.0

_LOGO_IMG = None
def _load_logo(url: str):
    global _LOGO_IMG, requests
    if _LOGO_IMG is not None: return _LOGO_IMG
    try:
        if requests is None:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as r: data = r.read()
        else:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            r.raise_for_status(); data = r.content
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        _LOGO_IMG = img; return img
    except Exception as e:
        log.warning(f"Falha logo: {e}")
        return None

def _add_logo(ax, pil_img, width_px=220, zorder=50):
    """Adiciona logo SEM ser afetada pelo darken (zorder alto)."""
    if pil_img is None: return
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    w,h = pil_img.size
    zoom = width_px/float(w)
    imagebox = OffsetImage(pil_img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0.985, 0.985), xycoords='axes fraction',
                        frameon=False, box_alignment=(1,1), pad=0, zorder=zorder)
    ax.add_artist(ab)

def hide_axes(ax):
    """Esconde contornos/bordas do eixo (compatível com várias versões)."""
    try:
        ax.set_axis_off()
    except Exception:
        pass
    try:
        op = getattr(ax, "outline_patch", None)
        if op is not None:
            op.set_visible(False)
    except Exception:
        pass
    try:
        for sp in getattr(ax, "spines", {}).values():
            sp.set_visible(False)
    except Exception:
        pass
    try:
        if hasattr(ax, "patch") and ax.patch is not None:
            ax.patch.set_visible(False)
    except Exception:
        pass

# ---- Ajuste de aspecto para eliminar laterais brancas ----
def _match_figure_to_extent(fig, extent84, enable=True):
    if not enable or extent84 is None:
        return
    def _mx(lon):  # lon/lat -> WebMercator (m)
        return 6378137.0 * math.radians(lon)
    def _my(lat):
        lat = max(-85.05112878, min(85.05112878, lat))
        return 6378137.0 * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    minlon, maxlon, minlat, maxlat = extent84
    w = _mx(maxlon) - _mx(minlon)
    h = _my(maxlat) - _my(minlat)
    if w <= 0 or h <= 0:
        return
    target_aspect = w / h
    h_in = fig.get_figheight()
    w_in = h_in * target_aspect
    w_in = max(6.0, min(14.0, w_in))
    fig.set_size_inches(w_in, h_in, forward=True)

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
    r = requests.get(url, headers=hdrs, timeout=timeout); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def _build_xyz_mosaic(extent_wgs84, z, provider, mapbox_token, tile_size=256):
    minlon, maxlon, minlat, maxlat = extent_wgs84
    px_min, py_max = _lonlat_to_pixel(minlon, minlat, z, tile_size)
    px_max, py_min = _lonlat_to_pixel(maxlon, maxlat, z, tile_size)

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
                url = _tile_url("osm", tx, ty, z, mapbox_token)
                tile = _fetch_tile(url)
                actual_provider = "osm"
            mosaic.paste(tile, ((tx - tmin_x) * tile_size, (ty - tmin_y) * tile_size))

    crop_left   = int(px_min - tmin_x * tile_size)
    crop_top    = int(py_min - tmin_y * tile_size)
    crop_right  = int(px_max - tmin_x * tile_size)
    crop_bottom = int(py_max - tmin_y * tile_size)
    crop = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    def _merc_x(lon): R=6378137.0; return math.radians(lon)*R
    def _merc_y(lat):
        R=6378137.0; lat=max(-85.05112878,min(85.05112878,lat))
        return R*math.log(math.tan(math.pi/4+math.radians(lat)/2))

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
        q = request.args
        provider     = q.get("provider", DEFAULT_PROVIDER).strip()
        darken_alpha = float(q.get("darken", DEFAULT_DARKEN_ALPHA))        # máscara externa (só fundo)
        poly_alpha   = float(q.get("poly_alpha", DEFAULT_POLY_ALPHA))      # amarelo dentro
        jpg_quality  = int(q.get("jpg_quality", DEFAULT_JPG_QUALITY))
        logo_url     = q.get("logo_url", DEFAULT_LOGO_URL)
        show_coords  = q.get("coords", "1").lower() in ("1","true","yes")
        mapbox_token = q.get("mapbox_token", MAPBOX_TOKEN_ENV)
        fit_aspect   = q.get("fit_aspect", "1").lower() in ("1","true","yes")

        # Pino por coordenadas
        pin_lat = q.get("pin_lat"); pin_lon = q.get("pin_lon")
        pin_lat = float(pin_lat) if pin_lat is not None else None
        pin_lon = float(pin_lon) if pin_lon is not None else None
        pin_text = q.get("pin_text")
        pin_color = q.get("pin_color", BRAND_COLOR)
        pin_zoom = q.get("pin_zoom"); pin_zoom = int(pin_zoom) if pin_zoom else None

        payload  = request.get_json(silent=True)
        kml_byt  = _extract_kml_bytes(payload)
        gdf      = _read_kml_gdf(kml_byt)
        if gdf is not None and not gdf.empty:
            gdf = gdf.to_crs(4326)
            geom = unary_union(gdf.geometry)
            if isinstance(geom, GeometryCollection) or geom.is_empty:
                geom = None
        else:
            geom = None

        cod = _extract_cod_imovel(gdf)
        label_text = f"CAR: {cod}" if cod else "CAR"

        # ===== Figura =====
        fig = Figure(figsize=(10, 8), dpi=150); fig.set_facecolor("white")
        canvas = FigureCanvas(fig)

        # ===== Extent/zoom =====
        if geom is not None:
            extent84 = _extent_with_padding(geom, pad_ratio=0.10)
            zoom = _zoom_from_lon_span(extent84[0], extent84[1])
            if pin_lat is not None and pin_lon is not None:
                minx, maxx, miny, maxy = extent84
                minx = min(minx, pin_lon); maxx = max(maxx, pin_lon)
                miny = min(miny, pin_lat); maxy = max(maxy, pin_lat)
                extent84 = [minx, maxx, miny, maxy]
        elif pin_lat is not None and pin_lon is not None:
            zoom = min(18, max(1, pin_zoom if pin_zoom is not None else 15))
            extent84 = _extent_from_center_zoom(pin_lon, pin_lat, zoom)
        else:
            return jsonify({"error": "Forneça KML em JSON['data'] ou 'pin_lat' e 'pin_lon' na query."}), 400

        # Ajuste do aspecto para eliminar laterais brancas
        _match_figure_to_extent(fig, extent84, enable=fit_aspect)

        # ===== Eixo/tiles =====
        if HAS_IMGTILES:
            tile_src, provider_name = _make_tile_source(provider, mapbox_token)
            ax = fig.add_axes([0, 0, 1, 1],
                              projection=getattr(tile_src, "crs", ccrs.epsg(3857)))
            ax.set_extent(extent84, crs=ccrs.PlateCarree())
            try:
                ax.add_image(tile_src, zoom, interpolation="spline36")  # zorder padrão ~1
            except Exception as e_tiles:
                log.warning(f"Falha add_image ({provider_name}): {e_tiles}. Usando XYZ manual.")
                ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.epsg(3857))
                img_bg, wm_extent, provider_name = _build_xyz_mosaic(extent84, zoom, provider, mapbox_token)
                ax.set_extent(wm_extent, crs=ccrs.epsg(3857))
                ax.imshow(img_bg, extent=wm_extent, transform=ccrs.epsg(3857),
                          origin="upper", interpolation="bilinear", zorder=1)
        else:
            ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.epsg(3857))
            img_bg, wm_extent, provider_name = _build_xyz_mosaic(extent84, zoom, provider, mapbox_token)
            ax.set_extent(wm_extent, crs=ccrs.epsg(3857))
            ax.imshow(img_bg, extent=wm_extent, transform=ccrs.epsg(3857),
                      origin="upper", interpolation="bilinear", zorder=1)

        hide_axes(ax)

        # ===== Máscara externa escura (SOMENTE acima do fundo) =====
        # zorder=4 (acima dos tiles z~1 e abaixo de polígono/labels/logo)
        if geom is not None and darken_alpha > 0:
            try:
                view_rect = box(extent84[0], extent84[2], extent84[1], extent84[3])
                mask_geom = view_rect.difference(geom.buffer(0))
                ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                                  facecolor=(0, 0, 0, max(0.0, min(1.0, darken_alpha))),
                                  edgecolor='none', zorder=4)
            except Exception as e_mask:
                log.warning(f"Máscara falhou: {e_mask}")

        # ===== Polígono (amarelo ajustável) =====
        if geom is not None:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor=(1.0, 1.0, 0.0, max(0.0, min(1.0, poly_alpha))),
                              edgecolor="#FFE14A", linewidth=3.0, zorder=7)
            for c in ax.collections[-1:]:
                c.set_path_effects([pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()])

        # ===== Pino do imóvel (representative_point) =====
        pt_obj = None
        if geom is not None:
            try:
                pt_obj = geom.representative_point()
                ax.scatter([pt_obj.x],[pt_obj.y], transform=ccrs.PlateCarree(),
                           s=70, zorder=8, marker='o', facecolor=BRAND_COLOR,
                           edgecolor='white', linewidth=1.6)
            except Exception:
                pt_obj = None

        # ===== Pino por coordenadas (query) =====
        if pin_lat is not None and pin_lon is not None:
            ax.scatter([pin_lon],[pin_lat], transform=ccrs.PlateCarree(),
                       s=80, zorder=9, marker='o', facecolor=pin_color,
                       edgecolor='white', linewidth=1.8)
            txt_label = pin_text if pin_text else f"{pin_lat:.5f}, {pin_lon:.5f}"
            ax.annotate(txt_label, xy=(pin_lon, pin_lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(0, 12), textcoords='offset points', ha='center', va='bottom',
                        fontsize=10, color='white',
                        path_effects=[pe.withStroke(linewidth=3.0, foreground='black')], zorder=10)

        # ===== Rótulo CAR =====
        if geom is not None:
            angle = _principal_orientation(geom)
            cx, cy = geom.centroid.x, geom.centroid.y
            txt = ax.text(cx, cy, label_text, transform=ccrs.PlateCarree(),
                          fontsize=13, fontweight="bold", color='white',
                          rotation=angle, ha='center', va='center', zorder=10)
            txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground='black')])

        # ===== Rodapé de coordenadas =====
        if show_coords:
            if pin_lat is not None and pin_lon is not None:
                lat, lon = pin_lat, pin_lon
            elif pt_obj is not None:
                lat, lon = pt_obj.y, pt_obj.x
            elif geom is not None:
                lat, lon = geom.centroid.y, geom.centroid.x
            else:
                lat, lon = 0.0, 0.0
            fig.text(0.015, 0.03, f"{lat:.5f}, {lon:.5f}", fontsize=8, color='white',
                     path_effects=[pe.withStroke(linewidth=2.6, foreground='black')])

        # ===== Atribuição + logo (logo sempre ACIMA da máscara) =====
        fig.text(0.012, 0.012, f"© {provider_name}", fontsize=6, color='white',
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])
        _add_logo(ax, _load_logo(logo_url), width_px=220, zorder=50)  # zorder alto

        # ===== PNG -> JPEG =====
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches=None, pad_inches=0)
        plt.close(fig); buf_png.seek(0)

        img = Image.open(buf_png).convert("RGB")
        buf_jpg = io.BytesIO()
        q = max(60, min(95, jpg_quality))
        img.save(buf_jpg, format="JPEG", quality=q, optimize=True, progressive=True)
        buf_jpg.seek(0)

        name = (cod or "mapa") + ".jpg"
        return send_file(buf_jpg, mimetype="image/jpeg", download_name=name)

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
