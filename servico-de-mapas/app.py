# app.py
import os, io, base64, math, logging, random
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from shapely.geometry import GeometryCollection, box, Polygon
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
USER_AGENT       = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0 (contato@smartfazendas.com.br)")
DEFAULT_LOGO_URL = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_DARKEN_ALPHA = float(os.getenv("DARKEN_ALPHA", "0.55"))   # escurecer fora do polígono
DEFAULT_POLY_ALPHA   = float(os.getenv("POLY_ALPHA", "0.28"))     # preenchimento amarelo
DEFAULT_JPG_QUALITY  = int(os.getenv("JPG_QUALITY", "82"))
BRAND_COLOR = os.getenv("BRAND_COLOR", "#346DFF")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))
MAPBOX_TOKEN_ENV = os.getenv("MAPBOX_TOKEN", "")

# Aspecto alvo 4:3 (largura:altura)
TARGET_ASPECT = 4.0 / 3.0

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

# ---------- provedores ----------
def _google_lyrs(provider: str) -> str:
    # m=mapa, s=sat, y=hybrid, p=terrain
    return {
        "google_streets": "m",
        "google_satellite": "s",
        "google_hybrid": "y",
        "google_terrain": "p",
    }.get(provider, "y")

def _mapbox_style(provider: str) -> str:
    return {
        "mapbox_satellite": "mapbox/satellite-v9",
        "mapbox_satellite_streets": "mapbox/satellite-streets-v12",
        "mapbox_streets": "mapbox/streets-v12",
        "mapbox_outdoors": "mapbox/outdoors-v12",
        "mapbox_light": "mapbox/light-v11",
        "mapbox_dark": "mapbox/dark-v11",
    }.get(provider, "mapbox/satellite-streets-v12")

def _provider_template(provider: str, mapbox_token: str) -> str | None:
    p = provider.strip().lower()
    if p.startswith("google_"):
        sub = random.choice(["mt0","mt1","mt2","mt3"])
        lyrs = _google_lyrs(p)
        return f"https://{sub}.google.com/vt/lyrs={lyrs}&x={{x}}&y={{y}}&z={{z}}"
    if p.startswith("mapbox:"):
        # estilo completo passado pelo usuário: mapbox:<user_or_ns>/<style-id>
        style = p.split(":",1)[1]
        if not mapbox_token: return None
        return f"https://api.mapbox.com/styles/v1/{style}/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={mapbox_token}"
    if p.startswith("mapbox_"):
        if not mapbox_token: return None
        style = _mapbox_style(p)
        return f"https://api.mapbox.com/styles/v1/{style}/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={mapbox_token}"
    if p in ("osm","openstreetmap"):
        return "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png".replace("{s}", random.choice(["a","b","c"]))
    # fallback
    sub = random.choice(["mt0","mt1","mt2","mt3"])
    return f"https://{sub}.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}"

def _make_tile_source(provider: str, mapbox_token: str):
    tpl = _provider_template(provider, mapbox_token)
    if tpl and XYZTiles is not None:
        return XYZTiles(tpl, cache=True, user_agent=USER_AGENT), provider
    return cimgt.OSM(cache=True, user_agent=USER_AGENT), "OpenStreetMap"

# ------------------ KML helpers ------------------
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

# Mercator helpers (para ajustar 4:3)
R_MERC = 6378137.0
def _merc_x(lon): return math.radians(lon) * R_MERC
def _merc_y(lat):
    lat = max(-85.05112878, min(85.05112878, lat))
    return R_MERC * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
def _inv_merc_x(x): return math.degrees(x / R_MERC)
def _inv_merc_y(y): return math.degrees(2*math.atan(math.exp(y / R_MERC)) - math.pi/2)

def _adjust_extent_to_aspect(ext84, target_aspect=TARGET_ASPECT):
    minlon, maxlon, minlat, maxlat = ext84
    xmin, xmax = _merc_x(minlon), _merc_x(maxlon)
    ymin, ymax = _merc_y(minlat), _merc_y(maxlat)
    width  = max(1e-9, xmax - xmin)
    height = max(1e-9, ymax - ymin)
    cur_aspect = width / height
    if abs(cur_aspect - target_aspect) < 1e-6:
        return ext84
    if cur_aspect > target_aspect:
        desired_h = width / target_aspect
        d = (desired_h - height) / 2.0
        ymin -= d; ymax += d
    else:
        desired_w = height * target_aspect
        d = (desired_w - width) / 2.0
        xmin -= d; xmax += d
    return [_inv_merc_x(xmin), _inv_merc_x(xmax), _inv_merc_y(ymin), _inv_merc_y(ymax)]

# -------- ocultar eixos + full-bleed
def hide_axes(ax):
    try: ax.set_axis_off()
    except: pass
    try:
        op = getattr(ax, "outline_patch", None)
        if op is not None: op.set_visible(False)
    except: pass
    try:
        for sp in getattr(ax, "spines", {}).values():
            sp.set_visible(False)
    except: pass
    try:
        if hasattr(ax, "patch") and ax.patch is not None:
            ax.patch.set_visible(False)
    except: pass

def force_full_bleed(ax):
    try: ax.set_aspect('auto', adjustable='box')
    except: pass
    try: ax.set_position([0, 0, 1, 1])
    except: pass

# ------------------ XYZ manual (fallback) ------------------
def _tile_url(provider, x, y, z, mapbox_token):
    p = provider.strip().lower()
    if p.startswith("google_"):
        sub = random.choice(["mt0","mt1","mt2","mt3"])
        lyrs = _google_lyrs(p)
        return f"https://{sub}.google.com/vt/lyrs={lyrs}&x={x}&y={y}&z={z}"
    if p.startswith("mapbox:"):
        style = p.split(":",1)[1]
        return f"https://api.mapbox.com/styles/v1/{style}/tiles/256/{z}/{x}/{y}@2x?access_token={mapbox_token}"
    if p.startswith("mapbox_"):
        style = _mapbox_style(p)
        return f"https://api.mapbox.com/styles/v1/{style}/tiles/256/{z}/{x}/{y}@2x?access_token={mapbox_token}"
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

def _lonlat_to_pixel(lon, lat, z, tile_size=256):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * tile_size
    lat = max(-85.05112878, min(85.05112878, lat))
    y = (1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n * tile_size
    return x, y

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

    xmin_merc, xmax_merc = _merc_x(minlon), _merc_x(maxlon)
    ymin_merc, ymax_merc = _merc_y(minlat), _merc_y(maxlat)
    extent3857 = [xmin_merc, xmax_merc, ymin_merc, ymax_merc]

    return crop, extent3857, actual_provider

# ------------------ Endpoints ------------------
@app.post("/generate-map")
def generate_map():
    try:
        q = request.args
        provider     = q.get("provider", DEFAULT_PROVIDER)
        darken_alpha = float(q.get("darken", DEFAULT_DARKEN_ALPHA))             # fora do polígono
        poly_alpha   = float(q.get("transparencia", q.get("poly_alpha", DEFAULT_POLY_ALPHA)))
        jpg_quality  = int(q.get("jpg_quality", DEFAULT_JPG_QUALITY))
        logo_url     = q.get("logo_url", DEFAULT_LOGO_URL)
        show_coords  = q.get("coords", "1").lower() in ("1","true","yes")
        mapbox_token = q.get("mapbox_token", MAPBOX_TOKEN_ENV)

        # CAR (título opcional)
        car_title = q.get("car")  # se None, não mostra título

        # Pino: somente se vierem latitude & longitude
        lat_raw = q.get("latitude", q.get("pin_lat"))
        lon_raw = q.get("longitude", q.get("pin_lon"))
        pin_lat = float(lat_raw) if lat_raw is not None else None
        pin_lon = float(lon_raw) if lon_raw is not None else None
        pin_text = q.get("pin_text")  # se não vier, usar exatamente o texto das queries (lat_raw, lon_raw)
        pin_color = q.get("pin_color", BRAND_COLOR)
        pin_zoom  = q.get("pin_zoom"); pin_zoom = int(pin_zoom) if pin_zoom else None

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

        # ===== Figura 4:3 (sem bordas) =====
        fig = Figure(figsize=(12, 9), dpi=150)   # 4:3
        fig.set_facecolor("white")
        FigureCanvas(fig)
        fig.subplots_adjust(0,0,1,1)

        # ===== Extent/zoom inicial =====
        if geom is not None:
            extent84 = _extent_with_padding(geom, pad_ratio=0.10)
            if pin_lat is not None and pin_lon is not None:
                minx, maxx, miny, maxy = extent84
                minx = min(minx, pin_lon); maxx = max(maxx, pin_lon)
                miny = min(miny, pin_lat); maxy = max(maxy, pin_lat)
                extent84 = [minx, maxx, miny, maxy]
            extent84 = _adjust_extent_to_aspect(extent84, TARGET_ASPECT)
            zoom = _zoom_from_lon_span(extent84[0], extent84[1])
        elif pin_lat is not None and pin_lon is not None:
            zoom = min(18, max(1, pin_zoom if pin_zoom is not None else 15))
            extent84 = _extent_from_center_zoom(pin_lon, pin_lat, zoom)
            extent84 = _adjust_extent_to_aspect(extent84, TARGET_ASPECT)
            zoom = _zoom_from_lon_span(extent84[0], extent84[1])
        else:
            return jsonify({"error": "Forneça KML em JSON['data'] ou 'latitude' e 'longitude' na query."}), 400

        # ===== Eixo/tiles =====
        if HAS_IMGTILES:
            tile_src, provider_name = _make_tile_source(provider, mapbox_token)
            ax = fig.add_axes([0, 0, 1, 1], projection=getattr(tile_src, "crs", ccrs.epsg(3857)))
            ax.set_extent(extent84, crs=ccrs.PlateCarree())
            try:
                ax.add_image(tile_src, zoom, interpolation="spline36")
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

        # full-bleed + esconder eixos
        force_full_bleed(ax)
        hide_axes(ax)

        # ===== Máscara externa escura =====
        if geom is not None and darken_alpha > 0:
            try:
                view_rect = box(extent84[0], extent84[2], extent84[1], extent84[3])
                mask_geom = view_rect.difference(geom.buffer(0))
                ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                                  facecolor=(0, 0, 0, max(0.0, min(1.0, darken_alpha))),
                                  edgecolor='none', zorder=6)
            except Exception as e_mask:
                log.warning(f"Máscara falhou: {e_mask}")

        # ===== Polígono (amarelo com transparência da query) =====
        if geom is not None:
            a = max(0.0, min(1.0, poly_alpha))
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor=(1.0, 1.0, 0.0, a),
                              edgecolor="#FFE14A", linewidth=3.0, zorder=7)
            for c in ax.collections[-1:]:
                c.set_path_effects([pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()])

        # ===== Pino por coordenadas (opcional) =====
        if (pin_lat is not None) and (pin_lon is not None):
            ax.scatter([pin_lon],[pin_lat], transform=ccrs.PlateCarree(),
                       s=110, zorder=9, marker='o', facecolor=pin_color,
                       edgecolor='white', linewidth=2.0)
            label_txt = pin_text if pin_text else (f"{lat_raw}, {lon_raw}")
            ax.annotate(label_txt, xy=(pin_lon, pin_lat),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(0, 14), textcoords='offset points', ha='center', va='bottom',
                        fontsize=12, color='white',
                        path_effects=[pe.withStroke(linewidth=3.2, foreground='black')], zorder=10)

        # ===== Título CAR (se car= for informado) =====
        if car_title:
            # reservo espaço da logo (que fica no canto superior direito)
            fig.text(0.82, 0.965, str(car_title),
                     ha="right", va="top", fontsize=18, fontweight="bold",
                     color="white",
                     path_effects=[pe.withStroke(linewidth=4.0, foreground='black')])

        # ===== Rodapé de coordenadas (opcional) =====
        if show_coords:
            # Se houver pino, prioriza ele; senão, usa centróide do polígono
            if (pin_lat is not None) and (pin_lon is not None):
                lat, lon = pin_lat, pin_lon
            elif geom is not None:
                cen = geom.centroid
                lat, lon = cen.y, cen.x
            else:
                lat, lon = 0.0, 0.0
            fig.text(0.015, 0.03, f"{lat:.5f}, {lon:.5f}", fontsize=9, color='white',
                     path_effects=[pe.withStroke(linewidth=2.6, foreground='black')])

        # ===== Atribuição + LOGO (sempre na frente) =====
        fig.text(0.012, 0.012, f"© {provider_name}", fontsize=7, color='white',
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

        # LOGO sempre por cima (sem escurecer)
        def _add_logo(ax, url: str, width_px=220):
            if not url: return
            try:
                if requests is None:
                    import urllib.request
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=10) as r: data = r.read()
                else:
                    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
                    r.raise_for_status(); data = r.content
                pil_img = Image.open(io.BytesIO(data)).convert("RGBA")
            except Exception as e:
                log.warning(f"Falha logo: {e}")
                return
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            w,h = pil_img.size
            zoom = width_px/float(w)
            imagebox = OffsetImage(pil_img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (0.985, 0.985), xycoords='axes fraction',
                                frameon=False, box_alignment=(1,1), pad=0)
            ab.set_zorder(10000)
            ab.set_clip_on(False)
            ax.add_artist(ab)

        _add_logo(ax, logo_url, width_px=220)

        # ===== PNG -> JPEG =====
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches=None, pad_inches=0)
        plt.close(fig); buf_png.seek(0)

        img = Image.open(buf_png).convert("RGB")
        buf_jpg = io.BytesIO()
        q = max(60, min(95, jpg_quality))
        img.save(buf_jpg, format="JPEG", quality=q, optimize=True, progressive=True)
        buf_jpg.seek(0)

        name = (car_title or "mapa") + ".jpg"
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
