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
DEFAULT_PROVIDER = os.getenv("TILE_PROVIDER", "mapbox_static")
GOOGLE_TEMPLATES = [
    "https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    "https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
]
# ⚠️ Agora não deixamos token padrão embutido
MAPBOX_TOKEN_ENV = os.getenv("MAPBOX_TOKEN", "")
USER_AGENT       = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0 (contato@smartfazendas.com.br)")
DEFAULT_LOGO_URL = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_DARKEN_ALPHA = float(os.getenv("DARKEN_ALPHA", "0.55"))  # máscara fora do polígono
DEFAULT_POLY_ALPHA   = float(os.getenv("POLY_ALPHA", "0.28"))    # opacidade do amarelo
DEFAULT_JPG_QUALITY  = int(os.getenv("JPG_QUALITY", "85"))
BRAND_COLOR = os.getenv("BRAND_COLOR", "#346DFF")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))

# 4:3 fixo
TARGET_ASPECT = 4.0 / 3.0
DPI_DEFAULT  = 150
OUT_W_PX_DEFAULT = 1440
OUT_H_PX_DEFAULT = 1080  # 4:3

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

# ---- Mercator helpers
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
        delta = (desired_h - height) / 2.0
        ymin -= delta; ymax += delta
    else:
        desired_w = height * target_aspect
        delta = (desired_w - width) / 2.0
        xmin -= delta; xmax += delta
    return [_inv_merc_x(xmin), _inv_merc_x(xmax), _inv_merc_y(ymin), _inv_merc_y(ymax)]

# ------------------ Mapbox STATIC ------------------
def _fetch_mapbox_static(extent84, width_px, height_px, style_id, token):
    if not token:
        raise ValueError("mapbox_token ausente. Envie ?mapbox_token=SEU_TOKEN na query.")
    minlon, maxlon, minlat, maxlat = extent84
    cx = (minlon + maxlon) / 2.0
    cy = (minlat + maxlat) / 2.0
    zoom = _zoom_from_lon_span(minlon, maxlon)

    base = f"https://api.mapbox.com/styles/v1/{style_id}/static/"
    url = f"{base}{cx},{cy},{zoom}/{width_px}x{height_px}@2x?access_token={token}"

    hdrs = {"User-Agent": USER_AGENT}
    try:
        if requests is None:
            import urllib.request
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=20) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGBA")
        else:
            r = requests.get(url, headers=hdrs, timeout=20)
            if r.status_code == 401:
                raise ValueError("Mapbox 401 Unauthorized: verifique se o token está completo e válido.")
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        raise ValueError(f"Falha ao obter imagem estática do Mapbox: {e}")

    # extent 3857 coerente com centro/zoom e tamanho
    mpp_equator = (2 * math.pi * R_MERC) / (256 * (2 ** zoom))
    mpp = mpp_equator * max(0.2, math.cos(math.radians(cy)))
    half_w = (width_px * mpp) / 2.0
    half_h = (height_px * mpp) / 2.0
    cxm, cym = _merc_x(cx), _merc_y(cy)
    extent3857 = [cxm - half_w, cxm + half_w, cym - half_h, cym + half_h]
    return img, extent3857, f"Mapbox Static ({style_id})"

# ------------------ XYZ (fallback) ------------------
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

    xmin_merc, xmax_merc = _merc_x(minlon), _merc_x(maxlon)
    ymin_merc, ymax_merc = _merc_y(minlat), _merc_y(maxlat)
    extent3857 = [xmin_merc, xmax_merc, ymin_merc, ymax_merc]

    prov_name = {"google_hybrid":"Google Hybrid",
                 "mapbox_hybrid":"Mapbox Satellite-Streets",
                 "osm":"OpenStreetMap"}.get(actual_provider, actual_provider)
    return crop, extent3857, prov_name

# ------------------ Endpoint ------------------
@app.post("/generate-map")
def generate_map():
    try:
        q = request.args
        provider     = q.get("provider", DEFAULT_PROVIDER).strip()
        darken_alpha = float(q.get("darken", DEFAULT_DARKEN_ALPHA))
        poly_alpha   = float(q.get("transparencia", q.get("poly_alpha", DEFAULT_POLY_ALPHA)))
        poly_alpha   = max(0.0, min(1.0, poly_alpha))
        jpg_quality  = int(q.get("jpg_quality", DEFAULT_JPG_QUALITY))
        logo_url     = q.get("logo_url", DEFAULT_LOGO_URL)
        show_coords  = q.get("coords", "1").lower() in ("1","true","yes")
        # token só por query (ou env), sem default embutido
        mapbox_token = (q.get("mapbox_token") or MAPBOX_TOKEN_ENV or "").strip().strip('"').strip("'")

        car_title = q.get("car", "").strip()

        out_w = int(q.get("width_px", OUT_W_PX_DEFAULT))
        out_h = int(q.get("height_px", OUT_H_PX_DEFAULT))
        if abs((out_w / max(1,out_h)) - TARGET_ASPECT) > 1e-6:
            out_h = int(round(out_w / TARGET_ASPECT))
        dpi = int(q.get("dpi", DPI_DEFAULT))

        # pino
        def _parse_num(s):
            if s is None: return None
            try:
                return float(str(s).replace(",", "."))
            except Exception:
                return None
        pin_lat = _parse_num(q.get("latitude", q.get("pin_lat")))
        pin_lon = _parse_num(q.get("longitude", q.get("pin_lon")))
        pin_text = q.get("pin_text")
        pin_color = q.get("pin_color", BRAND_COLOR)
        pin_zoom = q.get("pin_zoom"); pin_zoom = int(pin_zoom) if pin_zoom not in (None, "") else None

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

        fig = Figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
        fig.set_facecolor("white")
        canvas = FigureCanvas(fig)

        # extent
        if geom is not None:
            extent84 = _extent_with_padding(geom, pad_ratio=0.10)
            if pin_lat is not None and pin_lon is not None:
                minx, maxx, miny, maxy = extent84
                extent84 = [min(minx, pin_lon), max(maxx, pin_lon),
                            min(miny, pin_lat), max(maxy, pin_lat)]
            extent84 = _adjust_extent_to_aspect(extent84, TARGET_ASPECT)
            zoom = _zoom_from_lon_span(extent84[0], extent84[1])
        elif pin_lat is not None and pin_lon is not None:
            zoom = min(18, max(1, pin_zoom if pin_zoom is not None else 15))
            extent84 = _extent_from_center_zoom(pin_lon, pin_lat, zoom)
            extent84 = _adjust_extent_to_aspect(extent84, TARGET_ASPECT)
            zoom = _zoom_from_lon_span(extent84[0], extent84[1])
        else:
            return jsonify({"error": "Forneça KML no body (JSON['data']) ou 'latitude' + 'longitude' na query."}), 400

        # fundo
        provider_name = "OpenStreetMap"
        if provider.lower().startswith("mapbox_static"):
            if not mapbox_token:
                return jsonify({"error": "mapbox_token é obrigatório para provider=mapbox_static. "
                                         "Envie ?mapbox_token=SEU_TOKEN_na_query"}), 400
            style_id = q.get("style", "mapbox/satellite-v9")
            ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.epsg(3857))
            img_bg, wm_extent, provider_name = _fetch_mapbox_static(
                extent84, out_w, out_h, style_id, mapbox_token
            )
            ax.set_extent(wm_extent, crs=ccrs.epsg(3857))
            ax.imshow(img_bg, extent=wm_extent, transform=ccrs.epsg(3857),
                      origin="upper", interpolation="bilinear", zorder=1)
        else:
            if HAS_IMGTILES:
                tile_src, provider_name = _make_tile_source(provider, mapbox_token)
                ax = fig.add_axes([0, 0, 1, 1],
                                  projection=getattr(tile_src, "crs", ccrs.epsg(3857)))
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

        try: ax.set_axis_off()
        except Exception: pass

        # máscara externa
        if geom is not None:
            try:
                view_rect = box(extent84[0], extent84[2], extent84[1], extent84[3])
                mask_geom = view_rect.difference(geom.buffer(0))
                ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                                  facecolor=(0,0,0,max(0.0,min(1.0,darken_alpha))),
                                  edgecolor='none', zorder=6)
            except Exception as e_mask:
                log.warning(f"Máscara falhou: {e_mask}")

        # polígono
        if geom is not None:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor=(1.0,1.0,0.0,poly_alpha),
                              edgecolor="#FFE14A", linewidth=3.0, zorder=7)
            for c in ax.collections[-1:]:
                c.set_path_effects([pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()])

        # pin
        if pin_lat is not None and pin_lon is not None:
            ax.scatter([pin_lon],[pin_lat], transform=ccrs.PlateCarree(),
                       s=90, zorder=9, marker='o', facecolor=pin_color,
                       edgecolor='white', linewidth=1.8)
            lat_raw = request.args.get("latitude", request.args.get("pin_lat"))
            lon_raw = request.args.get("longitude", request.args.get("pin_lon"))
            label_txt = pin_text if (pin_text and pin_text.strip()) else \
                        (f"{lat_raw}, {lon_raw}" if (lat_raw and lon_raw) else f"{pin_lat:.5f}, {pin_lon:.5f}")
            ax.annotate(label_txt, xy=(pin_lon, pin_lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(0, 12), textcoords='offset points', ha='center', va='bottom',
                        fontsize=10, color='white',
                        path_effects=[pe.withStroke(linewidth=3.0, foreground='black')], zorder=10)

        # rótulo interno do CAR (quando existir no KML)
        if geom is not None and cod:
            angle = _principal_orientation(geom)
            cx, cy = geom.centroid.x, geom.centroid.y
            txt = ax.text(cx, cy, label_text, transform=ccrs.PlateCarree(),
                          fontsize=13, fontweight="bold", color='white',
                          rotation=angle, ha='center', va='center', zorder=9)
            txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground='black')])

        # título superior do CAR (opcional via ?car=...)
        if car_title:
            fig.text(0.02, 0.975, car_title,
                     fontsize=18, fontweight="bold", color='white', ha='left', va='top',
                     path_effects=[pe.withStroke(linewidth=4.0, foreground='black')])

        # rodapé coords
        if show_coords:
            if pin_lat is not None and pin_lon is not None:
                lat, lon = pin_lat, pin_lon
            elif geom is not None:
                c = geom.representative_point()
                lat, lon = c.y, c.x
            else:
                lat, lon = 0.0, 0.0
            fig.text(0.015, 0.03, f"{lat:.5f}, {lon:.5f}", fontsize=8, color='white',
                     path_effects=[pe.withStroke(linewidth=2.6, foreground='black')])

        # atribuição + logo
        fig.text(0.012, 0.012, f"© {provider_name}", fontsize=6, color='white',
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        def _load_logo(url: str):
            try:
                if requests is None:
                    import urllib.request
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=10) as r: data = r.read()
                else:
                    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
                    r.raise_for_status(); data = r.content
                return Image.open(io.BytesIO(data)).convert("RGBA")
            except Exception as e:
                log.warning(f"Falha logo: {e}")
                return None
        logo_img = _load_logo(logo_url)
        if logo_img is not None:
            w, h = logo_img.size
            zoom = 220 / float(w)
            imagebox = OffsetImage(logo_img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (0.985, 0.985), xycoords='axes fraction',
                                frameon=False, box_alignment=(1,1), pad=0)
            ab.set_zorder(1000)
            ax.add_artist(ab)

        # salvar JPEG
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
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
        info["tile_provider_selected"] = "Mapbox Static (default)" if DEFAULT_PROVIDER.startswith("mapbox_static") else DEFAULT_PROVIDER
        info["tile_src_type"] = "Static" if DEFAULT_PROVIDER.startswith("mapbox_static") else ("ImageTiles" if HAS_IMGTILES else "PIL-mosaic")
        info["mapbox_token_present"] = bool(mapbox_token)
    return info, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")))
