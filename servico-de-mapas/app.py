# app.py
import os, io, base64, json, math, zipfile, logging
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

try:
    import requests
except Exception:
    requests = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("map-croqui-static")

# ====================== Config ======================
DEFAULT_STYLE = os.getenv("MAPBOX_STYLE", "mapbox/satellite-v9")
DEFAULT_LOGO  = os.getenv("LOGO_URL", "https://raw.githubusercontent.com/rodrigocoladello/logomarca/main/Logo%20Smart%20Fazendas%20Roxo.png")
DEFAULT_W     = int(os.getenv("OUT_W", "1280"))   # 4:3
DEFAULT_H     = int(os.getenv("OUT_H", "960"))    # 4:3
DEFAULT_SCALE = int(os.getenv("SCALE", "1"))      # 1 ou 2 (@2x)
DEFAULT_DARK  = float(os.getenv("DARKEN", "0.55"))
DEFAULT_POLY_A= float(os.getenv("POLY_ALPHA", "0.28"))
DEFAULT_PAD   = float(os.getenv("FIT_PADDING", "0.10"))  # 0..0.4
DEFAULT_SIMP  = float(os.getenv("SIMPLIFY_DEG", "0.0025"))
DEFAULT_JPG_Q = int(os.getenv("JPG_QUALITY", "85"))
UA            = os.getenv("TILE_USER_AGENT", "SmartFazendas/1.0")

# ================== Helpers geom (JS -> PY) ==================
def get_sq_seg_dist(p, p1, p2):
    x, y = p1; dx, dy = p2[0]-x, p2[1]-y
    if dx != 0 or dy != 0:
        t = ((p[0]-x)*dx + (p[1]-y)*dy) / (dx*dx + dy*dy)
        if t > 1: x, y = p2
        elif t > 0: x += dx*t; y += dy*t
    dx = p[0]-x; dy = p[1]-y
    return dx*dx + dy*dy

def simplify_dp(points, sq_tol):
    if len(points) <= 2: return points[:]
    markers = [0]*len(points)
    first, last = 0, len(points)-1
    stack = []
    markers[first] = markers[last] = 1
    while True:
        max_sq = 0.0; index = None
        for i in range(first+1, last):
            sq = get_sq_seg_dist(points[i], points[first], points[last])
            if sq > max_sq: index, max_sq = i, sq
        if index is not None and max_sq > sq_tol:
            markers[index] = 1
            stack.append((first, index)); stack.append((index, last))
        if not stack: break
        first, last = stack.pop()
    return [pt for i,pt in enumerate(points) if markers[i]]

def simplify(points, tol_deg):
    if len(points) <= 2: return points[:]
    sq = (tol_deg or 0.0) ** 2
    return simplify_dp(points, sq)

def close_ring_if_needed(r):
    if not r: return r
    if r[0][0] != r[-1][0] or r[0][1] != r[-1][1]:
        r = r + [r[0]]
    return r

def bbox_from_ring(r):
    minLon = min(p[0] for p in r); maxLon = max(p[0] for p in r)
    minLat = min(p[1] for p in r); maxLat = max(p[1] for p in r)
    return [minLon, minLat, maxLon, maxLat]

def centroid_from_ring(r):
    # centróide de polígono (shoelace)
    A = 0.0; cx = 0.0; cy = 0.0
    for i in range(len(r)-1):
        x0,y0 = r[i]; x1,y1 = r[i+1]
        cross = x0*y1 - x1*y0
        A += cross; cx += (x0+x1)*cross; cy += (y0+y1)*cross
    A *= 0.5
    if abs(A) < 1e-12:
        # fallback: média simples
        xs = sum(p[0] for p in r)/len(r); ys = sum(p[1] for p in r)/len(r)
        return [xs, ys]
    return [cx/(6*A), cy/(6*A)]

# WebMercator helpers
def lon2x(lon): return (lon + 180.0) / 360.0
def lat2y(lat):
    lat = max(-85.05112878, min(85.05112878, lat))
    r = math.radians(lat)
    return (1.0 - math.log(math.tan(r) + 1.0/math.cos(r)) / math.pi) / 2.0

def compute_zoom_for_bbox(bbox, W, H, padding=0.10, tile=512):
    minLon,minLat,maxLon,maxLat = bbox
    x1,x2 = lon2x(minLon), lon2x(maxLon)
    y1,y2 = lat2y(maxLat), lat2y(minLat)  # y cresce p/ sul
    dx = max(1e-9, abs(x2-x1))
    dy = max(1e-9, abs(y2-y1))
    innerW = max(1.0, W * (1.0 - 2.0*padding))
    innerH = max(1.0, H * (1.0 - 2.0*padding))
    scaleX = innerW / (tile * dx)
    scaleY = innerH / (tile * dy)
    z = math.floor(math.log2(max(1e-9, min(scaleX, scaleY))))
    return int(max(3, min(20, z)))

def lonlat_to_image_px(lon, lat, cx, cy, zoom, W, H, tile=512):
    # world pixel coords at this zoom (origin = top-left)
    n = (2 ** zoom) * tile
    wx = lon2x(lon) * n
    wy = lat2y(lat) * n
    wcx = lon2x(cx) * n
    wcy = lat2y(cy) * n
    # screen px = delta to center + half of image
    px = (wx - wcx) + (W / 2.0)
    py = (wy - wcy) + (H / 2.0)
    return px, py

# ================== KMZ/KML parsing ==================
def extract_kml_bytes(payload):
    if payload is None:
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        raw = payload[0].get("data")
    elif isinstance(payload, dict):
        raw = payload.get("data")
    else:
        raw = None
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.lower().startswith("data:") and ";base64," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return s.encode("utf-8")

def kml_from_kmz_or_kml(kbytes):
    """Retorna string XML KML a partir de bytes (KMZ ou KML)."""
    if kbytes is None:
        return None
    # KMZ zip?
    try:
        with zipfile.ZipFile(io.BytesIO(kbytes)) as zf:
            # pega o 1º .kml
            for name in zf.namelist():
                if name.lower().endswith(".kml"):
                    with zf.open(name) as fh:
                        return fh.read().decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        pass
    # Senão, assume KML direto
    try:
        return kbytes.decode("utf-8", errors="ignore")
    except Exception:
        return None

def first_polygon_ring_from_kml(kml_text):
    """
    Procura o primeiro <Polygon> e retorna o anel externo como [[lon,lat],...].
    Se não encontrar, tenta o primeiro <coordinates> em qualquer lugar.
    """
    if not kml_text:
        return None
    try:
        root = ET.fromstring(kml_text)
    except Exception:
        # pequenos ajustes caso tenha namespaces inesperados
        try:
            root = ET.fromstring(kml_text.encode("utf-8", errors="ignore"))
        except Exception:
            return None
    # descobrir ns KML
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0].strip("{")
    def q(tag): return f"{{{ns}}}{tag}" if ns else tag

    # Tenta Polygon->outerBoundaryIs->LinearRing->coordinates
    for poly in root.findall(f".//{q('Polygon')}"):
        coords_el = poly.find(f".//{q('outerBoundaryIs')}/{q('LinearRing')}/{q('coordinates')}")
        if coords_el is None:
            coords_el = poly.find(f".//{q('coordinates')}")
        if coords_el is not None and coords_el.text:
            ring = []
            for tok in coords_el.text.replace("\n"," ").split():
                parts = tok.split(",")
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0]); lat = float(parts[1])
                        ring.append([lon, lat])
                    except: pass
            if len(ring) >= 3:
                return close_ring_if_needed(ring)

    # fallback: primeiro coordinates encontrado
    coords_el = root.find(f".//{q('coordinates')}")
    if coords_el is not None and coords_el.text:
        ring = []
        for tok in coords_el.text.replace("\n"," ").split():
            parts = tok.split(",")
            if len(parts) >= 2:
                try:
                    lon = float(parts[0]); lat = float(parts[1])
                    ring.append([lon, lat])
                except: pass
        if len(ring) >= 3:
            return close_ring_if_needed(ring)

    return None

# ================== Logo loader ==================
_logo_cache = None
def load_logo(url):
    global _logo_cache
    if _logo_cache is not None:
        return _logo_cache
    try:
        if requests is None:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = r.read()
        else:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=12)
            r.raise_for_status(); data = r.content
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        _logo_cache = img
        return img
    except Exception as e:
        log.warning(f"Falha ao carregar logo: {e}")
        return None

def draw_text_with_stroke(draw, xy, text, fill, stroke_fill, stroke_width, font=None, anchor=None):
    x, y = xy
    # 8 direções + centro
    offs = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    for dx,dy in offs:
        draw.text((x+dx*stroke_width, y+dy*stroke_width), text, font=font, fill=stroke_fill, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

# ================== Endpoint ==================
@app.post("/generate-map")
def generate_map():
    try:
        q = request.args

        # Mapbox
        token = q.get("mapbox_token", os.getenv("MAPBOX_TOKEN", "")).strip()
        if not token:
            return jsonify({"error":"Forneça ?mapbox_token=... na query ou defina MAPBOX_TOKEN no ambiente."}), 400
        style = q.get("style", DEFAULT_STYLE)

        # Saída
        out_w = int(q.get("out_w", DEFAULT_W))
        out_h = int(q.get("out_h", DEFAULT_H))
        scale = int(q.get("scale", DEFAULT_SCALE))
        retina = "@2x" if scale == 2 else ""

        # Estética
        darken = float(q.get("darken", DEFAULT_DARK))
        poly_alpha = float(q.get("transparencia", q.get("poly_alpha", DEFAULT_POLY_A)))
        pad = float(q.get("padding", DEFAULT_PAD))
        simp_tol = float(q.get("simplify", DEFAULT_SIMP))

        # Títulos / logo
        car_text = q.get("car")  # opcional: aparece como título no topo
        logo_url = q.get("logo_url", DEFAULT_LOGO)

        # Pin opcional
        lat = q.get("latitude") or q.get("lat") or q.get("Latitude")
        lon = q.get("longitude") or q.get("lon") or q.get("lng") or q.get("Longitude") or q.get("Lon") or q.get("Lng")
        pin_color = (q.get("pin_color") or "635aff").lstrip("#")
        have_pin = False
        if lat is not None and lon is not None:
            try:
                lat = float(lat); lon = float(lon)
                have_pin = True
            except:
                have_pin = False

        # KML/KMZ
        payload = request.get_json(silent=True)
        kbytes  = extract_kml_bytes(payload)
        kmltxt  = kml_from_kmz_or_kml(kbytes)
        ring    = first_polygon_ring_from_kml(kmltxt)
        if not ring:
            return jsonify({"error":"KML/KMZ sem anel poligonal válido."}), 400

        # Simplificar e limitar pontos p/ URL curta
        ring = simplify(ring, simp_tol)
        # limite duro de pontos para o overlay (evita 414/422)
        MAX_PTS = int(q.get("max_pts", "800"))
        if len(ring) > MAX_PTS:
            step = math.ceil(len(ring) / MAX_PTS)
            ring = ring[::step] + ([ring[0]] if ring[-1] != ring[0] else [])

        # Centro + zoom (se pin não vier, usa centróide)
        bbox = bbox_from_ring(ring)
        cx, cy = centroid_from_ring(ring)
        if have_pin and q.get("center_on_pin", "0").lower() in ("1","true","yes"):
            cx, cy = float(lon), float(lat)
        zoom = compute_zoom_for_bbox(bbox, out_w, out_h, padding=pad)

        # Overlays: sombra (donut) + imóvel
        feature_imovel = {
            "type":"Feature",
            "properties": { "stroke":"#ffff00", "stroke-width":3, "fill":"#ffff00", "fill-opacity": max(0.0, min(1.0, poly_alpha)) },
            "geometry": { "type":"Polygon", "coordinates":[ ring ] }
        }
        world = [[-179.999,-85], [179.999,-85], [179.999,85], [-179.999,85], [-179.999,-85]]
        hole  = list(reversed(ring))
        feature_sombra = {
            "type":"Feature",
            "properties": { "fill":"#000000", "fill-opacity": max(0.0, min(1.0, darken)), "stroke-width":0 },
            "geometry": { "type":"Polygon", "coordinates":[ world, hole ] }
        }
        features = [feature_sombra, feature_imovel]

        overlays_geojson = { "type":"FeatureCollection", "features": features }
        overlays_encoded = requests.utils.quote(json.dumps(overlays_geojson, separators=(',',':'))) if requests else json.dumps(overlays_geojson)

        # Pin no Mapbox (desenha o pin do mapa)
        overlays = f"geojson({overlays_encoded})"
        if have_pin:
            overlays += f",pin-s+{pin_color}({lon},{lat})"

        # Static URL
        base = f"https://api.mapbox.com/styles/v1/{style}/static/{overlays}/{cx},{cy},{zoom}/{out_w}x{out_h}{retina}"
        static_url = f"{base}?access_token={token}&logo=false&attribution=false"

        # Baixa mapa + logo
        headers = {"User-Agent": UA}
        try:
            if requests is None:
                import urllib.request
                req = urllib.request.Request(static_url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as r:
                    map_bytes = r.read()
            else:
                r = requests.get(static_url, headers=headers, timeout=20)
                r.raise_for_status()
                map_bytes = r.content
        except Exception as e:
            # esconde token no erro
            safe_url = static_url.replace(token, "TOKEN")
            return jsonify({"error": f"Falha ao obter imagem estática do Mapbox: {e}", "debug_url": safe_url}), 400

        img = Image.open(io.BytesIO(map_bytes)).convert("RGBA")

        # Desenha texto das coordenadas sobre o pin (em px corretos)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if have_pin:
            px, py = lonlat_to_image_px(lon, lat, cx, cy, zoom, out_w, out_h, tile=512)
            label = f"{lat:.5f}, {lon:.5f}"
            draw_text_with_stroke(draw, (px, py-16), label, fill="white",
                                  stroke_fill="black", stroke_width=2, font=font, anchor="mb")

        # Título CAR (opcional) no topo-esquerdo, respeitando área da logo
        if car_text:
            pad_x = 16; pad_y = 14
            title = f"CAR: {car_text}"
            draw_text_with_stroke(draw, (pad_x, pad_y), title, fill="white",
                                  stroke_fill="black", stroke_width=2, font=font, anchor="la")

        # Logo no topo-direito (sempre acima)
        try:
            logo = load_logo(logo_url)
            if logo is not None:
                target_w = int(out_w * 0.30)
                w,h = logo.size
                ratio = target_w / float(w)
                new_h = int(h * ratio)
                logo = logo.resize((target_w, new_h), Image.LANCZOS)
                lx = out_w - target_w - 16
                ly = 12
                img.alpha_composite(logo, (lx, ly))
        except Exception as e:
            log.warning(f"Logo skip: {e}")

        # JPEG saída
        out_rgb = img.convert("RGB")
        buf = io.BytesIO()
        q = max(60, min(95, DEFAULT_JPG_Q if "jpg_quality" not in q else int(q.get("jpg_quality"))))
        out_rgb.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg", download_name="mapa_smart_fazendas.jpg")
    except Exception as e:
        log.exception("Erro inesperado")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return {
        "ok": True,
        "style_default": DEFAULT_STYLE,
        "out_default": [DEFAULT_W, DEFAULT_H],
        "scale_default": DEFAULT_SCALE
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")))
