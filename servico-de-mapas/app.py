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
# ======== WebMercator & URL-safe simplificação (DP em pixels) ========
import urllib.parse

TILE_SIZE = 512
R_MERC = 6378137.0

def _m_x(lon): return math.radians(lon) * R_MERC
def _m_y(lat):
    lat = max(-85.05112878, min(85.05112878, lat))
    return R_MERC * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
def _inv_m_x(x): return math.degrees(x / R_MERC)
def _inv_m_y(y): return math.degrees(2*math.atan(math.exp(y/R_MERC)) - math.pi/2)

def _lon2x(lon): return (lon + 180.0) / 360.0
def _lat2y(lat):
    r = math.radians(lat)
    return (1.0 - math.log(math.tan(r) + 1.0/math.cos(r)) / math.pi) / 2.0

def meters_per_pixel(lat_deg, z, tile_size=TILE_SIZE):
    return (math.cos(math.radians(lat_deg)) * 2*math.pi*R_MERC) / (tile_size * (2**z))

def compute_zoom_for_bbox(ext84, out_w, out_h, padding=0.10, tile_size=TILE_SIZE):
    minlon,maxlon,minlat,maxlat = ext84
    x1,x2 = _lon2x(minlon), _lon2x(maxlon)
    y1,y2 = _lat2y(maxlat), _lat2y(minlat)  # y cresce p/ sul
    dx = max(1e-9, abs(x2-x1)); dy = max(1e-9, abs(y2-y1))
    innerW = out_w * (1 - 2*padding); innerH = out_h * (1 - 2*padding)
    scaleX = innerW / (tile_size * dx)
    scaleY = innerH / (tile_size * dy)
    z = math.floor(math.log2(max(1e-9, min(scaleX, scaleY))))
    return int(max(3, min(20, z)))

def _close_ring(ring):
    if not ring: return ring
    if ring[0][0]!=ring[-1][0] or ring[0][1]!=ring[-1][1]:
        ring = ring + [ring[0]]
    return ring

def _encoded_len_geojson(fc):
    s = json.dumps(fc, separators=(",",":"), ensure_ascii=False)
    return len(urllib.parse.quote(s, safe=""))

def _protect_corners_indices(ring_lonlat, angle_deg=18.0):
    keep = set([0, len(ring_lonlat)-1])
    for i in range(1, len(ring_lonlat)-1):
        x0,y0 = ring_lonlat[i-1]; x1,y1 = ring_lonlat[i]; x2,y2 = ring_lonlat[i+1]
        v1 = (x0-x1, y0-y1); v2 = (x2-x1, y2-y1)
        a1 = math.atan2(v1[1], v1[0]); a2 = math.atan2(v2[1], v2[0])
        ang = abs((a2 - a1 + math.pi)%(2*math.pi) - math.pi) * 180/math.pi
        if ang < angle_deg: keep.add(i)
    return keep

def _dp_with_forced_points_xy(points_xy, tol_m, forced_idx):
    import numpy as np
    pts = np.asarray(points_xy, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[list(forced_idx)] = True
    keep[0] = True; keep[-1] = True

    def seg_dp(a, b):
        p0, p1 = pts[a], pts[b]
        seg = p1 - p0
        L2 = (seg**2).sum() or 1e-12
        idx, dmax = -1, -1.0
        for i in range(a+1, b):
            if keep[i]:
                seg_dp(a, i); seg_dp(i, b)
                return
            v = pts[i] - p0
            t = max(0.0, min(1.0, (v@seg)/L2))
            proj = p0 + t*seg
            d = ((pts[i]-proj)**2).sum()**0.5
            if d > dmax: dmax, idx = d, i
        if dmax > tol_m and idx >= 0:
            keep[idx] = True
            seg_dp(a, idx); seg_dp(idx, b)

    seg_dp(0, len(pts)-1)
    out = [tuple(pts[i]) for i,flag in enumerate(keep) if flag]
    if out[0] != out[-1]: out.append(out[0])
    return out

def simplify_ring_for_url(
    ring_lonlat, ext84, out_w, out_h,
    padding=0.10, url_budget=7000,
    px_tol_min=0.6, px_tol_max=3.5, decimals=6,
    protect_corners=True, corner_angle_deg=18.0,
    include_shadow=True, shadow_opacity=0.55, poly_alpha=0.28
):
    ring_lonlat = _close_ring(ring_lonlat)
    # centro/zoom
    cx = (ext84[0]+ext84[1])/2.0; cy = (ext84[2]+ext84[3])/2.0
    z  = compute_zoom_for_bbox(ext84, out_w, out_h, padding=padding, tile_size=TILE_SIZE)
    mpp = meters_per_pixel(cy, z, tile_size=TILE_SIZE)

    ring_xy = [(_m_x(lon), _m_y(lat)) for lon,lat in ring_lonlat]
    forced = _protect_corners_indices(ring_lonlat, corner_angle_deg) if protect_corners else set()

    lo, hi = px_tol_min, px_tol_max
    best = None
    for _ in range(18):
        mid_px = (lo+hi)/2.0
        tol_m  = mid_px * mpp
        simp_xy = _dp_with_forced_points_xy(ring_xy, tol_m, forced)
        # volta p/ lon/lat arredondando
        simp_lonlat = [(round(_inv_m_x(x), decimals), round(_inv_m_y(y), decimals)) for x,y in simp_xy]

        feat_poly = {
            "type":"Feature","properties":{"stroke":"#ffff00","stroke-width":3,
                                           "fill":"#ffff00","fill-opacity":poly_alpha},
            "geometry":{"type":"Polygon","coordinates":[simp_lonlat]}
        }
        features = [feat_poly]
        if include_shadow:
            world = [[-179.999,-85], [179.999,-85], [179.999,85], [-179.999,85], [-179.999,-85]]
            features = [
                {"type":"Feature","properties":{"fill":"#000000","fill-opacity":shadow_opacity,"stroke-width":0},
                 "geometry":{"type":"Polygon","coordinates":[world, list(reversed(simp_lonlat))]}},
                feat_poly
            ]
        fc = {"type":"FeatureCollection","features":features}
        L = _encoded_len_geojson(fc)
        if L <= url_budget:
            best = (fc, {"z":z, "mpp":mpp, "px_tol":mid_px, "encoded_len":L, "ring":simp_lonlat})
            lo = mid_px
        else:
            hi = mid_px

    if best is None:
        # Sem sombra
        feat_poly = {
            "type":"Feature","properties":{"stroke":"#ffff00","stroke-width":3,
                                           "fill":"#ffff00","fill-opacity":poly_alpha},
            "geometry":{"type":"Polygon","coordinates":[ring_lonlat]}
        }
        fc = {"type":"FeatureCollection","features":[feat_poly]}
        best = (fc, {"z":z,"mpp":mpp,"px_tol":px_tol_min,"encoded_len":_encoded_len_geojson(fc),"ring":ring_lonlat})
    return best  # (fc, meta)

from PIL import ImageDraw, ImageFont

def _world_px(lon, lat, z, tile_size=TILE_SIZE):
    # pixel global no mundo (tile_size=512)
    n = tile_size * (2**z)
    x = _lon2x(lon) * n
    y = _lat2y(lat) * n
    return x, y

def _lonlat_ring_to_image_px(ring_lonlat, cx, cy, z, W, H, scale=2, tile_size=TILE_SIZE):
    cxw, cyw = _world_px(cx, cy, z, tile_size)
    Wp, Hp = W*scale, H*scale
    pts = []
    for lon,lat in ring_lonlat:
        xw, yw = _world_px(lon, lat, z, tile_size)
        x = (xw - cxw) * scale + Wp/2.0
        y = (yw - cyw) * scale + Hp/2.0
        pts.append((x,y))
    return pts  # lista de (x,y) em pixels da imagem final

def draw_overlay_locally(base_img, ring_lonlat, cx, cy, z, W, H, scale,
                         poly_alpha=0.28, darken_alpha=0.55,
                         stroke_px=3, stroke_rgb=(255,225,74)):
    """Desenha sombra + polígono amarelo em cima de base_img (Pillow)."""
    Wp, Hp = W*scale, H*scale
    base_img = base_img.convert("RGBA")

    # 1) sombra externa (máscara alpha com furo)
    if darken_alpha > 0:
        shade = Image.new("L", (Wp, Hp), int(darken_alpha*255))
        hole = Image.new("L", (Wp, Hp), 0)
        dr = ImageDraw.Draw(hole)
        pts = _lonlat_ring_to_image_px(ring_lonlat, cx, cy, z, W, H, scale)
        dr.polygon(pts, fill=255)  # 255 = área do imóvel
        # queremos escuro fora -> alpha = shade - hole
        import numpy as np
        alpha = np.array(shade, dtype=np.int16) - np.array(hole, dtype=np.int16)
        alpha = np.clip(alpha, 0, 255).astype("uint8")
        black = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
        black.putalpha(Image.fromarray(alpha))
        base_img = Image.alpha_composite(base_img, black)

    # 2) polígono amarelo
    overlay = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
    dr2 = ImageDraw.Draw(overlay)
    pts = _lonlat_ring_to_image_px(ring_lonlat, cx, cy, z, W, H, scale)
    fill = (255,255,0, int(max(0,min(1,poly_alpha))*255))
    dr2.polygon(pts, fill=fill, outline=stroke_rgb, width=max(1, int(stroke_px*scale)))
    base_img = Image.alpha_composite(base_img, overlay)

    return base_img.convert("RGB")

# ======== FIM DOS HELPERS NOVOS QUE BUSCAM SIMPLIFIAR AS FEIÇÕES CONFORME TAMANHO DA ÁREA ========


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

               # ===== 1) Enquadra e escolhe centro/zoom =====
        bbox = bbox_from_ring(ring)  # [minLon, minLat, maxLon, maxLat]
        cx, cy = centroid_from_ring(ring)
        if have_pin and q.get("center_on_pin", "0").lower() in ("1","true","yes"):
            cx, cy = float(lon), float(lat)
        zoom = compute_zoom_for_bbox(bbox, out_w, out_h, padding=pad)  # usa tile=512

        # ext84 no formato [minlon, maxlon, minlat, maxlat]
        ext84 = [bbox[0], bbox[2], bbox[1], bbox[3]]

        # ===== 2) Simplificação "em pixels" + proteção de cantos =====
        url_budget = int(q.get("url_budget", "7000"))  # margem contra 414/422
        fc, meta = simplify_ring_for_url(
            ring_lonlat=ring,
            ext84=ext84,
            out_w=out_w, out_h=out_h,
            padding=pad,
            url_budget=url_budget,
            px_tol_min=float(q.get("px_tol_min", "0.6")),
            px_tol_max=float(q.get("px_tol_max", "3.5")),
            protect_corners=True,
            corner_angle_deg=float(q.get("corner_angle_deg", "18")),
            include_shadow=(q.get("shadow", "1").lower() in ("1","true","yes")),
            shadow_opacity=darken,
            poly_alpha=poly_alpha
        )
        ring_simpl = meta["ring"]  # anel simplificado (lon/lat)

        # ===== 3) Decide: overlay via URL ou fallback local =====
        draw_local = (meta["encoded_len"] > url_budget)

        # Monta overlays se for usar no URL
        overlays = None
        if not draw_local:
            overlays_geojson = fc
            overlays_encoded = requests.utils.quote(
                json.dumps(overlays_geojson, separators=(',',':'))
            ) if requests else json.dumps(overlays_geojson)
            overlays = f"geojson({overlays_encoded})"
            # pin via Mapbox só se não for desenhar localmente
            if have_pin:
                pin_color = pin_color.lstrip("#")
                overlays += f",pin-s+{pin_color}({lon},{lat})"

        # ===== 4) Baixa a imagem base do Mapbox Static =====
        retina = f"@{scale}x" if scale in (2,3) else ""
        if overlays:
            base = f"https://api.mapbox.com/styles/v1/{style}/static/{overlays}/{cx},{cy},{zoom}/{out_w}x{out_h}{retina}"
        else:
            # sem overlay no URL (vai desenhar localmente)
            base = f"https://api.mapbox.com/styles/v1/{style}/static/{cx},{cy},{zoom}/{out_w}x{out_h}{retina}"

        static_url = f"{base}?access_token={token}&logo=false&attribution=false"

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
            safe_url = static_url.replace(token, "TOKEN")
            return jsonify({"error": f"Falha ao obter imagem estática do Mapbox: {e}", "debug_url": safe_url}), 400

        img = Image.open(io.BytesIO(map_bytes)).convert("RGBA")

        # ===== 5) Fallback: desenha sombra + polígono localmente se necessário =====
        if draw_local:
            img = draw_overlay_locally(
                base_img=img,
                ring_lonlat=ring_simpl,  # usa o simplificado protegido
                cx=cx, cy=cy, z=zoom,
                W=out_w, H=out_h, scale=scale,
                poly_alpha=poly_alpha,
                darken_alpha=darken,
                stroke_px=3, stroke_rgb=(255,225,74)
            )

        # ===== 6) Pin + rótulos (sempre por cima) =====
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if have_pin:
            # coordenadas em pixels (imagem)
            px, py = lonlat_to_image_px(lon, lat, cx, cy, zoom, out_w, out_h, tile=512, scale=scale)
            # se caiu no fallback, desenha o "alfinete" localmente
            if draw_local:
                rpx = max(3, int(5*scale))
                # borda branca
                draw.ellipse((px-rpx-2, py-rpx-2, px+rpx+2, py+rpx+2), fill=(255,255,255,255))
                # miolo na cor do pin
                pr,pg,pb = (int(pin_color[0:2],16), int(pin_color[2:4],16), int(pin_color[4:6],16))
                draw.ellipse((px-rpx, py-rpx, px+rpx, py+rpx), fill=(pr,pg,pb,255))
            # label do pin
            label = f"{lat:.5f}, {lon:.5f}"
            draw_text_with_stroke(draw, (px, py-16*scale), label, fill="white",
                                  stroke_fill="black", stroke_width=2*scale,
                                  font=font, anchor="mb")

        # ===== 7) Título CAR + Logo =====
        if car_text:
            pad_x = 16; pad_y = 14
            title = f"CAR: {car_text}"
            draw_text_with_stroke(draw, (pad_x, pad_y), title, fill="white",
                                  stroke_fill="black", stroke_width=2*scale,
                                  font=font, anchor="la")

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

        # ===== 8) JPEG saída =====
        out_rgb = img.convert("RGB")
        buf = io.BytesIO()
        q = int(q.get("jpg_quality", DEFAULT_JPG_Q))
        q = max(60, min(95, q))
        out_rgb.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg", download_name="mapa_smart_fazendas.jpg")


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
