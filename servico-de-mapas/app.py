import os, io, json, base64, math, logging
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from shapely.geometry import GeometryCollection
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB limite de payload

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate-map")

def _extract_kml_bytes(payload):
    """
    Aceita:
      - [{"data": "<kml>"}]  ou  {"data": "<kml>"}
      - Base64 com/sem prefixo data:...;base64,
    Retorna bytes do KML ou lança ValueError.
    """
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
    bio = io.BytesIO(kml_bytes)
    try:
        return gpd.read_file(bio, driver="LIBKML")
    except Exception:
        bio.seek(0)
        try:
            return gpd.read_file(bio, driver="KML")
        except Exception as e:
            raise RuntimeError(
                "Falha lendo KML: nenhum driver (LIBKML/KML) disponível no GDAL/Fiona. "
                f"Detalhes: {e}"
            )

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
    # procura coluna com nome compatível
    for col in gdf.columns:
        lc = col.lower()
        if lc == "cod_imovel" or "cod_imovel" in lc:
            series = gdf[col].dropna()
            if not series.empty:
                cod = str(series.iloc[0]).strip()
                if cod:
                    return cod
    return None

@app.route("/generate-map", methods=["POST"])
def generate_map_endpoint():
    try:
        payload = request.get_json(silent=True)
        kml_bytes = _extract_kml_bytes(payload)

        gdf = _read_kml_gdf(kml_bytes)
        if gdf.empty or gdf.geometry.isna().all():
            return jsonify({"error": "KML sem geometria válida."}), 400

        gdf = gdf.to_crs(epsg=4326)
        imovel_geom = gdf.unary_union
        if isinstance(imovel_geom, GeometryCollection) and imovel_geom.is_empty:
            return jsonify({"error": "Geometria vazia após união."}), 400

        # define tiles (OSM). Para alto volume, considere provedor com SLA/WMTS.
        imagery = cimgt.OSM(cache=True)

        fig = Figure(figsize=(10, 8), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

        extent = _extent_with_padding(imovel_geom, pad_ratio=0.10)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        zoom = _zoom_from_lon_span(extent[0], extent[1])
        try:
            ax.add_image(imagery, zoom, interpolation="spline36")
        except Exception as e_tiles:
            log.warning(f"Falha ao carregar tiles OSM: {e_tiles}. Prosseguindo sem fundo...")

        ax.add_geometries(
            [imovel_geom],
            crs=ccrs.PlateCarree(),
            facecolor=(1.0, 1.0, 0.0, 0.25),
            edgecolor=(1.0, 1.0, 0.0, 0.9),
            linewidth=2,
        )
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # nome do arquivo pelo cod_imovel, se disponível
        download_name = (_extract_cod_imovel(gdf) or "mapa") + ".png"
        return send_file(buf, mimetype="image/png", download_name=download_name)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log.exception("Erro inesperado")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Dev: flask run ou python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
