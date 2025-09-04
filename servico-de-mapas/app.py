import os
import io
import geopandas as gpd
import matplotlib

# Importante: Define o backend do Matplotlib para 'Agg'.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from flask import Flask, request, send_file, jsonify

# --- Configuração Inicial ---

# ✅ CORREÇÃO: A linha problemática foi REMOVIDA.
# Versões modernas do Geopandas não precisam mais disso e a linha estava causando o erro.
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

# Inicializa a aplicação web com Flask.
app = Flask(__name__)


# --- Endpoint da API ---

@app.route('/generate-map', methods=['POST'])
def generate_map_endpoint():
    try:
        kml_data = request.data
        if not kml_data:
            return jsonify({"error": "Corpo da requisição (KML) está vazio."}), 400

        print("Recebido KML. Lendo dados...")
        # Lemos o KML diretamente da memória. O Geopandas encontrará o driver KML automaticamente.
        gdf = gpd.read_file(io.BytesIO(kml_data), driver='KML')

        gdf = gdf.to_crs(epsg=4326)
        imovel_geom = gdf.unary_union

        print("Preparando o mapa...")
        imagery = cimgt.OSM()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

        bounds = imovel_geom.bounds
        padding_x = (bounds[2] - bounds[0]) * 0.1
        padding_y = (bounds[3] - bounds[1]) * 0.1
        extent = [bounds[0] - padding_x, bounds[2] + padding_x, bounds[1] - padding_y, bounds[3] + padding_y]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_image(imagery, 12, interpolation='spline36')

        print("Desenhando polígono...")
        ax.add_geometries([imovel_geom], crs=ccrs.PlateCarree(),
                          facecolor='yellow', edgecolor='yellow', linewidth=2, alpha=0.3)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        img_buffer.seek(0)

        print("Imagem gerada. Enviando resposta...")
        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        print(f"Erro no processamento: {e}")
        return jsonify({"error": str(e)}), 500
