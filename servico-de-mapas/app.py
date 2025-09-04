import os
import io
import geopandas as gpd
import matplotlib

# Importante: Define o backend do Matplotlib para 'Agg'.
# Isso permite que ele rode em um servidor sem ambiente gráfico (sem tela).
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from flask import Flask, request, send_file, jsonify

# Inicializa a aplicação web com Flask.
app = Flask(__name__)


# --- Endpoint da API ---

# Define a rota '/generate-map' que aceitará requisições do tipo POST.
@app.route('/generate-map', methods=['POST'])
def generate_map_endpoint():
    try:
        # Pega o conteúdo binário (o arquivo KML) enviado pelo n8n.
        kml_data = request.data
        if not kml_data:
            return jsonify({"error": "Corpo da requisição (KML) está vazio."}), 400

        print("Recebido KML. Lendo dados...")
        
        # Habilita o driver KML (forma segura dentro da função)
        with gpd.io.file.fiona.Env():
            gdf = gpd.read_file(io.BytesIO(kml_data), driver='KML')

        # Garante que a projeção dos dados esteja no padrão geográfico (WGS84).
        gdf = gdf.to_crs(epsg=4326)

        # Une todos os polígonos do KML em uma única geometria.
        imovel_geom = gdf.unary_union

        print("Preparando o mapa...")
        # Define a fonte das imagens de fundo. Usaremos OpenStreetMap.
        imagery = cimgt.OSM()

        # Cria a figura e o eixo do mapa com a projeção correta (Mercator).
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

        # Calcula a "caixa" (bounding box) que envolve o imóvel.
        bounds = imovel_geom.bounds
        
        # Adiciona uma margem de 10%.
        padding_x = (bounds[2] - bounds[0]) * 0.1
        padding_y = (bounds[3] - bounds[1]) * 0.1
        
        extent = [
            bounds[0] - padding_x,
            bounds[2] + padding_x,
            bounds[1] - padding_y,
            bounds[3] + padding_y,
        ]
        # Define a área de visualização do mapa.
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Adiciona a imagem de satélite/mapa de fundo.
        ax.add_image(imagery, 12, interpolation='spline36')

        print("Desenhando polígono...")
        # Adiciona a geometria do imóvel por cima do mapa de fundo.
        ax.add_geometries([imovel_geom], crs=ccrs.PlateCarree(),
                          facecolor='yellow', edgecolor='yellow', linewidth=2, alpha=0.3)

        # Salva a imagem gerada em um buffer na memória.
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        img_buffer.seek(0)

        print("Imagem gerada. Enviando resposta...")
        # Envia a imagem de volta para o n8n.
        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        print(f"Erro no processamento: {e}")
        return jsonify({"error": str(e)}), 500

