# Imagem com conda-forge pré-configurada (resolve GDAL/GEOS/PROJ)
FROM mambaorg/micromamba:1.5.8-jammy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Cria env com todas as libs nativas prontas
RUN micromamba create -y -n app -c conda-forge \
    python=3.11 flask requests geopandas cartopy matplotlib pillow gdal fiona shapely pyproj \
 && micromamba clean -a -y
SHELL ["bash", "-lc"]
ENV PATH=/opt/conda/envs/app/bin:$PATH

WORKDIR /app
COPY . /app

# Porta do serviço
ENV PORT=5000
EXPOSE 5000

# Produção: gunicorn (mais estável que flask run)
CMD ["bash", "-lc", "tini -g -- gunicorn app:app -w 2 -k gthread -b 0.0.0.0:${PORT} --timeout 120"]
