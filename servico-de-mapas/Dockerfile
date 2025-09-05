# Dockerfile
FROM mambaorg/micromamba:1.5.8-jammy

# utilitários mínimos + tini
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Ambiente conda-forge com TODAS as deps, incluindo GUNICORN
RUN micromamba create -y -n app -c conda-forge \
    python=3.11 \
    flask \
    gunicorn \
    requests \
    geopandas cartopy matplotlib pillow gdal fiona shapely pyproj \
 && micromamba clean -a -y
SHELL ["bash", "-lc"]
ENV PATH=/opt/conda/envs/app/bin:$PATH

WORKDIR /app
COPY . /app

ENV PORT=5000
EXPOSE 5000

# tini como init, e gunicorn como servidor WSGI
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["gunicorn","app:app","-w","2","-k","gthread","-b","0.0.0.0:5000","--timeout","120"]
