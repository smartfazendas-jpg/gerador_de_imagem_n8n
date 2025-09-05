# Dockerfile
FROM mambaorg/micromamba:1.5.8-jammy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Pinar versões evita API diferente do Cartopy e erros de binários
RUN micromamba create -y -n app -c conda-forge \
    python=3.11 \
    flask \
    gunicorn \
    requests \
    geopandas=0.14.* \
    cartopy=0.23.* \
    shapely=2.* \
    fiona=1.9.* \
    gdal=3.8.* \
    pyproj=3.6.* \
    matplotlib=3.8.* \
    pillow \
 && micromamba clean -a -y

SHELL ["bash","-lc"]
ENV PATH=/opt/conda/envs/app/bin:$PATH

# Corrige o erro do PROJ/GDAL nos logs
ENV PROJ_LIB=/opt/conda/envs/app/share/proj
ENV GDAL_DATA=/opt/conda/envs/app/share/gdal

WORKDIR /app
COPY . /app

EXPOSE 5000

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["gunicorn","app:app","-w","2","-k","gthread","-b","0.0.0.0:5000","--timeout","120"]
