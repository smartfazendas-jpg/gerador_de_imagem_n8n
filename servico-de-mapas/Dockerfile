# Usa uma imagem oficial do Miniconda como base, que é leve e vem com o Conda.
FROM continuumio/miniconda3

# Define o diretório de trabalho.
WORKDIR /app

# Copia o nosso novo arquivo de ambiente para o container.
COPY environment.yml .

# O comando mágico: O Conda lê o arquivo e instala TUDO (Python, GDAL, Geopandas, etc.)
# em um ambiente isolado e 100% compatível. Isso pode demorar vários minutos.
RUN conda env create -f environment.yml

# Define o shell para usar o ambiente que acabamos de criar.
SHELL ["conda", "run", "-n", "gis-environment", "/bin/bash", "-c"]

# Copia o resto do nosso código (o app.py).
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor Flask usando o ambiente Conda correto.
CMD ["conda", "run", "-n", "gis-environment", "flask", "run", "--host=0.0.0.0", "--port=80"]
