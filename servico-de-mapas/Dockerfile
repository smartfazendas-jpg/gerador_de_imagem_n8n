# Usa uma imagem oficial do Python como base. A versão 'slim' é mais leve.
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container.
WORKDIR /app

# ✅ CORREÇÃO: Instala as dependências de sistema (GDAL e GEOS) antes de tudo.
# Isso é crucial para que o geopandas e o cartopy funcionem corretamente.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo com a lista de dependências Python.
COPY requirements.txt .

# Agora, instala as bibliotecas Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da nossa aplicação.
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor web Flask.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
