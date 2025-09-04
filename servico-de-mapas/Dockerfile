# ✅ CORREÇÃO FINAL E DEFINITIVA: Começa com uma imagem oficial e garantida do Ubuntu.
FROM ubuntu:22.04

# Evita que a instalação peça por inputs interativos.
ENV DEBIAN_FRONTEND=noninteractive

# Instala as dependências de sistema essenciais: Python, PIP, e as bibliotecas GIS.
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho.
WORKDIR /app

# Copia a nossa lista de dependências Python.
COPY requirements.txt .

# Agora, instala as bibliotecas Python usando pip.
# Como as dependências de sistema já estão presentes, a instalação funcionará.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia o código da nossa aplicação.
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor web Flask.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
