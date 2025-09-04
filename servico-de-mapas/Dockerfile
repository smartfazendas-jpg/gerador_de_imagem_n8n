# ✅ CORREÇÃO: Usa uma imagem oficial e específica do OSGeo/GDAL que existe.
FROM osgeo/gdal:ubuntu-full-3.8.4-python3.10

# Instala o pip, o gerenciador de pacotes do Python.
RUN apt-get update && apt-get install -y python3-pip

# Define o diretório de trabalho.
WORKDIR /app

# Copia a nossa lista de dependências Python.
COPY requirements.txt .

# Agora, instala as bibliotecas Python usando pip.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da nossa aplicação.
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor web Flask.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
