# Usa a imagem oficial e mais recente do OSGeo que está no GitHub Container Registry (ghcr.io)
# Esta imagem já vem com GDAL e todas as dependências de sistema corretamente configuradas.
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Define o diretório de trabalho.
WORKDIR /app

# A imagem base já vem com Python e pip.
# Copia a nossa lista de dependências Python.
COPY requirements.txt .

# Instala as bibliotecas Python.
# O --break-system-packages é necessário em imagens mais novas para permitir a instalação via pip.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copia o código da nossa aplicação.
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor web Flask.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]

