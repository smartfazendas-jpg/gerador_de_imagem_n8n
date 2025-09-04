# Usa a imagem oficial do OSGeo/GDAL que já vem com Python 3.9 e TODO o ambiente GIS.
FROM osgeo/gdal:python3.9-ubuntu-full

# Instala o pip, o gerenciador de pacotes do Python.
RUN apt-get update && apt-get install -y python3-pip

# Define o diretório de trabalho.
WORKDIR /app

# Copia a nossa lista de dependências Python.
COPY requirements.txt .

# Agora, instala as bibliotecas Python usando pip.
# Como o ambiente já está preparado, a instalação será bem-sucedida.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da nossa aplicação.
COPY . .

# Expõe a porta 80.
EXPOSE 80

# Comando para iniciar o servidor web Flask.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
