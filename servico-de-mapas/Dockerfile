# Usa uma imagem oficial do Python como base. A vers‹o 'slim' Ž mais leve.
FROM python:3.9-slim

# Define o diret—rio de trabalho dentro do container.
WORKDIR /app

# Copia o arquivo com a lista de dependncias para dentro do container.
COPY requirements.txt .

# Instala as dependncias listadas no requirements.txt.
# A instala‹o do cartopy pode levar alguns minutos.
RUN apt-get update && apt-get install -y libgeos-dev && \
    pip install --no-cache-dir -r requirements.txt

# Copia todos os outros arquivos (o app.py) para o diret—rio de trabalho.
COPY . .

# Exp›e a porta 80, que Ž a porta padr‹o para servios web no EasyPanel.
EXPOSE 80

# O comando que ser‡ executado quando o container iniciar.
# Ele inicia o servidor web Flask e o torna acess’vel a partir de outros servios.
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
