# Use uma imagem base do Python
FROM python:3.11-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY . /app

# Instalar as dependências necessárias
RUN pip install --no-cache-dir \
    yfinance \
    numpy \
    pandas \
    scikit-learn \
    tensorflow \
    fastapi \
    uvicorn \
    absl-py

# Expor a porta que o FastAPI usará
EXPOSE 8888

# Comando para iniciar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
