# Dockerfile
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /apps

# Copiar arquivos
COPY . /apps

# Instalar dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expor porta
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
