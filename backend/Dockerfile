# ── YOLOv Orientación PoC — Backend ─────────────────────────────────────────
# Contexto de build: raíz del proyecto (un nivel arriba de backend/)
# docker build -f backend/Dockerfile .

FROM python:3.11-slim

# Dependencias del sistema necesarias para rasterio y opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python primero (capa cacheada)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY backend/app/ ./app/

# Copiar el pipeline (está un nivel arriba del backend)
COPY pipeline/ ./pipeline/

# Crear directorio de uploads
RUN mkdir -p /tmp/yolov-uploads

# Variables de entorno
ENV UPLOAD_DIR=/tmp/yolov-uploads \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]
