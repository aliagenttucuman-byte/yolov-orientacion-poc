"""
FastAPI application — YOLOv Orientación PoC
Puerto: 8020
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import historico, process, results, upload

# ── Crear directorios necesarios ─────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/yolov-uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Aplicación ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "YOLOv Orientación PoC — Tree Detection API",
    description = (
        "API para detección de árboles en ortofotos aéreas usando YOLO.\n\n"
        "Flujo típico:\n"
        "1. `POST /api/v1/upload`   → sube una ortofoto, recibe `job_id`\n"
        "2. `POST /api/v1/process`  → corre el pipeline YOLO, recibe detecciones\n"
        "3. `POST /api/v1/compare`  → compara varios modelos sobre el mismo job\n"
        "4. `GET  /api/v1/results/{job_id}` → consulta resultados guardados\n"
        "5. `GET  /api/v1/tiles/{job_id}/{tile_filename}` → visualiza tiles\n"
        "6. `GET  /api/v1/health`   → estado del servicio\n"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],          # Ajustar en producción
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(upload.router,    prefix=API_PREFIX, tags=["Upload"])
app.include_router(process.router,   prefix=API_PREFIX, tags=["Pipeline"])
app.include_router(results.router,   prefix=API_PREFIX, tags=["Results"])
app.include_router(historico.router, prefix=API_PREFIX, tags=["Histórico"])


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "YOLOv Orientación PoC — Tree Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }
