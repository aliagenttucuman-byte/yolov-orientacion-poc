"""
POST /api/v1/upload
Acepta multipart/form-data con campo 'file'.
Guarda la imagen con un UUID como nombre y retorna el job_id.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import UploadResponse

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/yolov-uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


@router.post("/upload", response_model=UploadResponse, summary="Subir ortofoto")
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Sube una imagen (GeoTIFF, JPG o PNG) al servidor.
    Retorna un job_id que se usará en los endpoints de procesamiento.
    """
    # Validar extensión
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Extensión '{suffix}' no soportada. "
                f"Usa una de: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # Generar nombre único con UUID, conservando extensión
    job_id    = str(uuid.uuid4())
    dest_name = f"{job_id}{suffix}"
    dest_path = UPLOAD_DIR / dest_name

    # Guardar archivo
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        job_id     = job_id,
        filename   = file.filename or dest_name,
        size_bytes = len(content),
        path       = str(dest_path),
    )
