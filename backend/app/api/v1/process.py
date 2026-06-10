"""
POST /api/v1/process   — Ejecuta el pipeline YOLO sobre un job ya subido.
POST /api/v1/compare   — Compara varios modelos sobre el mismo job.
GET  /api/v1/health    — Estado del servicio y modelos disponibles.
GET  /api/v1/tiles/{job_id}/{tile_filename} — Sirve tiles generados.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ModelResult,
    ProcessRequest,
    ProcessResponse,
    DetectionItem,
)
from app.services.yolo_service import (
    UPLOAD_DIR,
    VALID_MODELS,
    get_tiles_dir,
    run_compare_pipeline,
    run_pipeline,
)

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_uploaded_file(job_id: str) -> str:
    """Busca el archivo subido para el job_id dado (cualquier extensión)."""
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        candidate = UPLOAD_DIR / f"{job_id}{ext}"
        if candidate.exists():
            return str(candidate)
    raise HTTPException(
        status_code=404,
        detail=f"No se encontró el archivo para job_id='{job_id}'. ¿Hiciste el upload primero?"
    )


def _build_detections(raw: list[dict]) -> list[DetectionItem]:
    return [DetectionItem(**d) for d in raw]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/process", response_model=ProcessResponse, summary="Ejecutar pipeline YOLO")
def process(req: ProcessRequest) -> ProcessResponse:
    """
    Lanza el pipeline completo: tiling → inferencia YOLO → NMS global.

    - **job_id**: obtenido del endpoint /upload
    - **model_key**: yolov8n | yolov9c | yolo11n  (default: yolo11n)
    - **conf**: umbral de confianza (default: 0.25)
    - **iou**: umbral IoU para NMS (default: 0.45)
    - **tile_size**: tamaño de tile en píxeles (default: 1024)
    - **overlap**: solapamiento entre tiles (default: 128)
    """
    if req.model_key not in VALID_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model_key '{req.model_key}' no válido. Opciones: {VALID_MODELS}"
        )

    file_path = _find_uploaded_file(req.job_id)

    try:
        result = run_pipeline(
            job_id    = req.job_id,
            file_path = file_path,
            model_key = req.model_key,
            conf      = req.conf,
            iou       = req.iou,
            tile_size = req.tile_size,
            overlap   = req.overlap,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ProcessResponse(
        job_id                = result["job_id"],
        model_key             = result["model_key"],
        tree_count            = result["tree_count"],
        tiles_processed       = result["tiles_processed"],
        tiles_with_detections = result["tiles_with_detections"],
        elapsed_sec           = result["elapsed_sec"],
        tiles_per_sec         = result["tiles_per_sec"],
        detecciones           = _build_detections(result["detecciones"]),
    )


@router.post("/compare", response_model=CompareResponse, summary="Comparar modelos YOLO")
def compare(req: CompareRequest) -> CompareResponse:
    """
    Ejecuta el pipeline para cada modelo listado y retorna comparativa lado a lado.
    Los tiles se generan una única vez y se reusan entre modelos.
    """
    # Filtrar modelos válidos
    valid_requested = [m for m in req.models if m in VALID_MODELS]
    if not valid_requested:
        raise HTTPException(
            status_code=422,
            detail=f"Ningún modelo válido en la lista. Opciones: {VALID_MODELS}"
        )

    file_path = _find_uploaded_file(req.job_id)

    try:
        result = run_compare_pipeline(
            job_id    = req.job_id,
            file_path = file_path,
            models    = valid_requested,
            conf      = req.conf,
            iou       = req.iou,
            tile_size = req.tile_size,
            overlap   = req.overlap,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model_results = [
        ModelResult(
            model_key             = r["model_key"],
            tree_count            = r["tree_count"],
            tiles_processed       = r["tiles_processed"],
            tiles_with_detections = r["tiles_with_detections"],
            elapsed_sec           = r["elapsed_sec"],
            tiles_per_sec         = r["tiles_per_sec"],
            detecciones           = _build_detections(r["detecciones"]),
        )
        for r in result["results"]
    ]

    return CompareResponse(job_id=req.job_id, results=model_results)


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Retorna estado del servicio y lista de modelos disponibles."""
    return HealthResponse(status="ok", models_available=VALID_MODELS)


@router.get(
    "/tiles/{job_id}/{tile_filename}",
    summary="Servir tile generado",
    response_class=FileResponse,
)
def serve_tile(job_id: str, tile_filename: str) -> FileResponse:
    """
    Sirve un tile JPG generado durante el proceso de tiling.
    Ruta de almacenamiento: /tmp/yolov-uploads/{job_id}_tiles/{tile_filename}
    """
    # Seguridad: evitar path traversal
    if ".." in tile_filename or "/" in tile_filename:
        raise HTTPException(status_code=400, detail="Nombre de tile inválido.")

    tile_path = get_tiles_dir(job_id) / tile_filename
    if not tile_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Tile '{tile_filename}' no encontrado para job_id='{job_id}'."
        )

    return FileResponse(str(tile_path), media_type="image/jpeg")
