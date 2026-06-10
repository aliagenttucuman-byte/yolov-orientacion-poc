"""
Pydantic v2 schemas para la API de detección de árboles.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Upload ──────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    size_bytes: int
    path: str


# ── Process ─────────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    job_id: str
    model_key: str = "yolo11n"
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    tile_size: int = Field(default=1024, ge=128, le=4096)
    overlap: int = Field(default=128, ge=0, le=512)


class DetectionItem(BaseModel):
    tile_filename: str
    x1: int
    y1: int
    x2: int
    y2: int
    global_x1: int
    global_y1: int
    global_x2: int
    global_y2: int
    confidence: float


class ProcessResponse(BaseModel):
    job_id: str
    model_key: str
    tree_count: int
    tiles_processed: int
    tiles_with_detections: int
    elapsed_sec: float
    tiles_per_sec: float
    detecciones: List[DetectionItem]


# ── Compare ──────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    job_id: str
    models: List[str] = Field(default=["yolov8n", "yolov9c", "yolo11n"])
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    tile_size: int = Field(default=1024, ge=128, le=4096)
    overlap: int = Field(default=128, ge=0, le=512)


class ModelResult(BaseModel):
    model_key: str
    tree_count: int
    tiles_processed: int
    tiles_with_detections: int
    elapsed_sec: float
    tiles_per_sec: float
    detecciones: List[DetectionItem]


class CompareResponse(BaseModel):
    job_id: str
    results: List[ModelResult]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_available: List[str]


# ── Job result (almacenamiento en memoria) ───────────────────────────────────

class JobResult(BaseModel):
    job_id: str
    original_filename: str
    file_path: str
    size_bytes: int
    process_response: Optional[ProcessResponse] = None
    compare_response: Optional[CompareResponse] = None
    status: str = "uploaded"   # uploaded | processing | done | error
    error: Optional[str] = None
