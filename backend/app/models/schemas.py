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
    model_key: str = "yolo11n_forestai"
    conf: float = Field(default=0.55, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.40, ge=0.0, le=1.0)
    centroid_dist_px: int = Field(default=60, ge=0, le=200)
    tile_size: int = Field(default=1024, ge=128, le=4096)
    overlap: int = Field(default=200, ge=0, le=512)


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
    conf: float = Field(default=0.30, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.40, ge=0.0, le=1.0)
    centroid_dist_px: int = Field(default=40, ge=0, le=200)
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


# ── Classify (Species + Health) ──────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    job_id: str
    sample_tiles: int = Field(default=20, ge=1, le=100)
    max_crops_per_tile: int = Field(default=15, ge=1, le=50)
    concurrency: int = Field(default=5, ge=1, le=10)


class ClassifiedDetection(BaseModel):
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
    species: str
    health: str
    species_confidence: float
    cluster_id: int


class SpeciesSummary(BaseModel):
    species: str
    count: int
    pct: float
    avg_confidence: float
    health_saludable: int
    health_estresado: int
    health_enfermo: int


class ClassifyResponse(BaseModel):
    job_id: str
    total_trees: int
    classified_trees: int
    n_clusters: int
    tiles_sampled: int
    elapsed_sec: float
    species_summary: List[SpeciesSummary]
    detecciones: List[ClassifiedDetection]


# ── Classify Species Prod (YOLO fine-tuned + VLM fallback) ───────────────────

class SpeciesProdRequest(BaseModel):
    job_id: str
    conf_fallback: float = Field(default=0.50, ge=0.0, le=1.0,
        description="Umbral de confianza: por debajo activa el fallback VLM")

class SpeciesProdDetection(BaseModel):
    tile_filename: str
    x1: int; y1: int; x2: int; y2: int
    global_x1: int; global_y1: int; global_x2: int; global_y2: int
    confidence: float
    especie: str
    conf_especie: float
    via: str   # "yolo" | "vlm_fallback" | "error"

class SpeciesProdSummary(BaseModel):
    especie: str
    count: int
    pct: float
    avg_conf: float
    via_yolo: int
    via_vlm: int

class SpeciesProdResponse(BaseModel):
    job_id: str
    total_trees: int
    elapsed_sec: float
    resumen: List[SpeciesProdSummary]
    detecciones: List[SpeciesProdDetection]


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


# ── Histórico Satelital ─────────────────────────────────────────────────────

class HistoricoBBox(BaseModel):
    """Bounding box geográfico (WGS84)"""
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)
    nombre_zona: Optional[str] = None


class YearlyLoss(BaseModel):
    year: int
    loss_ha: float


class HistoricoResponse(BaseModel):
    """Métricas históricas de cobertura forestal"""
    bbox: HistoricoBBox
    area_total_ha: float
    cobertura_2000_ha: float
    cobertura_2000_pct: float
    perdida_total_ha: float
    perdida_total_pct: float
    perdida_por_year: List[YearlyLoss]
    year_pico_perdida: int
    perdida_year_pico_ha: float
    tasa_anual_promedio_ha: float
    timelapse_url: str
    timelapse_embed_url: str
    fuente: str = "Hansen Global Forest Change v1.11 (2001-2023) + Google Earth Timelapse"


class ChatHistoricoRequest(BaseModel):
    """Chat sobre un análisis histórico ya generado"""
    pregunta: str
    contexto: HistoricoResponse
    historial: Optional[List[dict]] = None


class ChatHistoricoResponse(BaseModel):
    respuesta: str
    modelo: str = "llama-3.3-70b-versatile"
