"""
YOLOService — orquesta tiler + detector del pipeline para la API.
El directorio raíz del proyecto se agrega a sys.path para poder importar pipeline.*
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# ── Asegurar que pipeline.* sea importable ───────────────────────────────────
# En producción (Docker) el pipeline se copia en /app/pipeline.
# En desarrollo local apuntamos al directorio raíz del proyecto.
_PIPELINE_CANDIDATES = [
    Path("/app"),                                                        # Docker
    Path(__file__).resolve().parents[4],                                 # …/backend/../../../ → raíz proyecto
    Path(__file__).resolve().parents[3],                                 # fallback
]
for _candidate in _PIPELINE_CANDIDATES:
    if (_candidate / "pipeline").is_dir():
        _root = str(_candidate)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

from pipeline.tiler    import generate_tiles                             # noqa: E402
from pipeline.detector import (                                          # noqa: E402
    load_model, run_inference, nms_global, SUPPORTED_MODELS,
)

# ── Configuración ────────────────────────────────────────────────────────────
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR", "/tmp/yolov-uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VALID_MODELS: List[str] = list(SUPPORTED_MODELS.keys())


# ── Funciones públicas ───────────────────────────────────────────────────────

def get_tiles_dir(job_id: str) -> Path:
    return UPLOAD_DIR / f"{job_id}_tiles"


def run_pipeline(
    job_id:     str,
    file_path:  str,
    model_key:  str  = "yolo11n",
    conf:       float = 0.25,
    iou:        float = 0.45,
    tile_size:  int  = 1024,
    overlap:    int  = 128,
) -> dict:
    """
    Pipeline completo: generate_tiles → load_model → run_inference → nms_global.

    Retorna un dict compatible con ProcessResponse.
    """
    if model_key not in VALID_MODELS:
        raise ValueError(
            f"Modelo '{model_key}' no soportado. Opciones: {VALID_MODELS}"
        )

    tiles_dir = str(get_tiles_dir(job_id))

    # 1. Tiling
    tiles_meta = generate_tiles(
        tif_path   = file_path,
        output_dir = tiles_dir,
        tile_size  = tile_size,
        overlap    = overlap,
    )

    # 2. Carga de modelo
    model = load_model(model_key)

    # 3. Inferencia
    detections_raw, elapsed = run_inference(
        model     = model,
        tiles     = tiles_meta,
        tiles_dir = tiles_dir,
        conf      = conf,
        iou       = iou,
    )

    # 4. NMS global (eliminar duplicados en solapamientos)
    detections = nms_global(detections_raw, iou_threshold=iou)

    # 5. Calcular estadísticas
    tiles_processed = len(tiles_meta)
    tiles_with_dets = len({d["tile_filename"] for d in detections})
    tiles_per_sec   = round(tiles_processed / elapsed, 2) if elapsed > 0 else 0.0

    return {
        "job_id":                 job_id,
        "model_key":              model_key,
        "tree_count":             len(detections),
        "tiles_processed":        tiles_processed,
        "tiles_with_detections":  tiles_with_dets,
        "elapsed_sec":            round(elapsed, 3),
        "tiles_per_sec":          tiles_per_sec,
        "detecciones":            detections,
    }


def run_compare_pipeline(
    job_id:    str,
    file_path: str,
    models:    List[str],
    conf:      float = 0.25,
    iou:       float = 0.45,
    tile_size: int   = 1024,
    overlap:   int   = 128,
) -> dict:
    """
    Ejecuta run_pipeline para cada modelo y devuelve comparativa.
    Los tiles se generan una sola vez y se reusan.
    """
    results = []
    tiles_dir = str(get_tiles_dir(job_id))

    # Generar tiles una sola vez
    tiles_meta = generate_tiles(
        tif_path   = file_path,
        output_dir = tiles_dir,
        tile_size  = tile_size,
        overlap    = overlap,
    )

    for model_key in models:
        if model_key not in VALID_MODELS:
            continue

        model = load_model(model_key)
        detections_raw, elapsed = run_inference(
            model     = model,
            tiles     = tiles_meta,
            tiles_dir = tiles_dir,
            conf      = conf,
            iou       = iou,
        )
        detections     = nms_global(detections_raw, iou_threshold=iou)
        tiles_with_dets = len({d["tile_filename"] for d in detections})
        tiles_per_sec   = round(len(tiles_meta) / elapsed, 2) if elapsed > 0 else 0.0

        results.append({
            "model_key":              model_key,
            "tree_count":             len(detections),
            "tiles_processed":        len(tiles_meta),
            "tiles_with_detections":  tiles_with_dets,
            "elapsed_sec":            round(elapsed, 3),
            "tiles_per_sec":          tiles_per_sec,
            "detecciones":            detections,
        })

    return {
        "job_id":  job_id,
        "results": results,
    }
