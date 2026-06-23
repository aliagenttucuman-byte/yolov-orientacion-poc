"""
POST /api/v1/process   — Ejecuta el pipeline YOLO sobre un job ya subido.
POST /api/v1/compare   — Compara varios modelos sobre el mismo job.
POST /api/v1/classify  — Clasifica especies + salud con LLM Vision + CLIP clustering.
GET  /api/v1/health    — Estado del servicio y modelos disponibles.
GET  /api/v1/tiles/{job_id}/{tile_filename} — Sirve tiles generados.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    ClassifiedDetection,
    ClassifyRequest,
    ClassifyResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ModelResult,
    ProcessRequest,
    ProcessResponse,
    DetectionItem,
    SpeciesSummary,
    SpeciesProdRequest,
    SpeciesProdDetection,
    SpeciesProdSummary,
    SpeciesProdResponse,
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
            nms_iou   = req.nms_iou,
            centroid_dist_px = req.centroid_dist_px,
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
            nms_iou   = req.nms_iou,
            centroid_dist_px = req.centroid_dist_px,
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


@router.post("/classify-species", response_model=SpeciesProdResponse, summary="Clasificar especies (YOLO + fallback VLM)")
def classify_species_prod(req: SpeciesProdRequest) -> SpeciesProdResponse:
    """
    Clasifica la especie de cada copa ya detectada usando:
    1. yolo26n_especies (fine-tuned NOA, rápido, sin costo)
    2. Si conf < conf_fallback → gpt-4o-mini refina (fallback)

    Requiere que el job haya sido procesado con /process primero.
    """
    from app.services.yolo_service import _job_results
    job_data = _job_results.get(req.job_id)
    if not job_data or not job_data.get("detecciones"):
        raise HTTPException(
            status_code=404,
            detail=f"No hay detecciones en memoria para job_id='{req.job_id}'. Ejecutá /process primero."
        )

    tiles_dir = str(get_tiles_dir(req.job_id))

    import sys
    from pathlib import Path as P
    pipeline_root = P("/app") if P("/app/pipeline").is_dir() else P(__file__).resolve().parent.parent.parent.parent
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))

    try:
        from pipeline.species_classifier_prod import classify_detections
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"species_classifier_prod no disponible: {e}")

    import copy, time
    detecciones = copy.deepcopy(job_data["detecciones"])

    t0 = time.time()
    try:
        enriched = classify_detections(
            detections=detecciones,
            tiles_dir=tiles_dir,
            conf_threshold=req.conf_fallback,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - t0, 2)

    # Resumen por especie
    from collections import defaultdict
    sp_map: dict = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "via_yolo": 0, "via_vlm": 0})
    for d in enriched:
        sp = d.get("especie", "Otro")
        sp_map[sp]["count"] += 1
        sp_map[sp]["conf_sum"] += d.get("conf_especie", 0.0)
        if d.get("via") == "vlm_fallback":
            sp_map[sp]["via_vlm"] += 1
        else:
            sp_map[sp]["via_yolo"] += 1

    total = len(enriched)
    resumen = [
        SpeciesProdSummary(
            especie=sp,
            count=v["count"],
            pct=round(v["count"] / total * 100, 1) if total else 0,
            avg_conf=round(v["conf_sum"] / v["count"], 3) if v["count"] else 0,
            via_yolo=v["via_yolo"],
            via_vlm=v["via_vlm"],
        )
        for sp, v in sorted(sp_map.items(), key=lambda x: -x[1]["count"])
    ]

    det_list = [SpeciesProdDetection(**d) for d in enriched]

    return SpeciesProdResponse(
        job_id=req.job_id,
        total_trees=total,
        elapsed_sec=elapsed,
        resumen=resumen,
        detecciones=det_list,
    )


@router.post("/classify", response_model=ClassifyResponse, summary="Clasificar especies + salud")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Clasifica especies y salud de los árboles detectados usando:
    1. LLM Vision (gpt-4o-mini) sobre tiles sampleados
    2. CLIP embeddings + clustering HDBSCAN
    3. Asignación de especie por cluster

    Requiere que el job haya sido procesado con /process primero (tiles en disco).
    """
    tiles_dir = get_tiles_dir(req.job_id)
    if not tiles_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron tiles para job_id='{req.job_id}'. Ejecutá /process primero."
        )

    # Reconstruir detecciones desde los tiles en disco
    tile_files = list(tiles_dir.glob("*.jpg"))
    if not tile_files:
        raise HTTPException(status_code=404, detail="No hay tiles generados para este job.")

    # Cargar detecciones del job desde el estado en memoria del yolo_service
    from app.services.yolo_service import _job_results
    job_data = _job_results.get(req.job_id)
    if not job_data or not job_data.get("detecciones"):
        raise HTTPException(
            status_code=404,
            detail=f"No hay detecciones en memoria para job_id='{req.job_id}'. Ejecutá /process primero."
        )

    detecciones_raw = job_data["detecciones"]

    # Importar y correr el clasificador en thread pool (asyncio dentro)
    try:
        import sys
        from pathlib import Path as P
        pipeline_root = P("/app") if P("/app/pipeline").is_dir() else P(__file__).resolve().parent.parent.parent.parent
        if str(pipeline_root) not in sys.path:
            sys.path.insert(0, str(pipeline_root))
        from pipeline.species_classifier import classify_species_sync
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"species_classifier no disponible: {e}")

    t0 = time.time()
    try:
        enriched = classify_species_sync(
            tiles_dir=tiles_dir,
            detecciones=detecciones_raw,
            sample_tiles=req.sample_tiles,
            max_crops_per_tile=req.max_crops_per_tile,
            concurrency=req.concurrency,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - t0, 2)

    # Construir summary por especie
    species_map: dict = defaultdict(lambda: {
        "count": 0, "conf_sum": 0.0,
        "saludable": 0, "estresado": 0, "enfermo": 0
    })
    classified_count = 0
    for d in enriched:
        sp = d.get("species", "No clasificado")
        if sp not in ("No clasificado", "Desconocida"):
            classified_count += 1
        species_map[sp]["count"] += 1
        species_map[sp]["conf_sum"] += d.get("species_confidence", 0.0)
        h = d.get("health", "desconocido")
        if h == "saludable":
            species_map[sp]["saludable"] += 1
        elif h == "estresado":
            species_map[sp]["estresado"] += 1
        elif h == "enfermo":
            species_map[sp]["enfermo"] += 1

    total = len(enriched)
    summary = [
        SpeciesSummary(
            species=sp,
            count=v["count"],
            pct=round(v["count"] / total * 100, 1) if total else 0,
            avg_confidence=round(v["conf_sum"] / v["count"], 2) if v["count"] else 0,
            health_saludable=v["saludable"],
            health_estresado=v["estresado"],
            health_enfermo=v["enfermo"],
        )
        for sp, v in sorted(species_map.items(), key=lambda x: -x[1]["count"])
    ]

    n_clusters = len(set(d.get("cluster_id", -1) for d in enriched)) - (
        1 if any(d.get("cluster_id") == -1 for d in enriched) else 0
    )

    det_list = [ClassifiedDetection(**d) for d in enriched]

    return ClassifyResponse(
        job_id=req.job_id,
        total_trees=total,
        classified_trees=classified_count,
        n_clusters=n_clusters,
        tiles_sampled=req.sample_tiles,
        elapsed_sec=elapsed,
        species_summary=summary,
        detecciones=det_list,
    )

