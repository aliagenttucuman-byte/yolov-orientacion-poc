"""
species_classifier_prod.py — Clasificador de especies para producción.

Flujo:
  1. yolo26n_especies clasifica la copa (YOLO fine-tuned, rápido)
  2. Si conf < threshold → gpt-4o-mini refina (fallback VLM)

Uso desde yolo_service.py:
  from pipeline.species_classifier_prod import classify_detections
  detections_with_species = classify_detections(detections, tiles_dir)
"""

import os
import sys
import base64
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI
from ultralytics import YOLO

log = logging.getLogger(__name__)

# Ruta al modelo fine-tuned
SPECIES_MODEL_PATH = os.environ.get(
    "SPECIES_MODEL_PATH",
    "/app/models/yolo26n_especies_noa_v1.pt"
)

# Especies del NOA (mismo orden que el entrenamiento)
ESPECIES_NOA = [
    "Quebracho blanco", "Quebracho colorado", "Algarrobo negro", "Algarrobo blanco",
    "Cebil colorado", "Tipa blanca", "Lapacho rosado", "Palo borracho",
    "Cedro tucumano", "Horco quebracho", "Guarán", "Vinal", "Otro",
]

# Umbral de confianza: por debajo activa el fallback VLM
CONF_FALLBACK_THRESHOLD = 0.50

_species_model: Optional[YOLO] = None


def _get_species_model() -> YOLO:
    global _species_model
    if _species_model is None:
        if not os.path.exists(SPECIES_MODEL_PATH):
            raise FileNotFoundError(f"Modelo de especies no encontrado: {SPECIES_MODEL_PATH}")
        log.info(f"[species_prod] Cargando modelo: {SPECIES_MODEL_PATH}")
        _species_model = YOLO(SPECIES_MODEL_PATH)
    return _species_model


def _crop_bbox(img: np.ndarray, det: dict, padding: int = 10) -> np.ndarray:
    """Recorta una copa del tile con padding."""
    h, w = img.shape[:2]
    x1 = max(0, det["x1"] - padding)
    y1 = max(0, det["y1"] - padding)
    x2 = min(w, det["x2"] + padding)
    y2 = min(h, det["y2"] + padding)
    return img[y1:y2, x1:x2]


def _classify_with_yolo(model: YOLO, crop: np.ndarray) -> tuple[str, float]:
    """Clasifica especie con el modelo YOLO fine-tuned."""
    try:
        results = model(crop, verbose=False, conf=0.01)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return "Otro", 0.0
        # Tomar la detección con mayor confianza
        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        conf = float(best.conf[0])
        cls_id = int(best.cls[0])
        especie = ESPECIES_NOA[cls_id] if cls_id < len(ESPECIES_NOA) else "Otro"
        return especie, conf
    except Exception as e:
        log.warning(f"[species_prod] YOLO classify error: {e}")
        return "Otro", 0.0


def _classify_with_vlm(client: OpenAI, crop: np.ndarray) -> tuple[str, float]:
    """Fallback: clasifica especie con gpt-4o-mini cuando YOLO tiene baja confianza."""
    try:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf.tobytes()).decode()

        especies_str = "\n".join(f"- {e}" for e in ESPECIES_NOA)
        prompt = f"""Sos un botánico experto en árboles nativos del NOA argentino (Tucumán).
Esta es una imagen aérea de drone de una copa de árbol vista desde arriba, zona urbana/periurbana.

Especies posibles:
{especies_str}

Respondé SOLO con JSON válido:
{{"especie": "<nombre exacto>", "confianza": <0.0-1.0>}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }}
            ]}],
            max_tokens=80,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(text)
        especie = data.get("especie", "Otro")
        if especie not in ESPECIES_NOA:
            especie = "Otro"
        return especie, float(data.get("confianza", 0.3))
    except Exception as e:
        log.warning(f"[species_prod] VLM fallback error: {e}")
        return "Otro", 0.0


def classify_detections(
    detections: list[dict],
    tiles_dir: str,
    conf_threshold: float = CONF_FALLBACK_THRESHOLD,
    openai_api_key: Optional[str] = None,
) -> list[dict]:
    """
    Clasifica la especie de cada detección.

    Para cada copa:
      - yolo26n_especies clasifica (rápido, sin costo)
      - Si conf < conf_threshold → gpt-4o-mini refina (fallback)

    Agrega a cada detección: especie, conf_especie, via ("yolo" | "vlm_fallback")
    """
    if not detections:
        return detections

    model = _get_species_model()
    oai_client = None
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    # Cache de tiles ya leídos
    tile_cache: dict[str, np.ndarray] = {}

    yolo_count = 0
    vlm_count = 0

    for det in detections:
        tile_fname = det.get("tile_filename", "")
        tile_path = os.path.join(tiles_dir, tile_fname)

        if tile_fname not in tile_cache:
            img = cv2.imread(tile_path)
            tile_cache[tile_fname] = img
        img = tile_cache.get(tile_fname)

        if img is None:
            det["especie"] = "Otro"
            det["conf_especie"] = 0.0
            det["via"] = "error"
            continue

        crop = _crop_bbox(img, det)
        if crop.size == 0:
            det["especie"] = "Otro"
            det["conf_especie"] = 0.0
            det["via"] = "error"
            continue

        # Paso 1 — YOLO fine-tuned
        especie, conf = _classify_with_yolo(model, crop)

        # Paso 2 — Fallback VLM si confianza baja
        if conf < conf_threshold and api_key:
            if oai_client is None:
                oai_client = OpenAI(api_key=api_key)
            especie_vlm, conf_vlm = _classify_with_vlm(oai_client, crop)
            # Usar VLM solo si da más confianza
            if conf_vlm > conf:
                especie, conf = especie_vlm, conf_vlm
                det["via"] = "vlm_fallback"
                vlm_count += 1
            else:
                det["via"] = "yolo"
                yolo_count += 1
        else:
            det["via"] = "yolo"
            yolo_count += 1

        det["especie"] = especie
        det["conf_especie"] = round(conf, 3)

    log.info(f"[species_prod] {len(detections)} copas — YOLO: {yolo_count} | VLM fallback: {vlm_count}")
    return detections
