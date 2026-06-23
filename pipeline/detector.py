"""
Detector — corre un modelo YOLO sobre los tiles y devuelve detecciones globales.
Soporta cualquier versión de ultralytics: yolov8n, yolo11n, yolov9c, etc.
"""
import os
import time
import numpy as np
from ultralytics import YOLO


SUPPORTED_MODELS = {
    "yolov8n":         "yolov8n.pt",
    "yolov8s":         "yolov8s.pt",
    "yolov9c":         "yolov9c.pt",
    "yolo11n":         "yolo11n.pt",
    "yolo11s":         "yolo11s.pt",
    "yolo11n_forestai": "/app/models/yolo11n_forestai_v2.pt",   # fine-tuned ortofotos NOA
    "yolo26n":         "yolo26n.pt",   # v8.4.0 — nuevo modelo base Ultralytics
    "yolo26n_especies": "/app/models/yolo26n_especies_noa_v1.pt",  # fine-tuned NOA: Tipa blanca, Lapacho rosado
}


def load_model(model_key: str, weights_path: str | None = None) -> YOLO:
    """
    Carga modelo YOLO.
    - Si se pasa weights_path (ej. best.pt fine-tuned), lo usa directamente.
    - Si el valor en SUPPORTED_MODELS es un path absoluto, lo usa directamente.
    - Si no, descarga el pretrained indicado por model_key.
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[detector] Cargando weights: {weights_path}")
        return YOLO(weights_path)

    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo '{model_key}' no soportado. Opciones: {list(SUPPORTED_MODELS)}")

    pt_file = SUPPORTED_MODELS[model_key]

    # Si es path absoluto (fine-tuned), usar directo
    if pt_file.startswith("/"):
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Weights fine-tuned no encontrados: {pt_file}")
        print(f"[detector] Cargando fine-tuned: {pt_file}")
        return YOLO(pt_file)

    print(f"[detector] Cargando modelo base: {pt_file}")
    return YOLO(pt_file)


def run_inference(
    model: YOLO,
    tiles: list[dict],
    tiles_dir: str,
    conf: float = 0.25,
    iou:  float = 0.45,
    imgsz: int = 640,
    min_bbox_px: int = 25,
    max_aspect_ratio: float = 2.5,
) -> tuple[list[dict], float]:
    """
    Corre inferencia sobre todos los tiles.
    Devuelve (detecciones_globales, tiempo_total_seg).
    Cada detección: {x1,y1,x2,y2,conf,tile_filename,global_x1,...}

    min_bbox_px: descarta bboxes menores a N px en ancho o alto (ruido, autos)
    max_aspect_ratio: descarta bboxes muy alargados (caminos, paredes) — copas son ~circulares
    """
    all_detections = []
    t0 = time.time()

    for tile in tiles:
        img_path = os.path.join(tiles_dir, tile["filename"])
        results  = model(img_path, verbose=False, conf=conf, iou=iou, imgsz=imgsz)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            w = x2 - x1
            h = y2 - y1

            # Filtro tamaño mínimo — autos/cruces/ruido son pequeños
            if w < min_bbox_px or h < min_bbox_px:
                continue

            # Filtro aspect ratio — copas son aproximadamente circulares
            ratio = w / max(h, 1)
            if ratio > max_aspect_ratio or ratio < (1 / max_aspect_ratio):
                continue

            score = float(box.conf[0])

            # Trasladar al espacio global de la ortofoto
            gx1 = int(tile["x0"]) + x1
            gy1 = int(tile["y0"]) + y1
            gx2 = int(tile["x0"]) + x2
            gy2 = int(tile["y0"]) + y2

            all_detections.append({
                "tile_filename": tile["filename"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "global_x1": gx1, "global_y1": gy1,
                "global_x2": gx2, "global_y2": gy2,
                "confidence": round(score, 4),
            })

    elapsed = time.time() - t0
    return all_detections, elapsed


def nms_global(
    detections: list[dict],
    iou_threshold: float = 0.40,
    centroid_dist_px: int = 40,
) -> list[dict]:
    """
    NMS sobre coordenadas globales para eliminar duplicados de tiles solapados.

    Dos criterios de supresión (OR):
      1. IoU > iou_threshold  → mismo árbol detectado en tiles solapados
      2. distancia entre centroides < centroid_dist_px → misma copa grande
         detectada múltiples veces con bboxes levemente desplazados

    centroid_dist_px=40 equivale a ~2.4m a 6cm/px → copa única fusionada.
    Subir a 60-80 para ortofotos con copas muy grandes (eucaliptos, quebrachos adultos).
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    suppressed = set()

    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(d)
        ax1, ay1, ax2, ay2 = d["global_x1"], d["global_y1"], d["global_x2"], d["global_y2"]
        area_a  = max(0, ax2-ax1) * max(0, ay2-ay1)
        acx, acy = (ax1+ax2) / 2, (ay1+ay2) / 2   # centroide A

        for j in range(i+1, len(dets)):
            if j in suppressed:
                continue
            b = dets[j]
            bx1, by1, bx2, by2 = b["global_x1"], b["global_y1"], b["global_x2"], b["global_y2"]

            # Criterio 1 — IoU
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter  = max(0, ix2-ix1) * max(0, iy2-iy1)
            area_b = max(0, bx2-bx1) * max(0, by2-by1)
            union  = area_a + area_b - inter
            iou_val = inter / union if union > 0 else 0
            if iou_val > iou_threshold:
                suppressed.add(j)
                continue

            # Criterio 2 — distancia entre centroides
            bcx, bcy = (bx1+bx2) / 2, (by1+by2) / 2
            dist = ((acx - bcx)**2 + (acy - bcy)**2) ** 0.5
            if dist < centroid_dist_px:
                suppressed.add(j)

    return keep
