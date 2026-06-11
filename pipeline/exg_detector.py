"""
ExG Detector — detección de copas individuales por Excess Green + Watershed.

Flujo:
  1. Máscara ExG (índice de vegetación)
  2. Morphological open/close para limpiar ruido
  3. Distance transform + Watershed para separar copas pegadas
  4. Bounding box por cada región Watershed que pase filtros de tamaño/forma

ExG = 2*g/(r+g+b) - r/(r+g+b) - b/(r+g+b)
Umbral 0.12 sobre RGB uint8.
"""
import cv2
import numpy as np
import os
import time


def exg_mask(rgb: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    tot = r + g + b + 1e-9
    exg = 2 * (g / tot) - (r / tot) - (b / tot)
    mask = (exg > threshold).astype(np.uint8) * 255
    # Limpieza morfológica
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def watershed_crowns(
    mask: np.ndarray,
    min_area: int = 400,
    max_area: int = 15000,
    min_side: int = 18,
    max_side: int = 200,
) -> list[tuple]:
    """
    Aplica Distance Transform + Watershed sobre la máscara ExG.
    Devuelve lista de (x1, y1, x2, y2).
    """
    # Distance transform — picos = centros de copas
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # Sure foreground: picos del distance transform (umbral adaptativo)
    _, sure_fg = cv2.threshold(dist, 0.35, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    # Sure background: dilatación de la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)

    # Zona desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores para watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Necesita imagen BGR para watershed
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)

    boxes = []
    n_labels = markers.max()
    for label in range(2, n_labels + 1):  # 1 = background, -1 = bordes
        region = (markers == label).astype(np.uint8)
        coords = cv2.findNonZero(region)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue
        # Filtro aspecto — copas son ~circulares
        ratio = w / max(h, 1)
        if ratio > 2.8 or ratio < 0.36:
            continue
        boxes.append((x, y, x + w, y + h))

    return boxes


def run_exg_inference(
    tiles: list[dict],
    tiles_dir: str,
    exg_threshold: float = 0.12,
    min_area: int = 400,
    max_area: int = 15000,
) -> tuple[list[dict], float]:
    """
    Corre detección ExG+Watershed sobre todos los tiles.
    Mismo formato de salida que run_inference() del detector YOLO.
    """
    all_detections = []
    t0 = time.time()

    for tile in tiles:
        img_path = os.path.join(tiles_dir, tile["filename"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = exg_mask(rgb, threshold=exg_threshold)
        boxes = watershed_crowns(mask, min_area=min_area, max_area=max_area)

        for (x1, y1, x2, y2) in boxes:
            gx1 = int(tile["x0"]) + x1
            gy1 = int(tile["y0"]) + y1
            gx2 = int(tile["x0"]) + x2
            gy2 = int(tile["y0"]) + y2

            all_detections.append({
                "tile_filename": tile["filename"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "global_x1": gx1, "global_y1": gy1,
                "global_x2": gx2, "global_y2": gy2,
                "confidence": 0.90,
            })

    elapsed = time.time() - t0
    return all_detections, elapsed
