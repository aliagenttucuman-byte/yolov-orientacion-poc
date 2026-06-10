#!/usr/bin/env python3
"""
Construye el dataset compartido para todos los spikes.
Extrae tiles de los TIFs disponibles, aplica pseudo-labels ExG,
augmenta y divide 80/20 train/val.

Uso:
  python dataset/build_dataset.py
  python dataset/build_dataset.py --tifs /ruta/a.tif /ruta/b.tif
"""
import argparse
import os
import random
import shutil

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

# ── Configuración ──────────────────────────────────────────────────────────
TILE_SIZE   = 640      # imgsz estándar para todos los modelos
STRIDE      = 480      # overlap de 160px
MIN_BOXES   = 5        # tiles sin árboles son descartados
AUG_FLIPS   = True
RANDOM_SEED = 42

# TIFs por defecto (los que ya tenemos)
DEFAULT_TIFS = [
    "/home/server/.hermes/document_cache/doc_7a6ea9f8381b_9deJulio.rgb.tif",
    "/home/server/.hermes/document_cache/doc_877ea5356955_Avellaneda.rgb.tif",
    "/home/server/proyectos/forestai-poc/uploads/055422ad-92da-4ac9-8a96-abea4b7629b2.tif",
    "/home/server/proyectos/forestai-poc/uploads/f96fd29a-433d-4d78-9714-24ac1a643225.tif",
]

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.dirname(os.path.abspath(__file__))


def exg_boxes(rgb, min_area=120, max_area=30000):
    r = rgb[:,:,0].astype(float)
    g = rgb[:,:,1].astype(float)
    b = rgb[:,:,2].astype(float)
    H, W = r.shape
    tot = r + g + b + 1e-9
    exg = 2*(g/tot) - (r/tot) - (b/tot)
    mask = (exg > 0.12).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((
            (x + w/2) / W,
            (y + h/2) / H,
            w / W,
            h / H,
        ))
    return boxes


def save_pair(bgr, boxes, img_out, lbl_out):
    cv2.imwrite(img_out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(lbl_out, "w") as f:
        for cx, cy, w, h in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def flip_boxes(boxes, code):
    out = []
    for cx, cy, w, h in boxes:
        if code == 1:   cx = 1.0 - cx
        elif code == 0: cy = 1.0 - cy
        elif code == -1: cx, cy = 1.0-cx, 1.0-cy
        out.append((cx, cy, w, h))
    return out


def main(tif_paths):
    for split in ["train", "val"]:
        os.makedirs(f"{DATASET}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET}/labels/{split}", exist_ok=True)

    all_pairs = []
    random.seed(RANDOM_SEED)

    for tif_path in tif_paths:
        if not os.path.exists(tif_path):
            print(f"  [skip] no existe: {tif_path}")
            continue

        tif_name = os.path.splitext(os.path.basename(tif_path))[0][:20]
        with rasterio.open(tif_path) as src:
            W, H = src.width, src.height
            data = src.read()

        rgb = np.stack([data[0], data[1], data[2]], axis=-1).astype(np.uint8)
        tile_count = 0

        for row in range(0, H - TILE_SIZE, STRIDE):
            for col in range(0, W - TILE_SIZE, STRIDE):
                tile = rgb[row:row+TILE_SIZE, col:col+TILE_SIZE]
                boxes = exg_boxes(tile)
                if len(boxes) < MIN_BOXES:
                    continue
                name = f"{tif_name}_r{row//100:03d}_c{col//100:03d}"
                all_pairs.append((tile, boxes, name))
                tile_count += 1
                if tile_count >= 8:
                    break
            if tile_count >= 8:
                break

        print(f"  {tif_name}: {tile_count} tiles extraídos")

    print(f"\nTotal tiles base: {len(all_pairs)}")

    random.shuffle(all_pairs)
    n_train = max(1, int(len(all_pairs) * 0.8))
    train_p = all_pairs[:n_train]
    val_p   = all_pairs[n_train:]

    # Train: con augmentación
    total_train = 0
    for rgb, boxes, name in train_p:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        variants = [
            (bgr,             boxes),
            (cv2.flip(bgr,1), flip_boxes(boxes, 1)),
            (cv2.flip(bgr,0), flip_boxes(boxes, 0)),
            (cv2.flip(bgr,-1),flip_boxes(boxes,-1)),
        ]
        for vi, (v_bgr, v_boxes) in enumerate(variants):
            tag = ["orig","flipH","flipV","flipHV"][vi]
            save_pair(v_bgr, v_boxes,
                      f"{DATASET}/images/train/{name}_{tag}.jpg",
                      f"{DATASET}/labels/train/{name}_{tag}.txt")
            total_train += 1

    # Val: solo original
    for rgb, boxes, name in val_p:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        save_pair(bgr, boxes,
                  f"{DATASET}/images/val/{name}.jpg",
                  f"{DATASET}/labels/val/{name}.txt")

    print(f"Train: {total_train} imágenes  |  Val: {len(val_p)} imágenes")
    print(f"Dataset listo en: {DATASET}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tifs", nargs="+", default=DEFAULT_TIFS)
    args = p.parse_args()
    main(args.tifs)
