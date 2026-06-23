"""
species_labeler.py — Pipeline de auto-etiquetado de especies del NOA

Flujo:
1. Carga ortofoto GeoTIFF
2. Genera tiles con tiler.py
3. Detecta copas con yolo26n
4. Recorta cada copa detectada
5. Clasifica especie con gpt-4o-mini (visión)
6. Guarda anotaciones YOLO multi-clase en disco

Salida:
  output_dir/
    images/   → tiles .jpg
    labels/   → anotaciones YOLO formato {class_id x_center y_center w h}
    crops/    → recortes de copa clasificados (para revisión)
    report.json → resumen de especies detectadas
"""

import os
import sys
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI
import anthropic

# --- path setup para importar pipeline desde cualquier CWD ---
def _find_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pipeline").is_dir():
            return p
    return Path(__file__).resolve().parent

_root = str(_find_root())
if _root not in sys.path:
    sys.path.insert(0, _root)

from pipeline.tiler import generate_tiles
from pipeline.detector import load_model, run_inference, nms_global

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Especies del NOA/Tucumán ─────────────────────────────────────────────────
ESPECIES_NOA = [
    "Quebracho blanco",
    "Quebracho colorado",
    "Algarrobo negro",
    "Algarrobo blanco",
    "Cebil colorado",
    "Tipa blanca",
    "Lapacho rosado",
    "Palo borracho",
    "Cedro tucumano",
    "Horco quebracho",
    "Guarán",
    "Vinal",
    "Otro",  # clase catch-all
]

# Índice especie → class_id
ESPECIE_TO_ID = {esp: i for i, esp in enumerate(ESPECIES_NOA)}
ID_TO_ESPECIE = {i: esp for i, esp in enumerate(ESPECIES_NOA)}


# ── VLM: clasificación por copa ──────────────────────────────────────────────
def _encode_crop(img_bgr: np.ndarray) -> str:
    """Convierte crop BGR a base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def classify_crop(client, crop_bgr: np.ndarray, especies: list[str], backend: str = "claude") -> tuple[str, float]:
    """
    Clasifica la especie de una copa usando VLM.
    backend="claude" → Azure Anthropic Claude Sonnet (para generar dataset — mejor calidad)
    backend="openai" → gpt-4o-mini (para inferencia en producción — más barato)
    Retorna (especie, confianza).
    """
    b64 = _encode_crop(crop_bgr)
    especies_str = "\n".join(f"- {e}" for e in especies)

    prompt = f"""Sos un botánico especialista en árboles nativos del NOA (Noroeste Argentino) y Tucumán.
Analizá esta imagen aérea de drone de una copa de árbol vista desde arriba, zona urbana/periurbana de Tucumán.

Especies posibles:
{especies_str}

Respondé SOLO con un JSON válido:
{{"especie": "<nombre exacto de la lista>", "confianza": <0.0-1.0>, "razon": "<1 frase breve>"}}

Si no podés determinar la especie con seguridad, usá "Otro"."""

    try:
        if backend == "claude":
            ant_client = client if isinstance(client, anthropic.Anthropic) else None
            if ant_client is None:
                raise ValueError("Para backend=claude pasar instancia anthropic.Anthropic")
            resp = ant_client.messages.create(
                model=os.environ.get("AZURE_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=150,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            text = resp.content[0].text.strip()
        else:
            oai_client = client if isinstance(client, OpenAI) else None
            if oai_client is None:
                raise ValueError("Para backend=openai pasar instancia OpenAI")
            resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }}
                ]}],
                max_tokens=150,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()

        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        especie = data.get("especie", "Otro")
        if especie not in especies:
            especie = "Otro"
        confianza = float(data.get("confianza", 0.5))
        return especie, confianza
    except Exception as e:
        log.warning(f"VLM error ({backend}): {e}")
        return "Otro", 0.0


# ── Recorte de copa ──────────────────────────────────────────────────────────
def crop_tree(tile_img: np.ndarray, det: dict, padding: int = 10) -> np.ndarray:
    """Recorta la copa de un árbol del tile con padding."""
    h, w = tile_img.shape[:2]
    x1 = max(0, det["x1"] - padding)
    y1 = max(0, det["y1"] - padding)
    x2 = min(w, det["x2"] + padding)
    y2 = min(h, det["y2"] + padding)
    return tile_img[y1:y2, x1:x2]


# ── Pipeline principal ───────────────────────────────────────────────────────
def run_species_labeling(
    tif_path: str,
    output_dir: str,
    model_key: str = "yolo26n",
    conf: float = 0.15,
    iou: float = 0.45,
    tile_size: int = 640,
    overlap: int = 128,
    centroid_dist_px: int = 60,
    min_bbox_px: int = 25,
    max_aspect_ratio: float = 2.5,
    min_conf_vlm: float = 0.30,   # descartar clasificaciones muy inseguras
    max_crops_per_tile: int = 20,  # limitar costo VLM
    openai_api_key: Optional[str] = None,
    save_crops: bool = True,
) -> dict:
    """
    Pipeline completo: TIF → tiles → detección → clasificación VLM → dataset YOLO.

    Retorna dict con resumen de especies y rutas de salida.
    """
    t_start = time.time()
    out = Path(output_dir)
    imgs_dir   = out / "images"
    labels_dir = out / "labels"
    crops_dir  = out / "crops"
    for d in [imgs_dir, labels_dir, crops_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Tiling ──────────────────────────────────────────────────────────
    log.info(f"Generando tiles de {tif_path} ...")
    tiles_tmp = str(out / "tiles_tmp")
    os.makedirs(tiles_tmp, exist_ok=True)
    tiles_meta = generate_tiles(
        tif_path,
        output_dir=tiles_tmp,
        tile_size=tile_size,
        overlap=overlap,
    )
    log.info(f"  {len(tiles_meta)} tiles generados")

    # ── 2. Detección yolo26n ───────────────────────────────────────────────
    log.info(f"Cargando modelo {model_key} ...")
    model = load_model(model_key)
    log.info("Corriendo inferencia ...")
    dets_raw, elapsed_infer = run_inference(
        model, tiles_meta, tiles_tmp,
        conf=conf, iou=iou, imgsz=tile_size,
        min_bbox_px=min_bbox_px,
        max_aspect_ratio=max_aspect_ratio,
    )
    dets = nms_global(dets_raw, iou_threshold=iou, centroid_dist_px=centroid_dist_px)
    log.info(f"  {len(dets)} copas detectadas en {elapsed_infer:.1f}s")

    # ── 3. Clasificación VLM ───────────────────────────────────────────────
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no configurada")

    # Backend de etiquetado: Claude (dataset) o gpt-4o-mini (producción)
    vlm_backend = os.environ.get("LABELER_BACKEND", "claude")
    if vlm_backend == "claude":
        azure_key = os.environ.get("AZURE_ANTHROPIC_API_KEY")
        azure_url = os.environ.get("AZURE_ANTHROPIC_BASE_URL")
        if not azure_key or not azure_url:
            log.warning("Azure Anthropic no configurado, fallback a gpt-4o-mini")
            vlm_backend = "openai"
            client = OpenAI(api_key=api_key)
        else:
            client = anthropic.Anthropic(api_key=azure_key, base_url=azure_url)
            log.info("VLM backend: Claude Sonnet (Azure Anthropic)")
    else:
        client = OpenAI(api_key=api_key)
        log.info("VLM backend: gpt-4o-mini (OpenAI)")

    # Agrupar detecciones por tile
    tile_map: dict[str, list[dict]] = {}
    for d in dets:
        tile_map.setdefault(d["tile_filename"], []).append(d)

    species_counts: dict[str, int] = {e: 0 for e in ESPECIES_NOA}
    labeled_dets: list[dict] = []
    total_vlm_calls = 0

    log.info(f"Clasificando copas con gpt-4o-mini ({len(tile_map)} tiles) ...")

    for tile_fname, tile_dets in tile_map.items():
        tile_path = os.path.join(tiles_tmp, tile_fname)
        if not os.path.exists(tile_path):
            continue

        tile_img = cv2.imread(tile_path)
        if tile_img is None:
            continue

        # Limitar crops por tile para controlar costo
        tile_dets_sorted = sorted(tile_dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
        tile_dets_limited = tile_dets_sorted[:max_crops_per_tile]

        for det in tile_dets_limited:
            crop = crop_tree(tile_img, det)
            if crop.size == 0:
                continue

            especie, conf_vlm = classify_crop(client, crop, ESPECIES_NOA, backend=vlm_backend)
            total_vlm_calls += 1

            det["especie"] = especie
            det["conf_vlm"] = conf_vlm
            labeled_dets.append(det)

            if conf_vlm >= min_conf_vlm:
                species_counts[especie] = species_counts.get(especie, 0) + 1

            # Guardar crop clasificado
            if save_crops:
                crop_subdir = crops_dir / especie.replace(" ", "_")
                crop_subdir.mkdir(exist_ok=True)
                tile_stem = Path(tile_fname).stem
                crop_name = f"{tile_stem}_x{det['x1']}_y{det['y1']}.jpg"
                cv2.imwrite(str(crop_subdir / crop_name), crop)

        # Copiar tile como imagen de entrenamiento
        dst_img = imgs_dir / tile_fname
        if not dst_img.exists():
            import shutil
            shutil.copy2(tile_path, dst_img)

        # Generar anotación YOLO para este tile
        label_path = labels_dir / (Path(tile_fname).stem + ".txt")
        with open(label_path, "w") as f:
            th, tw = tile_img.shape[:2]
            for det in tile_dets_limited:
                especie = det.get("especie", "Otro")
                conf_vlm = det.get("conf_vlm", 0.0)
                if conf_vlm < min_conf_vlm:
                    continue  # descartar etiquetas de baja confianza
                class_id = ESPECIE_TO_ID.get(especie, ESPECIE_TO_ID["Otro"])
                x_c = (det["x1"] + det["x2"]) / 2 / tw
                y_c = (det["y1"] + det["y2"]) / 2 / th
                w   = (det["x2"] - det["x1"]) / tw
                h   = (det["y2"] - det["y1"]) / th
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        log.info(f"  tile {tile_fname}: {len(tile_dets_limited)} copas procesadas")

    elapsed_total = time.time() - t_start

    # ── 4. Generar data.yaml ───────────────────────────────────────────────
    data_yaml = out / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {str(out.resolve())}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"nc: {len(ESPECIES_NOA)}\n")
        names_str = "[" + ", ".join(f'"{e}"' for e in ESPECIES_NOA) + "]"
        f.write(f"names: {names_str}\n")

    # ── 5. Reporte ─────────────────────────────────────────────────────────
    report = {
        "tif": tif_path,
        "total_copas_detectadas": len(dets),
        "total_copas_clasificadas": len(labeled_dets),
        "total_vlm_calls": total_vlm_calls,
        "elapsed_sec": round(elapsed_total, 1),
        "species_counts": {k: v for k, v in species_counts.items() if v > 0},
        "output_dir": str(out.resolve()),
        "data_yaml": str(data_yaml),
        "model_key": model_key,
        "params": {
            "conf": conf, "iou": iou, "tile_size": tile_size,
            "overlap": overlap, "centroid_dist_px": centroid_dist_px,
            "min_conf_vlm": min_conf_vlm,
        }
    }

    report_path = out / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Listo en {elapsed_total:.1f}s — {len(labeled_dets)} copas etiquetadas")
    log.info(f"   Especies: {report['species_counts']}")
    log.info(f"   VLM calls: {total_vlm_calls}")
    log.info(f"   Dataset: {out}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-etiquetado de especies NOA con yolo26n + gpt-4o-mini")
    parser.add_argument("tif", help="Ruta al GeoTIFF de ortofoto")
    parser.add_argument("--output", default="/tmp/species_dataset", help="Directorio de salida")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--max-crops", type=int, default=20, help="Máx crops por tile (controla costo VLM)")
    parser.add_argument("--min-conf-vlm", type=float, default=0.30)
    args = parser.parse_args()

    result = run_species_labeling(
        tif_path=args.tif,
        output_dir=args.output,
        conf=args.conf,
        tile_size=args.tile_size,
        max_crops_per_tile=args.max_crops,
        min_conf_vlm=args.min_conf_vlm,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
