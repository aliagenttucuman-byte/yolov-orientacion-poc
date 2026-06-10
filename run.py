#!/usr/bin/env python3
"""
YOLOv Orientación PoC — runner principal.

Uso:
  # Comparar todas las versiones YOLO (fine-tune + inferencia)
  python run.py --mode compare --tif /ruta/a/ortofoto.tif

  # Solo inferencia con un modelo ya entrenado
  python run.py --mode infer --tif /ruta/a/ortofoto.tif --model yolo11n

  # Solo fine-tuning (sin inferencia)
  python run.py --mode train --models yolov8n yolo11n yolov9c

  # Ver resultados de una corrida anterior
  python run.py --mode report
"""
import argparse
import os
import sys
import json
import time

from pipeline.tiler    import generate_tiles, TILE_SIZE, TILE_OVERLAP
from pipeline.detector import load_model, run_inference, nms_global, SUPPORTED_MODELS
from pipeline.trainer  import train_model, DEFAULT_TRAIN_ARGS
from pipeline.reporter import (
    save_metrics_csv, save_metrics_json,
    plot_comparison, annotate_tile,
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, "dataset")
DATA_YAML   = os.path.join(DATASET_DIR, "data.yaml")
RUNS_DIR    = os.path.join(ROOT, "runs")
TILES_DIR   = os.path.join(ROOT, "tiles_cache")        # tiles generados desde el TIF
RESULTS_DIR = os.path.join(ROOT, "resultados")

# Modelos a comparar por defecto
DEFAULT_MODELS = ["yolov8n", "yolo11n", "yolov9c"]


def cmd_train(args):
    """Fine-tune cada modelo sobre el dataset compartido."""
    models  = args.models or DEFAULT_MODELS
    metrics = []

    for model_key in models:
        from pipeline.detector import SUPPORTED_MODELS
        base_weights = SUPPORTED_MODELS.get(model_key)
        if not base_weights:
            print(f"[run] SKIP — modelo desconocido: {model_key}")
            continue

        result = train_model(
            model_key    = model_key,
            base_weights = base_weights,
            data_yaml    = DATA_YAML,
            output_dir   = RUNS_DIR,
            train_args   = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch},
        )
        metrics.append(result)
        print(f"  [{model_key}] mAP50={result['mAP50']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}")

    _save_and_plot(metrics)
    return metrics


def cmd_infer(args):
    """Inferencia de un modelo (pre-trained o fine-tuned) sobre un TIF."""
    if not args.tif or not os.path.exists(args.tif):
        print("[run] ERROR: --tif requerido y debe existir")
        sys.exit(1)

    # Generar tiles del TIF
    tif_name   = os.path.splitext(os.path.basename(args.tif))[0]
    tiles_dir  = os.path.join(TILES_DIR, tif_name)
    tiles_meta = generate_tiles(args.tif, tiles_dir,
                                tile_size=args.tile_size, overlap=args.overlap)

    # Determinar weights
    model_key    = args.model or "yolo11n"
    weights_path = _find_weights(model_key, args.weights)
    model        = load_model(model_key, weights_path)

    # Inferencia
    detections, elapsed = run_inference(model, tiles_meta, tiles_dir,
                                        conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    detections_nms = nms_global(detections)

    print(f"[run] {model_key} — {len(detections)} detecciones → {len(detections_nms)} tras NMS ({elapsed:.1f}s)")
    print(f"[run] Velocidad: {len(tiles_meta)/elapsed:.1f} tiles/seg")

    # Guardar detecciones
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, f"detecciones_{model_key}_{tif_name}.json")
    with open(out_json, "w") as f:
        json.dump({
            "model": model_key, "tif": args.tif,
            "tiles": len(tiles_meta), "detecciones_raw": len(detections),
            "detecciones_nms": len(detections_nms),
            "tiempo_seg": round(elapsed, 2),
            "tiles_por_seg": round(len(tiles_meta)/elapsed, 2),
            "trees": detections_nms,
        }, f, indent=2)
    print(f"[run] Resultados: {out_json}")

    # Anotar muestra de tiles (primeros 5 con detecciones)
    _annotate_samples(detections_nms, tiles_dir, model_key)
    return detections_nms


def cmd_compare(args):
    """Train + Infer todos los modelos y comparar."""
    print("=" * 60)
    print("YOLOv COMPARATIVA — ForestAI Pipeline")
    print("=" * 60)

    models  = args.models or DEFAULT_MODELS
    all_metrics = []

    for model_key in models:
        from pipeline.detector import SUPPORTED_MODELS
        base_weights = SUPPORTED_MODELS.get(model_key)
        if not base_weights:
            print(f"[compare] SKIP — {model_key} no soportado")
            continue

        print(f"\n─── {model_key} ───")
        # 1. Fine-tune
        result = train_model(
            model_key    = model_key,
            base_weights = base_weights,
            data_yaml    = DATA_YAML,
            output_dir   = RUNS_DIR,
            train_args   = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch},
        )

        # 2. Inferencia sobre TIF si se proveyó
        if args.tif and os.path.exists(args.tif):
            tif_name   = os.path.splitext(os.path.basename(args.tif))[0]
            tiles_dir  = os.path.join(TILES_DIR, tif_name)
            tiles_meta = generate_tiles(args.tif, tiles_dir, tile_size=args.tile_size)
            model      = load_model(model_key, result["weights_path"])
            dets, elapsed = run_inference(model, tiles_meta, tiles_dir, imgsz=args.imgsz)
            dets_nms   = nms_global(dets)
            result["detecciones"] = len(dets_nms)
            result["tiles_por_seg"] = round(len(tiles_meta)/elapsed, 2)

        all_metrics.append(result)
        print(f"  mAP50={result['mAP50']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}", end="")
        if "detecciones" in result:
            print(f"  dets={result['detecciones']}  tiles/s={result['tiles_por_seg']}", end="")
        print()

    _save_and_plot(all_metrics)
    _print_leaderboard(all_metrics)


def cmd_report(args):
    """Muestra los resultados de la última corrida."""
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[report] No hay resultados. Corré primero: python run.py --mode compare")
        return
    with open(metrics_path) as f:
        metrics = json.load(f)
    _print_leaderboard(metrics)


# ── helpers ────────────────────────────────────────────────────────────────
def _find_weights(model_key: str, weights_arg: str | None) -> str | None:
    if weights_arg and os.path.exists(weights_arg):
        return weights_arg
    candidate = os.path.join(RUNS_DIR, model_key, "weights", "best.pt")
    if os.path.exists(candidate):
        print(f"[run] Usando weights fine-tuned: {candidate}")
        return candidate
    return None


def _save_and_plot(metrics: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_metrics_csv(metrics, os.path.join(RESULTS_DIR, "metrics.csv"))
    save_metrics_json(metrics, os.path.join(RESULTS_DIR, "metrics.json"))
    plot_comparison(metrics, os.path.join(RESULTS_DIR, "comparativa_yolov.png"))


def _annotate_samples(detections: list[dict], tiles_dir: str, model_key: str, n: int = 5):
    """Anota los primeros N tiles que tienen detecciones."""
    seen_tiles = {}
    for d in detections:
        t = d["tile_filename"]
        seen_tiles.setdefault(t, []).append(d)
        if len(seen_tiles) >= n:
            break
    COLORS = {"yolov8n":(0,212,170), "yolov8s":(0,153,255),
               "yolo11n":(255,107,107), "yolov9c":(255,200,0)}
    color = COLORS.get(model_key, (200,200,200))
    for tile_file, dets in seen_tiles.items():
        src = os.path.join(tiles_dir, tile_file)
        dst = os.path.join(RESULTS_DIR, "samples", f"{model_key}_{tile_file.replace('.jpg','')}.jpg")
        annotate_tile(src, dets, dst, model_label=model_key, color=color)


def _print_leaderboard(metrics: list[dict]):
    print("\n" + "=" * 60)
    print("LEADERBOARD — YOLOv vs ForestAI Trees")
    print("=" * 60)
    sorted_m = sorted(metrics, key=lambda m: m.get("mAP50", 0), reverse=True)
    print(f"{'Modelo':12} {'mAP50':>7} {'P':>7} {'R':>7} {'Epochs':>7} {'Dets':>6}")
    print("-" * 60)
    for m in sorted_m:
        dets = m.get("detecciones", "-")
        print(f"{m['model_key']:12} {m['mAP50']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['epochs_completed']:7} {str(dets):>6}")
    winner = sorted_m[0]["model_key"] if sorted_m else "N/A"
    print(f"\nGanador: {winner}")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLOv Orientación PoC")
    p.add_argument("--mode",      choices=["train","infer","compare","report"], default="compare")
    p.add_argument("--tif",       help="Ruta al GeoTIFF de ortofoto")
    p.add_argument("--model",     help="Modelo para --mode infer (ej: yolo11n)")
    p.add_argument("--models",    nargs="+", help="Modelos para train/compare")
    p.add_argument("--weights",   help="Ruta a best.pt ya entrenado")
    p.add_argument("--epochs",    type=int, default=80)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--tile-size", type=int, default=1024, dest="tile_size")
    p.add_argument("--overlap",   type=int, default=128)
    args = p.parse_args()

    if   args.mode == "train":   cmd_train(args)
    elif args.mode == "infer":   cmd_infer(args)
    elif args.mode == "compare": cmd_compare(args)
    elif args.mode == "report":  cmd_report(args)
