"""
Trainer — fine-tuning de cualquier versión YOLO sobre el dataset compartido.
"""
import os
import csv
from ultralytics import YOLO


DEFAULT_TRAIN_ARGS = {
    "epochs":   80,
    "imgsz":    640,
    "batch":    8,
    "device":   "cpu",
    "patience": 20,
    "lr0":      0.005,
    "flipud":   0.5,
    "fliplr":   0.5,
    "mosaic":   0.8,
    "degrees":  15,
    "translate":0.1,
    "scale":    0.3,
    "verbose":  False,
    "exist_ok": True,
}


def train_model(
    model_key: str,
    base_weights: str,
    data_yaml: str,
    output_dir: str,
    train_args: dict | None = None,
) -> dict:
    """
    Fine-tunea el modelo y devuelve métricas del mejor epoch.

    Returns:
        {model_key, best_epoch, mAP50, mAP50_95, precision, recall,
         weights_path, epochs_completed}
    """
    args = {**DEFAULT_TRAIN_ARGS, **(train_args or {})}
    args["data"]    = data_yaml
    args["project"] = output_dir
    args["name"]    = model_key

    print(f"[trainer] Fine-tuning {model_key} ({base_weights}) → {output_dir}/{model_key}")
    model   = YOLO(base_weights)
    results = model.train(**args)

    # Leer métricas del CSV de resultados
    csv_path = os.path.join(str(results.save_dir), "results.csv")
    metrics  = _read_best_metrics(csv_path, model_key)
    metrics["weights_path"] = str(results.save_dir) + "/weights/best.pt"
    return metrics


def _read_best_metrics(csv_path: str, model_key: str) -> dict:
    """Lee el CSV de entrenamiento y devuelve métricas del mejor epoch."""
    if not os.path.exists(csv_path):
        return {"model_key": model_key, "mAP50": 0, "mAP50_95": 0,
                "precision": 0, "recall": 0, "best_epoch": 0, "epochs_completed": 0}

    rows = list(csv.DictReader(open(csv_path)))
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))
    return {
        "model_key":         model_key,
        "best_epoch":        int(float(best.get("epoch", 0))),
        "mAP50":             round(float(best.get("metrics/mAP50(B)", 0) or 0), 4),
        "mAP50_95":          round(float(best.get("metrics/mAP50-95(B)", 0) or 0), 4),
        "precision":         round(float(best.get("metrics/precision(B)", 0) or 0), 4),
        "recall":            round(float(best.get("metrics/recall(B)", 0) or 0), 4),
        "epochs_completed":  len(rows),
    }
