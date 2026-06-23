"""
trainer_species.py — Fine-tuning de yolo26n para clasificación de especies del NOA.

Toma el dataset generado por build_species_dataset.py y entrena yolo26n
con las 13 clases de especies nativas del NOA/Tucumán.

Uso:
  python pipeline/trainer_species.py
  python pipeline/trainer_species.py --data /tmp/species_dataset_final/data.yaml --epochs 80
"""

import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_species_model(
    data_yaml: str,
    base_model: str = "yolo26n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 20,
    project: str = "runs/species",
    name: str = "yolo26n_especies_noa",
    device: str = "cpu",
) -> dict:
    """
    Fine-tunea yolo26n con el dataset de especies del NOA.
    Retorna dict con rutas al modelo entrenado y métricas.
    """
    from ultralytics import YOLO

    log.info(f"Cargando modelo base: {base_model}")
    model = YOLO(base_model)

    log.info(f"Iniciando entrenamiento:")
    log.info(f"  data:    {data_yaml}")
    log.info(f"  epochs:  {epochs}")
    log.info(f"  imgsz:   {imgsz}")
    log.info(f"  batch:   {batch}")
    log.info(f"  device:  {device}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
        # Augmentaciones
        lr0=0.005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        degrees=15,
        translate=0.1,
        scale=0.3,
        # Guardar best.pt
        save=True,
        save_period=10,
    )

    # Ruta al mejor modelo
    best_pt = Path(project) / name / "weights" / "best.pt"
    last_pt = Path(project) / name / "weights" / "last.pt"

    # Validar modelo entrenado
    log.info("Evaluando modelo en val set ...")
    metrics = model.val()

    summary = {
        "best_model": str(best_pt.resolve()) if best_pt.exists() else None,
        "last_model": str(last_pt.resolve()) if last_pt.exists() else None,
        "map50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
        "map50_95": float(metrics.box.map) if hasattr(metrics, "box") else None,
        "epochs_ran": results.epoch if hasattr(results, "epoch") else epochs,
        "base_model": base_model,
        "data_yaml": data_yaml,
    }

    log.info(f"\n✅ Entrenamiento finalizado!")
    log.info(f"   mAP50:    {summary['map50']}")
    log.info(f"   mAP50-95: {summary['map50_95']}")
    log.info(f"   Best model: {summary['best_model']}")

    # Guardar resumen
    out_dir = Path(project) / name
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning yolo26n para especies NOA")
    parser.add_argument("--data", default="/tmp/species_dataset_final/data.yaml",
                        help="Path al data.yaml generado por build_species_dataset.py")
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="yolo26n_especies_noa")
    args = parser.parse_args()

    result = train_species_model(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
