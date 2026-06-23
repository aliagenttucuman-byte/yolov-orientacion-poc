"""
build_species_dataset.py — Construye dataset YOLO multi-clase para especies del NOA.

Puede combinar múltiples TIFs para más variedad de datos.
Hace split train/val (80/20) y genera el data.yaml final.

Uso:
  python dataset/build_species_dataset.py
  python dataset/build_species_dataset.py --tifs /ruta/a.tif /ruta/b.tif --output /tmp/species_ds
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path

# path setup
def _find_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pipeline").is_dir():
            return p
    return Path(__file__).resolve().parent

_root = str(_find_root())
if _root not in sys.path:
    sys.path.insert(0, _root)

from pipeline.species_labeler import run_species_labeling, ESPECIES_NOA

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# TIFs por defecto del equipo
DEFAULT_TIFS = [
    "/home/server/.hermes/document_cache/doc_877ea5356955_Avellaneda.rgb.tif",
    "/home/server/.hermes/document_cache/doc_7a6ea9f8381b_9deJulio.rgb.tif",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def build_dataset(
    tif_paths: list[str],
    output_dir: str = "/tmp/species_dataset_final",
    conf: float = 0.15,
    tile_size: int = 640,
    max_crops_per_tile: int = 15,
    min_conf_vlm: float = 0.30,
    openai_api_key: str | None = None,
) -> dict:
    out = Path(output_dir)
    train_imgs = out / "images" / "train"
    val_imgs   = out / "images" / "val"
    train_lbs  = out / "labels" / "train"
    val_lbs    = out / "labels" / "val"
    for d in [train_imgs, val_imgs, train_lbs, val_lbs]:
        d.mkdir(parents=True, exist_ok=True)

    all_reports = []

    # Procesar cada TIF con species_labeler
    for tif in tif_paths:
        if not os.path.exists(tif):
            log.warning(f"TIF no encontrado, skip: {tif}")
            continue

        tif_name = Path(tif).stem.replace(".", "_")
        tif_out = out / "raw" / tif_name

        log.info(f"\n{'='*50}")
        log.info(f"Procesando: {tif}")

        report = run_species_labeling(
            tif_path=tif,
            output_dir=str(tif_out),
            model_key="yolo26n",
            conf=conf,
            tile_size=tile_size,
            max_crops_per_tile=max_crops_per_tile,
            min_conf_vlm=min_conf_vlm,
            openai_api_key=openai_api_key,
            save_crops=True,
        )
        all_reports.append(report)

    # Split train/val sobre todas las imágenes combinadas
    all_images = list((out / "raw").rglob("images/*.jpg")) + \
                 list((out / "raw").rglob("images/*.png"))

    random.seed(RANDOM_SEED)
    random.shuffle(all_images)

    n_train = int(len(all_images) * TRAIN_RATIO)
    splits = {"train": all_images[:n_train], "val": all_images[n_train:]}

    counts = {"train": 0, "val": 0}
    for split, imgs in splits.items():
        dst_imgs = out / "images" / split
        dst_lbs  = out / "labels" / split
        for img_path in imgs:
            # copiar imagen
            shutil.copy2(img_path, dst_imgs / img_path.name)
            # copiar label correspondiente
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, dst_lbs / label_path.name)
            counts[split] += 1

    # Combinar species_counts de todos los TIFs
    total_species: dict[str, int] = {}
    for r in all_reports:
        for esp, cnt in r.get("species_counts", {}).items():
            total_species[esp] = total_species.get(esp, 0) + cnt

    # Filtrar clases con al menos 5 ejemplos para no entrenar con 1-2 muestras
    valid_classes = [e for e in ESPECIES_NOA if total_species.get(e, 0) >= 5]
    if not valid_classes:
        log.warning("Pocas muestras por clase — usando todas las clases igualmente")
        valid_classes = ESPECIES_NOA

    log.info(f"\nClases con >=5 ejemplos: {valid_classes}")

    # data.yaml final
    data_yaml = out / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {str(out.resolve())}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {len(ESPECIES_NOA)}\n")
        names_str = "[" + ", ".join(f'"{e}"' for e in ESPECIES_NOA) + "]"
        f.write(f"names: {names_str}\n")

    summary = {
        "output_dir": str(out.resolve()),
        "data_yaml": str(data_yaml),
        "tifs_procesados": len(all_reports),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "total_images": counts["train"] + counts["val"],
        "species_counts": total_species,
        "valid_classes": valid_classes,
        "nc": len(ESPECIES_NOA),
    }

    with open(out / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ Dataset listo!")
    log.info(f"   Train: {counts['train']} imgs | Val: {counts['val']} imgs")
    log.info(f"   Especies: {total_species}")
    log.info(f"   data.yaml: {data_yaml}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tifs", nargs="+", default=DEFAULT_TIFS)
    parser.add_argument("--output", default="/tmp/species_dataset_final")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--max-crops", type=int, default=15)
    parser.add_argument("--min-conf-vlm", type=float, default=0.30)
    args = parser.parse_args()

    result = build_dataset(
        tif_paths=args.tifs,
        output_dir=args.output,
        conf=args.conf,
        max_crops_per_tile=args.max_crops,
        min_conf_vlm=args.min_conf_vlm,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
