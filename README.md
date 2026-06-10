# YOLOv Orientación PoC

> PoC de investigación independiente para determinar la mejor configuración de YOLO
> aplicada a detección de copas de árboles en ortofotos aéreas.

**Proyecto:** AlegentAI / ForestAI  
**Estado:** En curso  
**Stack:** Python 3.11 · ultralytics · rasterio · OpenCV

---

## Pregunta central

¿Qué combinación de (versión YOLO) + (tamaño de imagen) + (calidad de anotaciones)
maximiza mAP50 para detección de árboles en ortofotos urbanas/rurales?

---

## Spikes

| # | Spike | Pregunta | Estado |
|---|-------|----------|--------|
| 001 | [yolo-versions](spikes/001-yolo-versions/) | YOLOv8n vs YOLO11n vs YOLOv9c — ¿cuál gana con datos forestales? | 🔄 pendiente |
| 002 | [tile-size](spikes/002-tile-size/) | imgsz 640 vs 1024 vs 1280 — impacto en copas pequeñas | 🔄 pendiente |
| 003 | [label-quality](spikes/003-label-quality/) | ExG pseudo-labels vs anotación manual — ¿cuánto ruido hay? | 🔄 pendiente |
| 004 | [inference-speed](spikes/004-inference-speed/) | Throughput CPU vs GPU — tiles/minuto | 🔄 pendiente |

---

## Estructura

```
yolov-orientacion-poc/
├── README.md
├── requirements.txt
├── dataset/                  # tiles compartidos por todos los spikes
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
├── spikes/
│   ├── 001-yolo-versions/
│   ├── 002-tile-size/
│   ├── 003-label-quality/
│   └── 004-inference-speed/
├── resultados/               # plots y comparativas finales
└── docs/
    └── conclusion.md         # recomendación final para ForestAI
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Reglas del proyecto

1. Cada spike es autónomo — tiene su propio `README.md` con veredicto
2. Todos usan `dataset/` compartido — resultados comparables
3. Nada de código de producción — esto es investigación
4. Al terminar: `docs/conclusion.md` con la recomendación para ForestAI
