# YOLOv Orientación PoC

> Comparativa experimental para determinar la mejor configuración de YOLO
> aplicada a detección de copas de árboles en ortofotos aéreas (ForestAI).

Repo: https://github.com/aliagenttucuman-byte/yolov-orientacion-poc

---

## Flujo completo

```
GeoTIFF (ortofoto)
      │
      ▼
 [tiler.py]          tiles 1024px con overlap 128px + ExG filter
      │
      ▼
 [detector.py]       YOLO inference por tile → bboxes locales
      │
      ▼
 [NMS global]        elimina duplicados de tiles solapados
      │
      ▼
 [reporter.py]       métricas, comparativa, plots, JSON
```

El mismo pipeline corre para **yolov8n / yolo11n / yolov9c** en paralelo.

---

## Setup rápido

```bash
git clone https://github.com/aliagenttucuman-byte/yolov-orientacion-poc
cd yolov-orientacion-poc
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Uso

```bash
# 1. Construir dataset (una sola vez)
make dataset

# 2. Comparativa completa (train + infer + plot)
make compare TIF=/ruta/a/ortofoto.tif

# 3. Solo inferencia rápida con un modelo
make infer TIF=/ruta/a/ortofoto.tif

# 4. Ver leaderboard
make report
```

### Equivalente directo con Python

```bash
# Dataset
python dataset/build_dataset.py

# Fine-tune los 3 modelos y comparar
python run.py --mode compare \
  --tif /home/server/proyectos/forestai-poc/uploads/tu_ortofoto.tif \
  --models yolov8n yolo11n yolov9c \
  --epochs 80

# Solo inferencia (ya entrenado)
python run.py --mode infer \
  --tif /ruta/a/ortofoto.tif \
  --model yolo11n
```

---

## Modelos soportados

| Key | Base model | Params | Velocidad |
|-----|-----------|--------|-----------|
| yolov8n | yolov8n.pt | 3.2M | ★★★★★ |
| yolov8s | yolov8s.pt | 11M | ★★★★ |
| yolov9c | yolov9c.pt | 25M | ★★★ |
| yolo11n | yolo11n.pt | 2.6M | ★★★★★ |
| yolo11s | yolo11s.pt | 9.4M | ★★★★ |

---

## Spikes

| # | Nombre | Pregunta |
|---|--------|----------|
| 001 | [yolo-versions](spikes/001-yolo-versions/) | ¿Cuál versión obtiene mayor mAP50? |
| 002 | [tile-size](spikes/002-tile-size/) | ¿Qué imgsz es óptimo para copas pequeñas? |
| 003 | [label-quality](spikes/003-label-quality/) | ¿Cuánto ruido introduce ExG vs manual? |
| 004 | [inference-speed](spikes/004-inference-speed/) | ¿Cuál es el throughput CPU vs GPU? |

---

## Output

```
resultados/
├── metrics.csv              ← tabla comparativa de todos los modelos
├── metrics.json             ← para consumo programático
├── comparativa_yolov.png    ← bar chart mAP50 / Precision / Recall
└── samples/                 ← tiles anotados por cada modelo
```

---

## Diferencias con ForestAI

| Aspecto | ForestAI | Este PoC |
|---------|----------|----------|
| Modelo | DeepForest (pretrained) | YOLOv fine-tuned |
| Labels | Pre-entrenado NEON | ExG pseudo-labels |
| Segmentación | SAM (polígonos) | Bounding boxes |
| Objetivo | Producción | Investigación/comparativa |
