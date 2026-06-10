.PHONY: dataset train infer compare report clean

TIF ?= /ruta/a/tu/ortofoto.tif
MODELS ?= yolov8n yolo11n yolov9c
EPOCHS ?= 80

# 1. Construir dataset compartido
dataset:
	python dataset/build_dataset.py

# 2. Fine-tune todos los modelos
train:
	python run.py --mode train --models $(MODELS) --epochs $(EPOCHS)

# 3. Inferencia sobre un TIF con un modelo específico
infer:
	python run.py --mode infer --tif $(TIF) --model yolo11n

# 4. Comparativa completa (train + infer + plot)
compare:
	python run.py --mode compare --tif $(TIF) --models $(MODELS) --epochs $(EPOCHS)

# 5. Ver resultados
report:
	python run.py --mode report

# Limpiar runs y cache
clean:
	rm -rf runs/ tiles_cache/ resultados/

# Setup entorno
setup:
	pip install -r requirements.txt
