# Conclusión — Recomendación para ForestAI

> Este documento se completa al finalizar los 4 spikes.

## Veredicto esperado

- Versión YOLO recomendada: TBD
- Tamaño de imagen óptimo: TBD
- Estrategia de anotación: TBD
- Throughput estimado en producción: TBD

## Criterios de aceptación

- mAP50 >= 0.65 en tiles de validación no vistos
- Inference < 5 seg/tile en CPU (1024px)
- Pipeline reproducible con `python run_spike.py`
