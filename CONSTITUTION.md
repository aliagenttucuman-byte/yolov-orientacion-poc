# CONSTITUTION — YOLOv Orientación PoC

**Fecha:** 2026-06-10
**Autor:** Tony Stark + JARVIS
**Tipo:** I+D+I — PoC de investigación
**Repo:** https://github.com/aliagenttucuman-byte/yolov-orientacion-poc

---

## 1. Propósito

Validar qué versión de YOLO (v8n / v9c / v11n) detecta mejor copas de árboles
en ortofotos aéreas, con una interfaz manual donde el operador puede:

1. Subir un ortomosaico (.tif / .jpg / .png)
2. Elegir la versión YOLO y parámetros de inferencia
3. Ver los resultados: conteo, bboxes sobre el tile, métricas

Esto informa la decisión técnica para incorporar YOLO a ForestAI.

---

## 2. Stack

### Backend
- **Python 3.11** + **FastAPI** + Pydantic v2
- **Procesamiento:** ultralytics (YOLO), rasterio, opencv-python, numpy
- **Background jobs:** ninguno — procesamiento síncrono (PoC)
- **Testing:** mínimo — solo smoke tests de endpoint
- **Puerto:** 8020

### Frontend
- **React 19** + **Vite 6** + **TypeScript**
- **Styling:** Tailwind CSS 4
- **Estado:** React Query 5 (TanStack)
- **UI:** drag & drop para subir TIF, selector de modelo, visor de resultados
- **Puerto:** 3020

### Infraestructura
- **Docker Compose** — backend + frontend + nginx proxy
- **Puerto público:** 9020 (nginx spa_proxy estilo ForestAI)
- **Upload dir:** `/tmp/yolov-uploads` (se borra en reinicio — es PoC)
- **Sin DB** — resultados en memoria/JSON por sesión
- **Túnel:** Cloudflare si se necesita demo remota

---

## 3. Reglas de Negocio

1. El operador sube un ortomosaico → el sistema lo tiletea automáticamente
2. El operador elige modelo YOLO (yolov8n / yolov9c / yolo11n)
3. El sistema corre inferencia tile por tile y aplica NMS global
4. Los resultados muestran: conteo total, tiles procesados, tiempo, y tiles anotados
5. El operador puede comparar 2 modelos sobre el mismo TIF (modo comparativa)
6. Los archivos subidos son temporales — no persisten entre reinicios

---

## 4. Criterios de Aceptación (Definition of Done PoC)

- [ ] Drag & drop de ortomosaico funciona (TIF, JPG, PNG)
- [ ] Selector de modelo YOLO con al menos 3 opciones
- [ ] Pipeline completo: upload → tiling → inference → NMS → resultado
- [ ] Resultado muestra: árbol count, tiempo, tiles procesados
- [ ] Vista de tiles anotados (al menos los 5 primeros con detecciones)
- [ ] Modo comparativa: 2 modelos sobre el mismo TIF, side by side
- [ ] Funciona en el servidor ai-server (CPU, sin GPU dedicada)

---

## 5. Restricciones

- Sin autenticación (PoC interno)
- Sin persistencia entre sesiones
- Tiempo máximo de procesamiento por tile: no hay límite en PoC
- Compatible con CPU (no depender de CUDA)
- NO mezclar código con ForestAI — repos separados

---

## 6. Equipo I+D+I

| Agente | Rol |
|--------|-----|
| Julián | Backend FastAPI + pipeline YOLO |
| Mercedes | Frontend React + UI upload + visor |
| JARVIS | Arquitectura, coordinación, revisión |
| Tony | Decisión, criterios, validación final |
