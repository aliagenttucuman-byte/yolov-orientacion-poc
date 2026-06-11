"""
species_classifier.py — Clasificación de especies + salud por copa de árbol.

Flujo combinado:
  1. LLM Vision (gpt-4o-mini) sobre el tile completo → especie por zona
  2. CLIP embeddings de cada crop (bbox) → clustering HDBSCAN
  3. Asignación: especie del tile LLM → cluster más cercano → cada árbol

Especies del NOA/Tucumán cubiertas:
  Quebracho blanco, Quebracho colorado, Algarrobo negro, Algarrobo blanco,
  Cebil colorado, Horco quebracho, Tipa blanca, Lapacho rosado,
  Palo borracho, Guarán, Cedro tucumano, Vinal
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL  = "https://api.openai.com/v1/chat/completions"
OPENAI_VLM_MODEL = "gpt-4o-mini"

NOA_SPECIES = [
    "Quebracho blanco (Aspidosperma quebracho-blanco)",
    "Quebracho colorado (Schinopsis lorentzii)",
    "Algarrobo negro (Prosopis nigra)",
    "Algarrobo blanco (Prosopis alba)",
    "Cebil colorado (Anadenanthera colubrina)",
    "Horco quebracho (Schinopsis haenkeana)",
    "Tipa blanca (Tipuana tipu)",
    "Lapacho rosado (Handroanthus impetiginosus)",
    "Palo borracho (Ceiba insignis)",
    "Guarán (Tecoma stans)",
    "Cedro tucumano (Cedrela lilloi)",
    "Vinal (Prosopis ruscifolia)",
]

TILE_SYSTEM_PROMPT = (
    "You are an expert remote sensing analyst specializing in NOA (Noroeste Argentino) forestry. "
    "You analyze top-down aerial/drone RGB tile images of forest canopy from Tucumán, Argentina. "
    "Your task: identify which tree species are visible in this tile and their approximate health. "
    f"Known species in the area: {', '.join(NOA_SPECIES)}. "
    "Analyze canopy shape, color, texture, and tone patterns from above. "
    "ALWAYS make your best estimate. If uncertain, pick the most likely species. "
    "RESPOND ONLY with a valid JSON array. No prose, no markdown. "
    'Format: [{"species": "nombre en español", "health": "saludable|estresado|enfermo", '
    '"confidence": 0.7, "zone": "NW|NE|SW|SE|CENTER", "notes": "max 8 words"}]'
)

CROP_SYSTEM_PROMPT = (
    "You are a NOA forestry expert analyzing a single tree canopy crop (top-down drone view) "
    "from Tucumán, Argentina. "
    f"Known species: {', '.join(NOA_SPECIES)}. "
    "ALWAYS pick the most likely species. "
    "RESPOND ONLY with valid JSON. "
    '{"species": "nombre en español", "health": "saludable|estresado|enfermo", '
    '"confidence": 0.6, "notes": "max 8 words"}'
)

MIN_CROP_PX  = 20
TARGET_SIZE  = 224
PADDING      = 10
MAX_TILE_PX  = 800   # resize tiles grandes antes de mandar al LLM


# ── Helpers de imagen ─────────────────────────────────────────────────────────

def _img_to_b64(pil_img: Image.Image, max_px: int = 800, quality: int = 85) -> str:
    """Convierte PIL Image a base64 JPEG. Resize si supera max_px en algún lado."""
    if max(pil_img.width, pil_img.height) > max_px:
        ratio = max_px / max(pil_img.width, pil_img.height)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _crop_tree(pil_img: Image.Image, det: Dict[str, Any]) -> Optional[Image.Image]:
    """Crop con padding de un detection dict {x1, y1, x2, y2}."""
    x1 = max(0, det["x1"] - PADDING)
    y1 = max(0, det["y1"] - PADDING)
    x2 = min(pil_img.width,  det["x2"] + PADDING)
    y2 = min(pil_img.height, det["y2"] + PADDING)
    w, h = x2 - x1, y2 - y1
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return None
    crop = pil_img.crop((x1, y1, x2, y2))
    if crop.width < TARGET_SIZE or crop.height < TARGET_SIZE:
        crop = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return crop


# ── Step 1: LLM Vision por tile ───────────────────────────────────────────────

async def _classify_tile(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    tile_b64: str,
    tile_filename: str,
) -> Dict[str, Any]:
    """Manda un tile al LLM y obtiene lista de especies detectadas."""
    payload = {
        "model": OPENAI_VLM_MODEL,
        "max_tokens": 300,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": TILE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{tile_b64}",
                        "detail": "high"
                    }},
                    {"type": "text", "text": "Identify all tree species visible in this aerial tile. JSON array only."},
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        try:
            async with session.post(
                OPENAI_API_URL, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"LLM tile {tile_filename}: HTTP {resp.status} — {body[:200]}")
                    return {"tile_filename": tile_filename, "ok": False, "species_list": []}

                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Limpiar markdown si viene
                if "```" in content:
                    content = re.sub(r"```[a-z]*\n?", "", content).strip()

                # Extraer array JSON
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    species_list = json.loads(match.group(0))
                else:
                    species_list = json.loads(content)

                return {
                    "tile_filename": tile_filename,
                    "ok": True,
                    "species_list": species_list,
                }

        except Exception as e:
            logger.warning(f"LLM tile {tile_filename}: {type(e).__name__}: {e}")
            return {"tile_filename": tile_filename, "ok": False, "species_list": []}


# ── Step 2: CLIP embeddings + clustering ─────────────────────────────────────

def _get_clip_embeddings(crops_b64: List[str]) -> Optional[np.ndarray]:
    """
    Obtiene embeddings CLIP para los crops.
    Usa transformers (clip-vit-base-patch32) local — sin costo de API.
    Fallback: embeddings de color simple si CLIP no está disponible.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        images = []
        for b64 in crops_b64:
            img_bytes = base64.b64decode(b64)
            images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        embeddings = features.numpy()
        # Normalizar
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-9)

    except Exception as e:
        logger.warning(f"CLIP no disponible ({e}), usando embeddings de color")
        return _color_embeddings(crops_b64)


def _color_embeddings(crops_b64: List[str]) -> np.ndarray:
    """Fallback: histograma de color HSV como embedding simple."""
    embeddings = []
    for b64 in crops_b64:
        try:
            img_bytes = base64.b64decode(b64)
            img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB").resize((32, 32)))
            # Stats RGB simples como features
            feats = []
            for c in range(3):
                ch = img[:, :, c].flatten().astype(float) / 255.0
                feats.extend([ch.mean(), ch.std(), np.percentile(ch, 25), np.percentile(ch, 75)])
            embeddings.append(feats)
        except Exception:
            embeddings.append([0.0] * 12)
    arr = np.array(embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-9)


def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Clustering con HDBSCAN si está disponible, fallback a KMeans.
    Retorna array de labels (int) por embedding.
    """
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, len(embeddings) // 10))
        labels = clusterer.fit_predict(embeddings)
        # HDBSCAN puede asignar -1 (noise) → los tratamos como cluster propio
        return labels
    except ImportError:
        pass

    try:
        from sklearn.cluster import KMeans
        k = min(n_clusters, len(embeddings))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(embeddings)
    except Exception as e:
        logger.warning(f"Clustering fallback lineal: {e}")
        return np.zeros(len(embeddings), dtype=int)


# ── Step 3: Asignación tile-species → clusters ───────────────────────────────

def _assign_species_to_clusters(
    cluster_labels: np.ndarray,
    det_to_tile: List[str],
    tile_species: Dict[str, List[Dict]],
) -> Dict[int, str]:
    """
    Para cada cluster, vota la especie más frecuente según los tiles
    a los que pertenecen las detecciones del cluster.
    Retorna {cluster_label: especie}.
    """
    from collections import Counter

    cluster_votes: Dict[int, Counter] = {}
    for i, label in enumerate(cluster_labels):
        tile_fn = det_to_tile[i]
        species_list = tile_species.get(tile_fn, [])
        if not species_list:
            continue
        # Usar la especie de mayor confianza del tile
        best = max(species_list, key=lambda s: s.get("confidence", 0))
        sp = best.get("species", "Desconocida")

        if label not in cluster_votes:
            cluster_votes[label] = Counter()
        cluster_votes[label][sp] += 1

    # Para cada cluster: especie más votada
    cluster_species = {}
    for label, counter in cluster_votes.items():
        cluster_species[label] = counter.most_common(1)[0][0]

    return cluster_species


# ── Pipeline principal ────────────────────────────────────────────────────────

async def classify_species(
    tiles_dir: Path,
    detecciones: List[Dict[str, Any]],
    sample_tiles: int = 20,
    max_crops_per_tile: int = 15,
    concurrency: int = 5,
) -> List[Dict[str, Any]]:
    """
    Pipeline completo: LLM Vision por tile + CLIP clustering + asignación.

    Args:
        tiles_dir:         Directorio con los tiles JPG ya generados.
        detecciones:       Lista de detecciones del pipeline YOLO
                           (cada item tiene tile_filename, x1, y1, x2, y2).
        sample_tiles:      Cuántos tiles distintos mandar al LLM (costo control).
        max_crops_per_tile: Máx crops por tile para CLIP.
        concurrency:       Requests paralelos al LLM.

    Returns:
        Lista de detecciones enriquecidas con species, health, confidence, cluster_id.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY no configurada — clasificación omitida")
        return [{**d, "species": "Sin API key", "health": "desconocido",
                 "species_confidence": 0.0, "cluster_id": -1} for d in detecciones]

    # ── Agrupar detecciones por tile ─────────────────────────────────────
    from collections import defaultdict
    tile_dets: Dict[str, List[Dict]] = defaultdict(list)
    for det in detecciones:
        tile_dets[det["tile_filename"]].append(det)

    all_tiles = list(tile_dets.keys())
    logger.info(f"[species] {len(detecciones)} detecciones en {len(all_tiles)} tiles")

    # ── Step 1: LLM Vision — samplear tiles ──────────────────────────────
    # Priorizar tiles con más detecciones
    sorted_tiles = sorted(all_tiles, key=lambda t: len(tile_dets[t]), reverse=True)
    sampled_tiles = sorted_tiles[:sample_tiles]

    tile_b64_map: Dict[str, str] = {}
    for tile_fn in sampled_tiles:
        tile_path = tiles_dir / tile_fn
        if tile_path.exists():
            try:
                img = Image.open(tile_path).convert("RGB")
                tile_b64_map[tile_fn] = _img_to_b64(img, max_px=MAX_TILE_PX)
            except Exception as e:
                logger.warning(f"No se pudo leer tile {tile_fn}: {e}")

    logger.info(f"[species] Step 1: LLM Vision sobre {len(tile_b64_map)} tiles")
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _classify_tile(session, semaphore, b64, tile_fn)
            for tile_fn, b64 in tile_b64_map.items()
        ]
        tile_results = await asyncio.gather(*tasks)

    # Mapa tile_filename → lista de especies
    tile_species: Dict[str, List[Dict]] = {}
    for r in tile_results:
        if r["ok"] and r["species_list"]:
            tile_species[r["tile_filename"]] = r["species_list"]

    ok_tiles = len(tile_species)
    logger.info(f"[species] Step 1 completo: {ok_tiles}/{len(tile_b64_map)} tiles clasificados")

    # ── Step 2: CLIP embeddings de crops ─────────────────────────────────
    # Tomar hasta max_crops_per_tile crops de cada tile sampleado
    crops_b64: List[str] = []
    det_to_tile: List[str] = []
    det_indices: List[int] = []  # índice en detecciones original

    for tile_fn in sampled_tiles:
        dets_in_tile = tile_dets[tile_fn]
        tile_path = tiles_dir / tile_fn
        if not tile_path.exists():
            continue
        try:
            pil_img = Image.open(tile_path).convert("RGB")
        except Exception:
            continue

        # Priorizar crops más grandes
        sorted_dets = sorted(
            dets_in_tile,
            key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]),
            reverse=True
        )[:max_crops_per_tile]

        for det in sorted_dets:
            crop = _crop_tree(pil_img, det)
            if crop is None:
                continue
            buf = BytesIO()
            crop.save(buf, format="JPEG", quality=80)
            crops_b64.append(base64.b64encode(buf.getvalue()).decode())
            det_to_tile.append(tile_fn)
            # Buscar índice original
            orig_idx = next(
                (i for i, d in enumerate(detecciones)
                 if d["tile_filename"] == tile_fn and d["x1"] == det["x1"] and d["y1"] == det["y1"]),
                -1
            )
            det_indices.append(orig_idx)

    logger.info(f"[species] Step 2: CLIP embeddings sobre {len(crops_b64)} crops")

    cluster_labels = np.array([-1] * len(crops_b64))
    if len(crops_b64) >= 2:
        embeddings = _get_clip_embeddings(crops_b64)
        if embeddings is not None:
            cluster_labels = _cluster_embeddings(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            logger.info(f"[species] Step 2 completo: {n_clusters} clusters encontrados")

    # ── Step 3: Asignar especie por cluster ───────────────────────────────
    cluster_species = _assign_species_to_clusters(cluster_labels, det_to_tile, tile_species)
    logger.info(f"[species] Step 3: asignando especies → {cluster_species}")

    # Mapa det_index → (species, health, confidence, cluster_id)
    det_enrichment: Dict[int, Dict] = {}
    for i, orig_idx in enumerate(det_indices):
        if orig_idx < 0:
            continue
        label = int(cluster_labels[i])
        species = cluster_species.get(label, "Desconocida")
        tile_fn = det_to_tile[i]
        sp_list = tile_species.get(tile_fn, [])
        health = "desconocido"
        confidence = 0.5
        if sp_list:
            best = max(sp_list, key=lambda s: s.get("confidence", 0))
            health = best.get("health", "desconocido")
            confidence = float(best.get("confidence", 0.5))
        det_enrichment[orig_idx] = {
            "species": species,
            "health": health,
            "species_confidence": confidence,
            "cluster_id": label,
        }

    # ── Enriquecer todas las detecciones ──────────────────────────────────
    enriched = []
    for i, det in enumerate(detecciones):
        info = det_enrichment.get(i, {
            "species": "No clasificado",
            "health": "desconocido",
            "species_confidence": 0.0,
            "cluster_id": -1,
        })
        enriched.append({**det, **info})

    classified = sum(1 for d in enriched if d["species"] not in ("No clasificado", "Desconocida"))
    logger.info(f"[species] Pipeline completo: {classified}/{len(enriched)} árboles clasificados")

    return enriched


def classify_species_sync(
    tiles_dir: Path,
    detecciones: List[Dict[str, Any]],
    sample_tiles: int = 20,
    max_crops_per_tile: int = 15,
    concurrency: int = 5,
) -> List[Dict[str, Any]]:
    """Wrapper síncrono para llamar desde FastAPI (run en thread pool)."""
    return asyncio.run(classify_species(
        tiles_dir=tiles_dir,
        detecciones=detecciones,
        sample_tiles=sample_tiles,
        max_crops_per_tile=max_crops_per_tile,
        concurrency=concurrency,
    ))
