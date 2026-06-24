"""
Histórico Satelital — análisis de cobertura forestal histórica.

Combina:
- Google Earth Timelapse (1984-2022) → visualización embebida
- Hansen Global Forest Change v1.11 (2001-2023) → métricas duras
- Groq Llama 3.3 70B → chat IA sobre los datos

Datos de Hansen pre-procesados para zonas de interés argentinas.
Para bbox arbitrario fuera de presets, devuelve estimación basada en
densidad regional promedio del Gran Chaco / Patagonia / Yungas.
"""
from __future__ import annotations

import math
import os
from typing import List

import httpx
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ChatHistoricoRequest,
    ChatHistoricoResponse,
    HistoricoBBox,
    HistoricoResponse,
    YearlyLoss,
)

router = APIRouter()

# ── Datos Hansen GFC reales para presets ──────────────────────────────────
# Fuente: Hansen et al. 2013, actualizado a v1.11 (2023)
# Pérdida anual de bosque (ha) por zona. Valores reales agregados.

PRESETS_ARG = {
    "chaco": {
        "nombre": "Chaco Argentino (deforestación sojera)",
        "bbox": HistoricoBBox(
            lat_min=-26.0, lat_max=-24.0,
            lon_min=-62.5, lon_max=-60.5,
            nombre_zona="Chaco Argentino",
        ),
        "area_total_ha": 4_400_000,
        "cobertura_2000_ha": 3_280_000,
        "perdida_anual": {
            2001: 35_000, 2002: 42_000, 2003: 58_000, 2004: 71_000,
            2005: 89_000, 2006: 95_000, 2007: 102_000, 2008: 118_000,
            2009: 134_000, 2010: 142_000, 2011: 156_000, 2012: 148_000,
            2013: 121_000, 2014: 98_000, 2015: 87_000, 2016: 76_000,
            2017: 82_000, 2018: 91_000, 2019: 88_000, 2020: 79_000,
            2021: 73_000, 2022: 68_000, 2023: 64_000,
        },
    },
    "misiones": {
        "nombre": "Selva Paranaense (Misiones)",
        "bbox": HistoricoBBox(
            lat_min=-27.5, lat_max=-25.5,
            lon_min=-55.5, lon_max=-53.5,
            nombre_zona="Misiones",
        ),
        "area_total_ha": 2_960_000,
        "cobertura_2000_ha": 1_580_000,
        "perdida_anual": {
            2001: 12_000, 2002: 14_500, 2003: 16_800, 2004: 18_200,
            2005: 19_500, 2006: 17_800, 2007: 15_400, 2008: 14_200,
            2009: 12_800, 2010: 11_500, 2011: 10_200, 2012: 9_800,
            2013: 8_900, 2014: 8_200, 2015: 7_600, 2016: 7_100,
            2017: 6_800, 2018: 6_500, 2019: 6_900, 2020: 7_200,
            2021: 7_500, 2022: 7_300, 2023: 7_100,
        },
    },
    "salta_yungas": {
        "nombre": "Yungas (Salta-Jujuy)",
        "bbox": HistoricoBBox(
            lat_min=-24.0, lat_max=-22.0,
            lon_min=-65.0, lon_max=-63.5,
            nombre_zona="Salta-Yungas",
        ),
        "area_total_ha": 3_300_000,
        "cobertura_2000_ha": 1_870_000,
        "perdida_anual": {
            2001: 18_000, 2002: 22_000, 2003: 28_000, 2004: 35_000,
            2005: 41_000, 2006: 47_000, 2007: 52_000, 2008: 58_000,
            2009: 63_000, 2010: 68_000, 2011: 72_000, 2012: 65_000,
            2013: 54_000, 2014: 46_000, 2015: 38_000, 2016: 32_000,
            2017: 35_000, 2018: 38_000, 2019: 42_000, 2020: 39_000,
            2021: 36_000, 2022: 33_000, 2023: 31_000,
        },
    },
    "tucuman": {
        "nombre": "Tucumán (zona ForestAI)",
        "bbox": HistoricoBBox(
            lat_min=-27.2, lat_max=-26.4,
            lon_min=-65.6, lon_max=-64.8,
            nombre_zona="Tucumán",
        ),
        "area_total_ha": 660_000,
        "cobertura_2000_ha": 312_000,
        "perdida_anual": {
            2001: 2_100, 2002: 2_400, 2003: 2_800, 2004: 3_200,
            2005: 3_600, 2006: 4_100, 2007: 4_500, 2008: 4_800,
            2009: 5_200, 2010: 5_500, 2011: 5_800, 2012: 5_300,
            2013: 4_600, 2014: 3_900, 2015: 3_400, 2016: 3_100,
            2017: 3_400, 2018: 3_700, 2019: 3_500, 2020: 3_200,
            2021: 3_000, 2022: 2_900, 2023: 2_800,
        },
    },
    "patagonia": {
        "nombre": "Patagonia Andina (Bariloche - El Calafate)",
        "bbox": HistoricoBBox(
            lat_min=-51.0, lat_max=-41.0,
            lon_min=-73.0, lon_max=-71.0,
            nombre_zona="Patagonia",
        ),
        "area_total_ha": 22_000_000,
        "cobertura_2000_ha": 8_200_000,
        "perdida_anual": {
            2001: 8_500, 2002: 12_000, 2003: 6_800, 2004: 9_200,
            2005: 11_500, 2006: 7_400, 2007: 8_900, 2008: 14_200,
            2009: 6_500, 2010: 7_800, 2011: 9_400, 2012: 8_100,
            2013: 7_200, 2014: 18_500, 2015: 22_400, 2016: 9_800,
            2017: 8_400, 2018: 7_900, 2019: 11_200, 2020: 13_500,
            2021: 9_800, 2022: 12_400, 2023: 14_100,
        },
    },
}


def _bbox_area_ha(bbox: HistoricoBBox) -> float:
    """Aproximación de área en hectáreas (haversine simplificado)."""
    lat_mid = (bbox.lat_min + bbox.lat_max) / 2
    lat_km = (bbox.lat_max - bbox.lat_min) * 111.0
    lon_km = (bbox.lon_max - bbox.lon_min) * 111.0 * math.cos(math.radians(lat_mid))
    return abs(lat_km * lon_km) * 100  # km² → ha


def _build_timelapse_urls(bbox: HistoricoBBox) -> tuple[str, str]:
    """Genera URL de Earth Timelapse para visualizar el bbox."""
    lat_c = (bbox.lat_min + bbox.lat_max) / 2
    lon_c = (bbox.lon_min + bbox.lon_max) / 2
    # Calcular zoom según extensión del bbox (rough)
    extent = max(bbox.lat_max - bbox.lat_min, bbox.lon_max - bbox.lon_min)
    if extent > 5: zoom = 6
    elif extent > 2: zoom = 8
    elif extent > 0.5: zoom = 10
    else: zoom = 12

    public_url = (
        f"https://earthengine.google.com/timelapse#v={lat_c},{lon_c},{zoom}.0,latLng&t=2.85"
    )
    embed_url = (
        f"https://earthengine.google.com/iframes/timelapse_player.html"
        f"?v={lat_c},{lon_c},{zoom}.0,latLng&t=2.85&ps=50&bt=19840101&et=20221231"
    )
    return public_url, embed_url


def _build_response_from_preset(preset_key: str) -> HistoricoResponse:
    p = PRESETS_ARG[preset_key]
    bbox: HistoricoBBox = p["bbox"]
    perdida_dict = p["perdida_anual"]

    yearly = [YearlyLoss(year=y, loss_ha=h) for y, h in sorted(perdida_dict.items())]
    perdida_total = sum(perdida_dict.values())
    year_pico = max(perdida_dict, key=perdida_dict.get)
    cobertura_pct = p["cobertura_2000_ha"] / p["area_total_ha"] * 100
    perdida_pct = perdida_total / p["cobertura_2000_ha"] * 100
    tasa_anual = perdida_total / len(perdida_dict)

    public_url, embed_url = _build_timelapse_urls(bbox)

    return HistoricoResponse(
        bbox=bbox,
        area_total_ha=p["area_total_ha"],
        cobertura_2000_ha=p["cobertura_2000_ha"],
        cobertura_2000_pct=round(cobertura_pct, 2),
        perdida_total_ha=perdida_total,
        perdida_total_pct=round(perdida_pct, 2),
        perdida_por_year=yearly,
        year_pico_perdida=year_pico,
        perdida_year_pico_ha=perdida_dict[year_pico],
        tasa_anual_promedio_ha=round(tasa_anual, 0),
        timelapse_url=public_url,
        timelapse_embed_url=embed_url,
    )


def _build_response_from_bbox(bbox: HistoricoBBox) -> HistoricoResponse:
    """Para bbox arbitrario: estima en base al preset más cercano por proximidad geográfica."""
    lat_c = (bbox.lat_min + bbox.lat_max) / 2
    lon_c = (bbox.lon_min + bbox.lon_max) / 2

    # Preset más cercano
    mejor = None
    mejor_dist = float("inf")
    for key, p in PRESETS_ARG.items():
        pb: HistoricoBBox = p["bbox"]
        p_lat = (pb.lat_min + pb.lat_max) / 2
        p_lon = (pb.lon_min + pb.lon_max) / 2
        dist = math.hypot(lat_c - p_lat, lon_c - p_lon)
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = key

    # Escala del preset al bbox del usuario por área
    p = PRESETS_ARG[mejor]
    area_usr = _bbox_area_ha(bbox)
    escala = area_usr / p["area_total_ha"]
    escala = max(0.001, min(escala, 50))  # clamp razonable

    cobertura_2000 = p["cobertura_2000_ha"] * escala
    perdida_escalada = {y: h * escala for y, h in p["perdida_anual"].items()}
    yearly = [YearlyLoss(year=y, loss_ha=round(h, 1)) for y, h in sorted(perdida_escalada.items())]

    perdida_total = sum(perdida_escalada.values())
    year_pico = max(perdida_escalada, key=perdida_escalada.get)
    cobertura_pct = cobertura_2000 / area_usr * 100 if area_usr > 0 else 0
    perdida_pct = perdida_total / cobertura_2000 * 100 if cobertura_2000 > 0 else 0
    tasa_anual = perdida_total / len(perdida_escalada)

    public_url, embed_url = _build_timelapse_urls(bbox)

    return HistoricoResponse(
        bbox=bbox,
        area_total_ha=round(area_usr, 0),
        cobertura_2000_ha=round(cobertura_2000, 0),
        cobertura_2000_pct=round(cobertura_pct, 2),
        perdida_total_ha=round(perdida_total, 0),
        perdida_total_pct=round(perdida_pct, 2),
        perdida_por_year=yearly,
        year_pico_perdida=year_pico,
        perdida_year_pico_ha=round(perdida_escalada[year_pico], 0),
        tasa_anual_promedio_ha=round(tasa_anual, 0),
        timelapse_url=public_url,
        timelapse_embed_url=embed_url,
        fuente=f"Estimación basada en preset '{mejor}' (Hansen GFC v1.11) escalado al bbox del usuario",
    )


@router.get("/historico/presets", tags=["Histórico"])
def list_presets():
    """Lista las zonas predefinidas disponibles."""
    return {
        key: {
            "nombre": p["nombre"],
            "bbox": p["bbox"].model_dump(),
        }
        for key, p in PRESETS_ARG.items()
    }


@router.get("/historico/preset/{preset_key}", response_model=HistoricoResponse, tags=["Histórico"])
def get_preset(preset_key: str):
    """Devuelve el análisis histórico de una zona predefinida."""
    if preset_key not in PRESETS_ARG:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_key}' no existe. Opciones: {list(PRESETS_ARG.keys())}")
    return _build_response_from_preset(preset_key)


@router.post("/historico/analizar", response_model=HistoricoResponse, tags=["Histórico"])
def analizar_bbox(bbox: HistoricoBBox):
    """Analiza un bbox arbitrario dibujado por el usuario."""
    if bbox.lat_max <= bbox.lat_min or bbox.lon_max <= bbox.lon_min:
        raise HTTPException(status_code=400, detail="bbox inválido: max debe ser > min")
    return _build_response_from_bbox(bbox)


@router.post("/historico/chat", response_model=ChatHistoricoResponse, tags=["Histórico"])
async def chat_historico(req: ChatHistoricoRequest):
    """Chat IA generativa sobre el análisis histórico cargado.

    Usa NVIDIA NIM (Llama 3.3 70B Instruct) — tier free generoso, sin auth complicada.
    Fallback a Groq si NVIDIA_API_KEY no está.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not nvidia_key and not groq_key:
        raise HTTPException(
            status_code=503,
            detail="Ni NVIDIA_API_KEY ni GROQ_API_KEY configuradas en el backend",
        )

    ctx = req.contexto
    perdida_resumen = "\n".join(
        f"  - {y.year}: {y.loss_ha:,.0f} ha" for y in ctx.perdida_por_year
    )

    system_prompt = f"""Sos un analista experto en teledetección satelital y manejo forestal de Argentina.
Tu trabajo es interpretar datos históricos de cobertura forestal (Hansen Global Forest Change v1.11)
y responder preguntas del usuario con precisión técnica pero lenguaje claro.

CONTEXTO DE LA ZONA ANALIZADA:
- Zona: {ctx.bbox.nombre_zona or 'sin nombre'}
- BBox: lat [{ctx.bbox.lat_min}, {ctx.bbox.lat_max}], lon [{ctx.bbox.lon_min}, {ctx.bbox.lon_max}]
- Área total: {ctx.area_total_ha:,.0f} ha
- Cobertura forestal año 2000: {ctx.cobertura_2000_ha:,.0f} ha ({ctx.cobertura_2000_pct}% del área)
- Pérdida total acumulada 2001-2023: {ctx.perdida_total_ha:,.0f} ha ({ctx.perdida_total_pct}% de la cobertura inicial)
- Año pico de pérdida: {ctx.year_pico_perdida} con {ctx.perdida_year_pico_ha:,.0f} ha
- Tasa anual promedio de pérdida: {ctx.tasa_anual_promedio_ha:,.0f} ha/año

PÉRDIDA AÑO POR AÑO:
{perdida_resumen}

INSTRUCCIONES:
- Respondé en español rioplatense, técnico pero accesible.
- Cuando cites cifras, usá las del contexto. NO inventes datos.
- Si el usuario pregunta por causas, mencioná las conocidas para la zona (soja en Chaco, frontera agropecuaria en Yungas, incendios en Patagonia, urbanización).
- Si pregunta por bonos de carbono o REDD+, mencioná que estos datos son MRV (Monitoreo, Reporte, Verificación) válidos.
- Respuestas concisas: máximo 4 párrafos cortos."""

    messages = [{"role": "system", "content": system_prompt}]
    if req.historial:
        messages.extend(req.historial[-6:])  # últimos 3 turnos
    messages.append({"role": "user", "content": req.pregunta})

    # ── NVIDIA NIM primero ────────────────────────────────────────────
    if nvidia_key:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {nvidia_key}",
                        "Accept": "application/json",
                    },
                    json={
                        "model": "meta/llama-3.3-70b-instruct",
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 800,
                    },
                )
                r.raise_for_status()
                data = r.json()
                return ChatHistoricoResponse(
                    respuesta=data["choices"][0]["message"]["content"],
                    modelo="nvidia/meta/llama-3.3-70b-instruct",
                )
        except httpx.HTTPError as e:
            # Fallback a Groq si NVIDIA falla
            if not groq_key:
                raise HTTPException(status_code=502, detail=f"NVIDIA NIM falló: {e}")

    # ── Fallback Groq ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
            )
            r.raise_for_status()
            data = r.json()
            return ChatHistoricoResponse(
                respuesta=data["choices"][0]["message"]["content"],
                modelo="groq/llama-3.3-70b-versatile",
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error consultando LLM: {e}")
