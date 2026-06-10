"""
GET /api/v1/results/{job_id}
Retorna el resultado almacenado de un job procesado.

Nota: en este PoC los resultados se guardan en memoria (dict global).
En producción se reemplazaría por una base de datos o Redis.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import JobResult

router = APIRouter()

# Almacén en memoria: job_id → JobResult
# Los endpoints de process/compare pueden actualizar este registro.
_job_store: dict[str, JobResult] = {}


def store_job(job: JobResult) -> None:
    """Registra o actualiza un job en el store."""
    _job_store[job.job_id] = job


def get_job(job_id: str) -> JobResult | None:
    return _job_store.get(job_id)


@router.get(
    "/results/{job_id}",
    response_model=JobResult,
    summary="Obtener resultado de un job",
)
def get_results(job_id: str) -> JobResult:
    """
    Retorna los datos del job: archivo subido, estado y resultado del pipeline.

    Si el job_id no existe retorna 404.
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron resultados para job_id='{job_id}'."
        )
    return job
