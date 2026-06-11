import { useState } from 'react'
import { classifySpecies, ClassifyResponse, SpeciesSummary } from '../api/client'

interface Props {
  jobId: string
}

const HEALTH_COLORS: Record<string, string> = {
  saludable: '#22c55e',
  estresado: '#f59e0b',
  enfermo: '#ef4444',
}

function HealthBadge({ label, count }: { label: string; count: number }) {
  if (count === 0) return null
  return (
    <span
      className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
      style={{ backgroundColor: HEALTH_COLORS[label] + '33', color: HEALTH_COLORS[label] }}
    >
      {label} {count}
    </span>
  )
}

function SpeciesRow({ s }: { s: SpeciesSummary }) {
  const barW = Math.round(s.pct)
  return (
    <div className="flex flex-col gap-1 py-2 border-b border-slate-800 last:border-0">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-200">{s.species}</span>
        <span className="text-xs font-mono text-slate-400">
          {s.count} árboles · {s.pct}%
        </span>
      </div>

      {/* Barra de progreso */}
      <div className="w-full h-1.5 rounded-full bg-slate-700 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${barW}%`, backgroundColor: '#22c55e' }}
        />
      </div>

      {/* Health badges */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <HealthBadge label="saludable" count={s.health_saludable} />
        <HealthBadge label="estresado" count={s.health_estresado} />
        <HealthBadge label="enfermo" count={s.health_enfermo} />
        <span className="text-[10px] text-slate-600 ml-auto">
          confianza avg {Math.round(s.avg_confidence * 100)}%
        </span>
      </div>
    </div>
  )
}

export default function SpeciesPanel({ jobId }: Props) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ClassifyResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sampleTiles, setSampleTiles] = useState(20)

  async function handleClassify() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await classifySpecies({
        job_id: jobId,
        sample_tiles: sampleTiles,
        max_crops_per_tile: 15,
        concurrency: 5,
      })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="rounded-2xl border p-5 flex flex-col gap-4"
      style={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-200">🌿 Clasificación de Especies</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            GPT-4o-mini Vision · CLIP · HDBSCAN — NOA/Tucumán
          </p>
        </div>

        {/* Tiles a samplear */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Tiles a analizar:</span>
          <select
            value={sampleTiles}
            onChange={(e) => setSampleTiles(Number(e.target.value))}
            disabled={loading}
            className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-200"
          >
            {[10, 20, 30, 50].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Botón */}
      {!result && !loading && (
        <button
          onClick={handleClassify}
          className="w-full py-3 rounded-xl font-semibold text-sm bg-emerald-700 hover:bg-emerald-600 text-white transition-colors"
        >
          🔬 Clasificar Especies + Salud
        </button>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center gap-3 py-8">
          <div
            className="w-10 h-10 border-4 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: '#22c55e', borderTopColor: 'transparent' }}
          />
          <p className="text-sm text-slate-400">
            Analizando {sampleTiles} tiles con GPT-4o-mini + CLIP...
          </p>
          <p className="text-xs text-slate-600">Esto puede tardar 30-90 segundos</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="rounded-xl border px-4 py-3 text-sm"
          style={{ backgroundColor: 'rgba(127,29,29,0.3)', borderColor: '#991b1b', color: '#fca5a5' }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* Resultados */}
      {result && (
        <div className="flex flex-col gap-4">
          {/* Stats rápidas */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Clasificados', value: `${result.classified_trees}/${result.total_trees}` },
              { label: 'Clusters', value: result.n_clusters },
              { label: 'Tiempo', value: `${result.elapsed_sec}s` },
            ].map((s) => (
              <div
                key={s.label}
                className="rounded-xl border p-3 text-center"
                style={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
              >
                <p className="text-lg font-bold text-white">{s.value}</p>
                <p className="text-[10px] text-slate-500 mt-0.5">{s.label}</p>
              </div>
            ))}
          </div>

          {/* Tabla de especies */}
          <div
            className="rounded-xl border px-4 py-2"
            style={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
          >
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">
              Composición del rodal
            </p>
            {result.species_summary.map((s) => (
              <SpeciesRow key={s.species} s={s} />
            ))}
          </div>

          {/* Volver a clasificar */}
          <button
            onClick={() => { setResult(null); setError(null) }}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors text-center"
          >
            ↩ Volver a clasificar
          </button>
        </div>
      )}
    </div>
  )
}
