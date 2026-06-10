import { ProcessResponse } from '../types'
import ResultsPanel from './ResultsPanel'

interface Props {
  results: [ProcessResponse, ProcessResponse]
}

const MODEL_LABELS: Record<string, string> = {
  yolov8n: 'YOLOv8n',
  yolov9c: 'YOLOv9c',
  yolo11n: 'YOLO11n',
}

export default function ComparePanel({ results }: Props) {
  const [a, b] = results
  const diff = b.tree_count - a.tree_count
  const fasterModel =
    a.elapsed_sec < b.elapsed_sec
      ? MODEL_LABELS[a.model_key] ?? a.model_key
      : MODEL_LABELS[b.model_key] ?? b.model_key
  const fasterTime = Math.min(a.elapsed_sec, b.elapsed_sec)
  const slowerTime = Math.max(a.elapsed_sec, b.elapsed_sec)
  const speedupPct = slowerTime > 0
    ? (((slowerTime - fasterTime) / slowerTime) * 100).toFixed(0)
    : '0'

  return (
    <div className="w-full flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-base font-semibold text-slate-200">Comparativa de modelos</h2>
        <span className="text-xs text-blue-400 bg-blue-950/40 border border-blue-800/50 px-2 py-0.5 rounded-full">
          {MODEL_LABELS[a.model_key]} vs {MODEL_LABELS[b.model_key]}
        </span>
      </div>

      {/* Two columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-800/30 border border-slate-700 rounded-2xl p-4">
          <ResultsPanel result={a} label="Modelo A" />
        </div>
        <div className="bg-slate-800/30 border border-slate-700 rounded-2xl p-4">
          <ResultsPanel result={b} label="Modelo B" />
        </div>
      </div>

      {/* Difference summary */}
      <div className="bg-slate-800/50 border border-slate-600 rounded-2xl p-5 flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Resumen comparativo
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Tree count diff */}
          <div className="flex flex-col gap-1 items-center text-center bg-slate-800 rounded-xl p-4 border border-slate-700">
            <span className="text-xs text-slate-500 uppercase tracking-wider">Diferencia árboles</span>
            <span
              className={`text-3xl font-black tabular-nums ${
                diff > 0 ? 'text-green-400' : diff < 0 ? 'text-red-400' : 'text-slate-500'
              }`}
            >
              {diff > 0 ? `+${diff}` : diff}
            </span>
            <span className="text-xs text-slate-500">
              {diff === 0
                ? 'Coinciden exactamente'
                : diff > 0
                ? `${MODEL_LABELS[b.model_key]} detectó más`
                : `${MODEL_LABELS[a.model_key]} detectó más`}
            </span>
          </div>

          {/* Faster model */}
          <div className="flex flex-col gap-1 items-center text-center bg-slate-800 rounded-xl p-4 border border-slate-700">
            <span className="text-xs text-slate-500 uppercase tracking-wider">Modelo más rápido</span>
            <span className="text-xl font-bold text-yellow-400">{fasterModel}</span>
            <span className="text-xs text-slate-500">
              {fasterTime.toFixed(1)}s vs {slowerTime.toFixed(1)}s
            </span>
            <span className="text-xs text-yellow-600">{speedupPct}% más rápido</span>
          </div>

          {/* Tiles with detections diff */}
          <div className="flex flex-col gap-1 items-center text-center bg-slate-800 rounded-xl p-4 border border-slate-700">
            <span className="text-xs text-slate-500 uppercase tracking-wider">Tiles c/ detecciones</span>
            <div className="flex items-center gap-2 text-slate-300">
              <span className="text-xl font-bold text-blue-400">{a.tiles_with_detections}</span>
              <span className="text-slate-600 text-sm">vs</span>
              <span className="text-xl font-bold text-purple-400">{b.tiles_with_detections}</span>
            </div>
            <span className="text-xs text-slate-500">
              {MODEL_LABELS[a.model_key]} / {MODEL_LABELS[b.model_key]}
            </span>
          </div>
        </div>

        {/* Per-tile confidence comparison */}
        <div className="grid grid-cols-2 gap-3 mt-1">
          {[a, b].map((r, i) => {
            const confs = r.detecciones.map((d) => d.confidence)
            const avg =
              confs.length > 0
                ? (confs.reduce((s, v) => s + v, 0) / confs.length)
                : 0
            const max = confs.length > 0 ? Math.max(...confs) : 0
            return (
              <div
                key={i}
                className="bg-slate-900/50 rounded-xl p-3 border border-slate-700 flex flex-col gap-1"
              >
                <span className="text-xs text-slate-500">
                  {i === 0 ? MODEL_LABELS[a.model_key] : MODEL_LABELS[b.model_key]} — confianza
                </span>
                <div className="flex gap-4 text-xs">
                  <span className="text-slate-400">
                    Prom:{' '}
                    <span className="text-slate-200 font-mono">
                      {(avg * 100).toFixed(1)}%
                    </span>
                  </span>
                  <span className="text-slate-400">
                    Máx:{' '}
                    <span className="text-slate-200 font-mono">
                      {(max * 100).toFixed(1)}%
                    </span>
                  </span>
                </div>
                {/* Confidence bar */}
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden mt-1">
                  <div
                    className={`h-full rounded-full ${i === 0 ? 'bg-blue-500' : 'bg-purple-500'}`}
                    style={{ width: `${avg * 100}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
