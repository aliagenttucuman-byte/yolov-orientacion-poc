import { ProcessResponse } from '../types'

interface Props {
  result: ProcessResponse
  label?: string
}

const MODEL_LABELS: Record<string, string> = {
  yolov8n: 'YOLOv8n',
  yolov9c: 'YOLOv9c',
  yolo11n: 'YOLO11n',
}

function StatCard({
  title,
  value,
  sub,
  highlight,
}: {
  title: string
  value: string | number
  sub?: string
  highlight?: boolean
}) {
  return (
    <div
      className={[
        'flex flex-col gap-1 rounded-xl p-4 border',
        highlight
          ? 'bg-green-950/30 border-green-800/50'
          : 'bg-slate-800/50 border-slate-700',
      ].join(' ')}
    >
      <span className="text-xs text-slate-500 uppercase tracking-wider">{title}</span>
      <span
        className={`text-2xl font-bold tabular-nums ${
          highlight ? 'text-green-400' : 'text-slate-200'
        }`}
      >
        {value}
      </span>
      {sub && <span className="text-xs text-slate-500">{sub}</span>}
    </div>
  )
}

export default function ResultsPanel({ result, label }: Props) {
  const hasDetections = result.tree_count > 0
  const modelLabel = MODEL_LABELS[result.model_key] ?? result.model_key
  const detectionRate =
    result.tiles_processed > 0
      ? ((result.tiles_with_detections / result.tiles_processed) * 100).toFixed(1)
      : '0'

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-slate-200">
            {label ?? 'Resultados'}
          </span>
          <span className="text-xs font-medium bg-blue-900/60 text-blue-300 border border-blue-700/50 px-2 py-0.5 rounded-full">
            {modelLabel}
          </span>
        </div>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full ${
            hasDetections
              ? 'bg-green-900/50 text-green-300 border border-green-700/50'
              : 'bg-slate-800 text-slate-500 border border-slate-700'
          }`}
        >
          {hasDetections ? '✓ Detecciones encontradas' : '— Sin detecciones'}
        </span>
      </div>

      {/* Main count */}
      <div
        className={[
          'rounded-2xl border p-6 flex flex-col items-center gap-1',
          hasDetections
            ? 'bg-green-950/20 border-green-800/40'
            : 'bg-slate-800/30 border-slate-700',
        ].join(' ')}
      >
        <span className="text-xs text-slate-500 uppercase tracking-widest">Árboles detectados</span>
        <span
          className={`text-6xl font-black tabular-nums ${
            hasDetections ? 'text-green-400' : 'text-slate-600'
          }`}
        >
          {result.tree_count.toLocaleString()}
        </span>
        <span className="text-xs text-slate-500">
          en {result.tiles_with_detections} de {result.tiles_processed} tiles
        </span>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          title="Tiles procesados"
          value={result.tiles_processed.toLocaleString()}
        />
        <StatCard
          title="Con detecciones"
          value={result.tiles_with_detections.toLocaleString()}
          sub={`${detectionRate}% del total`}
          highlight={hasDetections}
        />
        <StatCard
          title="Tiempo total"
          value={`${result.elapsed_sec.toFixed(1)}s`}
        />
        <StatCard
          title="Tiles / seg"
          value={result.tiles_per_sec.toFixed(1)}
        />
      </div>
    </div>
  )
}
