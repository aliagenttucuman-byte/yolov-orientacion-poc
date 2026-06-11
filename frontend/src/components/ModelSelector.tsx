import { useState } from 'react'
import { ProcessRequest } from '../types'

interface Props {
  jobId: string | null
  onProcess: (req: ProcessRequest) => void
  onProcessCompare: (req1: ProcessRequest, req2: ProcessRequest) => void
  processing: boolean
}

const MODELS = [
  { key: 'yolo11n_forestai', label: 'YOLO11n ForestAI (fine-tuned)' },
  { key: 'yolo11n',          label: 'YOLO11n base' },
  { key: 'yolov8n',          label: 'YOLOv8n base' },
  { key: 'exg',              label: 'ExG — Excess Green (sin ML)' },
]

// Defaults calibrados para demo ReforestLatam (9 de Julio, 2026-06-11)
// tile=1024px → captura copas grandes que a 640px se pierden
// conf=0.55 → ~300 árboles en zona urbana/periurbana — número creíble
// centroid=60px → fusiona detecciones duplicadas de la misma copa
const DEFAULT_NMS_IOU      = 0.40
const DEFAULT_IOU          = 0.45
const DEFAULT_OVERLAP      = 0.20

const MODEL_DEFAULTS: Record<string, { centroid: number; conf: number; tile_size: number }> = {
  yolo11n_forestai: { centroid: 90,  conf: 0.65, tile_size: 640  },
  yolo11n:          { centroid: 60,  conf: 0.25, tile_size: 640  },
  yolov8n:          { centroid: 60,  conf: 0.25, tile_size: 640  },
  exg:              { centroid: 90,  conf: 0.50, tile_size: 1024 },
}

function ParamSlider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span className="font-mono text-slate-200">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-slate-700 cursor-pointer"
      />
      {hint && (
        <p className="text-[10px] text-slate-600 italic">{hint}</p>
      )}
    </div>
  )
}

export default function ModelSelector({ jobId, onProcess, processing }: Props) {
  const [conf, setConf]           = useState(MODEL_DEFAULTS[MODELS[0].key].conf)
  const [nmsIou, setNmsIou]       = useState(DEFAULT_NMS_IOU)
  const [centroid, setCentroid]   = useState(MODEL_DEFAULTS[MODELS[0].key].centroid)
  const [tileSize, setTileSize]   = useState(MODEL_DEFAULTS[MODELS[0].key].tile_size)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [modelKey, setModelKey]   = useState(MODELS[0].key)

  function handleModelChange(key: string) {
    setModelKey(key)
    const d = MODEL_DEFAULTS[key]
    if (d) { setConf(d.conf); setCentroid(d.centroid); setTileSize(d.tile_size) }
  }

  function handleProcess() {
    if (!jobId) return
    onProcess({
      job_id:           jobId,
      model_key:        modelKey,
      conf,
      iou:              DEFAULT_IOU,
      nms_iou:          nmsIou,
      centroid_dist_px: centroid,
      tile_size:        tileSize,
      overlap:          Math.round(DEFAULT_OVERLAP * tileSize),
    })
  }

  const canProcess = !!jobId && !processing

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col gap-6">

      {/* Selector de modelo */}
      <div
        className="flex flex-col gap-2 rounded-xl border px-4 py-3"
        style={{ backgroundColor: '#1e293b', borderColor: '#22c55e44' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-xl">🌲</span>
          <span className="text-sm font-bold text-green-400">Detector</span>
        </div>
        <select
          value={modelKey}
          onChange={(e) => handleModelChange(e.target.value)}
          className="w-full rounded-lg px-3 py-2 text-sm font-mono text-slate-200"
          style={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
        >
          {MODELS.map(m => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
        {modelKey === 'exg' && (
          <p className="text-[10px] text-slate-500 italic">
            No usa ML — detecta vegetación por índice ExG (Excess Green). Muy rápido, sin falsos positivos de vehículos.
          </p>
        )}
      </div>

      {/* Parámetro principal */}
      <div
        className="border rounded-xl p-4 flex flex-col gap-4"
        style={{ backgroundColor: '#1e293b44', borderColor: '#334155' }}
      >
        <p className="text-xs text-slate-500 uppercase tracking-wider">Sensibilidad de detección</p>

        <ParamSlider
          label="Confianza mínima"
          hint="Más alto = menos árboles pero más precisos. Demo calibrada en 0.65"
          value={conf}
          min={0.45}
          max={0.85}
          step={0.05}
          onChange={setConf}
        />

        {/* Toggle avanzado */}
        <button
          className="text-[11px] text-slate-500 hover:text-slate-300 text-left transition-colors"
          onClick={() => setShowAdvanced(v => !v)}
        >
          {showAdvanced ? '▾ Ocultar parámetros avanzados' : '▸ Parámetros avanzados (fusión de copas)'}
        </button>

        {showAdvanced && (
          <div className="flex flex-col gap-4 border-t pt-3" style={{ borderColor: '#334155' }}>
            <ParamSlider
              label="NMS global entre tiles"
              hint="Elimina detecciones duplicadas en bordes de tiles. Default 0.40"
              value={nmsIou}
              min={0.10}
              max={0.80}
              step={0.05}
              onChange={setNmsIou}
            />
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Fusión de copa (centroide)</span>
                <span className="font-mono text-slate-200">{centroid}px</span>
              </div>
              <input
                type="range"
                min={20}
                max={120}
                step={10}
                value={centroid}
                onChange={(e) => setCentroid(parseInt(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none bg-slate-700 cursor-pointer"
              />
              <p className="text-[10px] text-slate-600 italic">
                Fusiona detecciones de la misma copa a menos de {centroid}px (~{(centroid * 0.06).toFixed(1)}m). Subir para copas grandes (quebrachos adultos).
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Botón */}
      <button
        onClick={handleProcess}
        disabled={!canProcess}
        className={[
          'w-full py-3.5 rounded-xl font-semibold text-sm transition-all',
          canProcess
            ? 'bg-green-600 hover:bg-green-500 text-white cursor-pointer active:scale-95 shadow-lg shadow-green-900/30'
            : 'bg-slate-700 text-slate-500 cursor-not-allowed',
        ].join(' ')}
      >
        {processing ? '⏳ Procesando...' : '🚀 Detectar Árboles'}
      </button>

      {!jobId && (
        <p className="text-center text-xs text-slate-600">
          Subí un ortomosaico primero para habilitar el procesamiento
        </p>
      )}
    </div>
  )
}
