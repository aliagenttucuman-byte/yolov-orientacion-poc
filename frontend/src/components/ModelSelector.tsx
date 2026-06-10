import { useState } from 'react'
import { ModelInfo, ProcessRequest } from '../types'

interface Props {
  jobId: string | null
  onProcess: (req: ProcessRequest) => void
  onProcessCompare: (req1: ProcessRequest, req2: ProcessRequest) => void
  processing: boolean
}

const MODELS: ModelInfo[] = [
  {
    key: 'yolov8n',
    label: 'YOLOv8n',
    description: 'Nano — alta velocidad, ideal para exploración rápida',
    speed: '~0.8s/tile',
  },
  {
    key: 'yolov9c',
    label: 'YOLOv9c',
    description: 'Compact — mejor precisión, velocidad moderada',
    speed: '~1.4s/tile',
  },
  {
    key: 'yolo11n',
    label: 'YOLO11n',
    description: 'Nano v11 — arquitectura actualizada, eficiente',
    speed: '~0.9s/tile',
  },
]

function ModelCard({
  model,
  selected,
  onSelect,
}: {
  model: ModelInfo
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={[
        'flex flex-col gap-1 rounded-xl border-2 p-4 text-left transition-all cursor-pointer w-full',
        selected
          ? 'border-blue-500 bg-blue-950/50 shadow-lg shadow-blue-900/20'
          : 'border-slate-700 bg-slate-800/50 hover:border-slate-500',
      ].join(' ')}
    >
      <div className="flex items-center justify-between">
        <span className={`font-bold text-sm ${selected ? 'text-blue-300' : 'text-slate-200'}`}>
          {model.label}
        </span>
        {selected && (
          <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full">
            ✓ Seleccionado
          </span>
        )}
      </div>
      <span className="text-xs text-slate-400 leading-snug">{model.description}</span>
      <span className="text-xs text-slate-500 mt-1">⚡ {model.speed}</span>
    </button>
  )
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
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
      <div className="flex justify-between text-[10px] text-slate-600">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function ModelSelector({ jobId, onProcess, onProcessCompare, processing }: Props) {
  const [modelA, setModelA] = useState('yolov8n')
  const [modelB, setModelB] = useState('yolov9c')
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.45)
  const [tileSize, setTileSize] = useState(640)
  const [overlap, setOverlap] = useState(0.2)
  const [compareMode, setCompareMode] = useState(false)

  function buildReq(modelKey: string): ProcessRequest {
    return { job_id: jobId!, model_key: modelKey, conf, iou, tile_size: tileSize, overlap }
  }

  function handleProcess() {
    if (!jobId) return
    if (compareMode) {
      onProcessCompare(buildReq(modelA), buildReq(modelB))
    } else {
      onProcess(buildReq(modelA))
    }
  }

  const canProcess = !!jobId && !processing

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col gap-6">

      {/* Compare toggle */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-200">Configuración del modelo</h2>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <span className="text-xs text-slate-400">Modo comparativa</span>
          <div
            onClick={() => setCompareMode((v) => !v)}
            className={[
              'relative w-10 h-5 rounded-full transition-colors',
              compareMode ? 'bg-blue-600' : 'bg-slate-700',
            ].join(' ')}
          >
            <div
              className={[
                'absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform',
                compareMode ? 'translate-x-5' : 'translate-x-0.5',
              ].join(' ')}
            />
          </div>
        </label>
      </div>

      {/* Model selection */}
      {!compareMode ? (
        <div>
          <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Modelo</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {MODELS.map((m) => (
              <ModelCard
                key={m.key}
                model={m}
                selected={modelA === m.key}
                onSelect={() => setModelA(m.key)}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Modelo A</p>
            <div className="flex flex-col gap-2">
              {MODELS.map((m) => (
                <ModelCard
                  key={m.key}
                  model={m}
                  selected={modelA === m.key}
                  onSelect={() => setModelA(m.key)}
                />
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Modelo B</p>
            <div className="flex flex-col gap-2">
              {MODELS.map((m) => (
                <ModelCard
                  key={m.key}
                  model={m}
                  selected={modelB === m.key}
                  onSelect={() => setModelB(m.key)}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Parameters */}
      <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex flex-col gap-4">
        <p className="text-xs text-slate-500 uppercase tracking-wider">Parámetros de detección</p>
        <ParamSlider
          label="Confianza mínima (conf)"
          value={conf}
          min={0.1}
          max={0.9}
          step={0.01}
          onChange={setConf}
        />
        <ParamSlider
          label="IoU threshold"
          value={iou}
          min={0.1}
          max={0.9}
          step={0.01}
          onChange={setIou}
        />
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-slate-400">Tile size (px)</span>
            <select
              value={tileSize}
              onChange={(e) => setTileSize(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200"
            >
              {[320, 640, 1024, 1280].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label="Overlap"
            value={overlap}
            min={0.0}
            max={0.5}
            step={0.05}
            onChange={setOverlap}
          />
        </div>
      </div>

      {/* Process button */}
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
        {processing
          ? '⏳ Procesando...'
          : compareMode
          ? `🔬 Comparar ${modelA} vs ${modelB}`
          : `🚀 Procesar con ${MODELS.find((m) => m.key === modelA)?.label}`}
      </button>

      {!jobId && (
        <p className="text-center text-xs text-slate-600">
          Subí un ortomosaico primero para habilitar el procesamiento
        </p>
      )}
    </div>
  )
}
