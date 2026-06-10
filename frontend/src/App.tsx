import { useState } from 'react'
import DropZone from './components/DropZone'
import ModelSelector from './components/ModelSelector'
import ResultsPanel from './components/ResultsPanel'
import TileViewer from './components/TileViewer'
import ComparePanel from './components/ComparePanel'
import { processJob, compareModels } from './api/client'
import {
  UploadResponse,
  ProcessRequest,
  ProcessResponse,
  AppState,
} from './types'

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle')
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [compareResults, setCompareResults] = useState<
    [ProcessResponse, ProcessResponse] | null
  >(null)
  const [error, setError] = useState<string | null>(null)

  function handleUploaded(res: UploadResponse) {
    setUpload(res)
    setAppState('uploaded')
    setResult(null)
    setCompareResults(null)
    setError(null)
  }

  async function handleProcess(req: ProcessRequest) {
    setAppState('processing')
    setError(null)
    setResult(null)
    setCompareResults(null)
    try {
      const res = await processJob(req)
      setResult(res)
      setAppState('done')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido')
      setAppState('error')
    }
  }

  async function handleProcessCompare(req1: ProcessRequest, req2: ProcessRequest) {
    setAppState('processing')
    setError(null)
    setResult(null)
    setCompareResults(null)
    try {
      const res = await compareModels(req1.job_id, [req1.model_key, req2.model_key])
      if (res.results.length >= 2) {
        setCompareResults([res.results[0], res.results[1]])
        setAppState('comparing')
      } else {
        throw new Error('Respuesta de comparación inválida')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido')
      setAppState('error')
    }
  }

  function reset() {
    setAppState('idle')
    setUpload(null)
    setResult(null)
    setCompareResults(null)
    setError(null)
  }

  const isProcessing = appState === 'processing'

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0f172a' }}>
      {/* Header */}
      <header
        className="border-b px-4 py-3 flex items-center justify-between"
        style={{ borderColor: '#1e293b', backgroundColor: '#0f172a' }}
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">🌲</span>
          <div>
            <h1 className="text-base font-bold text-white leading-none">ForestAI</h1>
            <p className="text-xs" style={{ color: '#64748b' }}>
              Detección de árboles — YOLO PoC
            </p>
          </div>
        </div>
        {appState !== 'idle' && (
          <button
            onClick={reset}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors"
            style={{ color: '#94a3b8', backgroundColor: '#1e293b' }}
          >
            ↩ Reiniciar
          </button>
        )}
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8 flex flex-col gap-8">

        {/* Step 1 — Drop zone (always visible until processing) */}
        {appState !== 'processing' && appState !== 'done' && appState !== 'comparing' && (
          <section className="flex flex-col gap-3">
            <SectionLabel step={1} text="Cargar ortomosaico" />
            <DropZone onUploaded={handleUploaded} />
          </section>
        )}

        {/* Step 2 — Model selector (visible once uploaded) */}
        {(appState === 'uploaded' || appState === 'done' || appState === 'comparing') && !isProcessing && (
          <section className="flex flex-col gap-3">
            <SectionLabel step={2} text="Configurar y procesar" />
            {upload && (
              <div
                className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg border"
                style={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#94a3b8' }}
              >
                <span>📄</span>
                <span className="font-medium" style={{ color: '#e2e8f0' }}>
                  {upload.filename}
                </span>
                <span style={{ color: '#475569' }}>·</span>
                <span>job: {upload.job_id}</span>
              </div>
            )}
            <ModelSelector
              jobId={upload?.job_id ?? null}
              onProcess={handleProcess}
              onProcessCompare={handleProcessCompare}
              processing={isProcessing}
            />
          </section>
        )}

        {/* Processing spinner */}
        {isProcessing && (
          <div className="flex flex-col items-center justify-center gap-4 py-24">
            <div
              className="w-16 h-16 border-4 border-t-transparent rounded-full animate-spin"
              style={{ borderColor: '#22c55e', borderTopColor: 'transparent' }}
            />
            <p className="text-lg font-semibold" style={{ color: '#94a3b8' }}>
              Procesando tiles...
            </p>
            <p className="text-sm" style={{ color: '#475569' }}>
              Esto puede tardar unos minutos según el tamaño del ortomosaico
            </p>
          </div>
        )}

        {/* Error */}
        {appState === 'error' && error && (
          <div
            className="flex items-start gap-3 rounded-xl border px-4 py-3"
            style={{ backgroundColor: 'rgba(127,29,29,0.3)', borderColor: '#991b1b' }}
          >
            <span className="text-xl">⚠️</span>
            <div>
              <p className="text-sm font-semibold" style={{ color: '#fca5a5' }}>
                Error durante el procesamiento
              </p>
              <p className="text-xs mt-1" style={{ color: '#f87171' }}>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* Step 3 — Results (single model) */}
        {appState === 'done' && result && (
          <section className="flex flex-col gap-6">
            <SectionLabel step={3} text="Resultados" />
            <ResultsPanel result={result} />
            <TileViewer result={result} />
          </section>
        )}

        {/* Step 3 — Results (compare) */}
        {appState === 'comparing' && compareResults && (
          <section className="flex flex-col gap-6">
            <SectionLabel step={3} text="Comparativa" />
            <ComparePanel results={compareResults} />
            {/* Also show tile viewer for the first result */}
            <div>
              <h3 className="text-sm font-semibold mb-3" style={{ color: '#94a3b8' }}>
                Tiles — {compareResults[0].model_key}
              </h3>
              <TileViewer result={compareResults[0]} />
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-xs" style={{ color: '#334155' }}>
        ForestAI PoC · YOLO Tree Detection · {new Date().getFullYear()}
      </footer>
    </div>
  )
}

function SectionLabel({ step, text }: { step: number; text: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
        style={{ backgroundColor: '#1e3a5f', color: '#3b82f6' }}
      >
        {step}
      </span>
      <span className="text-sm font-semibold" style={{ color: '#94a3b8' }}>
        {text}
      </span>
    </div>
  )
}
