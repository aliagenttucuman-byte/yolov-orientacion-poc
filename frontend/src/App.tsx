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
import SpeciesPanel from './components/SpeciesPanel'

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
  const showHero = appState === 'idle'

  return (
    <div
      style={{
        minHeight: '100vh',
        background:
          'radial-gradient(ellipse at top, #0f2027 0%, #0a1520 50%, #050a14 100%)',
        fontFamily:
          "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: '#e2e8f0',
      }}
    >
      {/* NAVBAR */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 50,
          backdropFilter: 'blur(12px)',
          background: 'rgba(10, 21, 32, 0.85)',
          borderBottom: '1px solid rgba(34, 197, 94, 0.15)',
          padding: '1rem 2rem',
        }}
      >
        <div
          style={{
            maxWidth: 1280,
            margin: '0 auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background:
                  'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 22,
                boxShadow: '0 8px 24px rgba(34, 197, 94, 0.25)',
              }}
            >
              🌲
            </div>
            <div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  color: '#fff',
                  letterSpacing: '-0.01em',
                  lineHeight: 1,
                }}
              >
                ForestAI
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: '#64748b',
                  marginTop: 3,
                  fontWeight: 500,
                  letterSpacing: '0.02em',
                }}
              >
                Tree Detection · YOLO + Drone Imagery
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 12,
                color: '#94a3b8',
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#22c55e',
                  boxShadow: '0 0 12px #22c55e',
                }}
              />
              Modelo activo
            </div>
            {appState !== 'idle' && (
              <button
                onClick={reset}
                style={{
                  fontSize: 12,
                  padding: '0.5rem 1rem',
                  borderRadius: 8,
                  background: 'rgba(30, 41, 59, 0.6)',
                  border: '1px solid #334155',
                  color: '#cbd5e1',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                ↩  Nueva detección
              </button>
            )}
          </div>
        </div>
      </header>

      {/* HERO — visible solo en idle */}
      {showHero && (
        <section
          style={{
            maxWidth: 900,
            margin: '0 auto',
            padding: '4rem 2rem 2rem',
            textAlign: 'center',
          }}
        >
          <h1
            style={{
              fontSize: 'clamp(2rem, 5vw, 3.25rem)',
              fontWeight: 800,
              color: '#fff',
              lineHeight: 1.1,
              letterSpacing: '-0.025em',
              marginBottom: '1rem',
            }}
          >
            PoC para detección de
            <br />
            <span
              style={{
                background:
                  'linear-gradient(135deg, #22c55e 0%, #4ade80 50%, #86efac 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              árboles
            </span>
          </h1>
        </section>
      )}

      {/* MAIN CONTENT — centrado */}
      <main
        style={{
          maxWidth: 1100,
          margin: '0 auto',
          padding: showHero ? '0 2rem 4rem' : '3rem 2rem 4rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '2rem',
          alignItems: 'center',
        }}
      >
        {/* STEP 1 — DropZone */}
        {appState !== 'processing' && appState !== 'done' && appState !== 'comparing' && (
          <section
            style={{
              width: '100%',
              maxWidth: 760,
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              alignItems: 'center',
            }}
          >
            <SectionLabel step={1} text="Cargar ortomosaico aéreo" />
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              <DropZone onUploaded={handleUploaded} />
            </div>
          </section>
        )}

        {/* STEP 2 — Model selector */}
        {(appState === 'uploaded' || appState === 'done' || appState === 'comparing') && !isProcessing && (
          <section
            style={{
              width: '100%',
              maxWidth: 880,
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              alignItems: 'center',
            }}
          >
            <SectionLabel step={2} text="Seleccionar modelo y procesar" />
            {upload && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  fontSize: 13,
                  padding: '0.875rem 1.25rem',
                  borderRadius: 10,
                  background:
                    'linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.03) 100%)',
                  border: '1px solid rgba(34, 197, 94, 0.25)',
                  color: '#94a3b8',
                }}
              >
                <span style={{ fontSize: 20 }}>📄</span>
                <span style={{ color: '#e2e8f0', fontWeight: 600 }}>
                  {upload.filename}
                </span>
                <span style={{ color: '#475569' }}>·</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  Job ID: {upload.job_id}
                </span>
              </div>
            )}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              <ModelSelector
                jobId={upload?.job_id ?? null}
                onProcess={handleProcess}
                onProcessCompare={handleProcessCompare}
                processing={isProcessing}
              />
            </div>
          </section>
        )}

        {/* PROCESSING */}
        {isProcessing && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '1.5rem',
              padding: '6rem 0',
              width: '100%',
            }}
          >
            <div
              style={{
                position: 'relative',
                width: 80,
                height: 80,
              }}
            >
              <div
                style={{
                  width: 80,
                  height: 80,
                  border: '4px solid rgba(34, 197, 94, 0.15)',
                  borderTopColor: '#22c55e',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 28,
                }}
              >
                🌲
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  color: '#e2e8f0',
                  marginBottom: 8,
                }}
              >
                Procesando ortomosaico
              </p>
              <p style={{ fontSize: 14, color: '#64748b' }}>
                Generando tiles, ejecutando inferencia YOLO y consolidando detecciones...
              </p>
            </div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* ERROR */}
        {appState === 'error' && error && (
          <div
            style={{
              width: '100%',
              maxWidth: 760,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 12,
              padding: '1rem 1.25rem',
              borderRadius: 12,
              background: 'rgba(127, 29, 29, 0.2)',
              border: '1px solid rgba(220, 38, 38, 0.4)',
            }}
          >
            <span style={{ fontSize: 24 }}>⚠️</span>
            <div>
              <p style={{ fontSize: 14, fontWeight: 600, color: '#fca5a5' }}>
                Error durante el procesamiento
              </p>
              <p style={{ fontSize: 13, marginTop: 4, color: '#f87171' }}>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* RESULTS */}
        {appState === 'done' && result && (
          <section
            style={{
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
              alignItems: 'center',
            }}
          >
            <SectionLabel step={3} text="Resultados de la detección" />
            <div style={{ width: '100%' }}>
              <ResultsPanel result={result} />
            </div>
            <div style={{ width: '100%' }}>
              <SpeciesPanel jobId={result.job_id} />
            </div>
            <div style={{ width: '100%' }}>
              <TileViewer result={result} />
            </div>
          </section>
        )}

        {/* COMPARE */}
        {appState === 'comparing' && compareResults && (
          <section
            style={{
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
              alignItems: 'center',
            }}
          >
            <SectionLabel step={3} text="Comparativa entre modelos" />
            <div style={{ width: '100%' }}>
              <ComparePanel results={compareResults} />
            </div>
            <div style={{ width: '100%' }}>
              <h3
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: '#94a3b8',
                  marginBottom: 12,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                Tiles · {compareResults[0].model_key}
              </h3>
              <TileViewer result={compareResults[0]} />
            </div>
          </section>
        )}
      </main>

      {/* FOOTER */}
      <footer
        style={{
          textAlign: 'center',
          padding: '2rem 1rem',
          fontSize: 12,
          color: '#475569',
          borderTop: '1px solid rgba(30, 41, 59, 0.5)',
          marginTop: 'auto',
        }}
      >
        <div style={{ marginBottom: 6, color: '#64748b', fontWeight: 500 }}>
          ForestAI · Tree Detection from Drone Imagery
        </div>
        <div>
          Powered by YOLO v11 fine-tuned · AlegentAI {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  )
}

function SectionLabel({ step, text }: { step: number; text: string }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 4,
      }}
    >
      <span
        style={{
          width: 32,
          height: 32,
          borderRadius: '50%',
          background:
            'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 14,
          fontWeight: 700,
          color: '#fff',
          boxShadow: '0 4px 12px rgba(34, 197, 94, 0.3)',
        }}
      >
        {step}
      </span>
      <span
        style={{
          fontSize: 15,
          fontWeight: 600,
          color: '#e2e8f0',
          letterSpacing: '-0.01em',
        }}
      >
        {text}
      </span>
    </div>
  )
}
