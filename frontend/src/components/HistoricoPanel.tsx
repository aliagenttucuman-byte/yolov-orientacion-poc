import { useState, useEffect, useRef } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────
interface HistoricoBBox {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
  nombre_zona?: string
}

interface YearlyLoss {
  year: number
  loss_ha: number
}

interface HistoricoResponse {
  bbox: HistoricoBBox
  area_total_ha: number
  cobertura_2000_ha: number
  cobertura_2000_pct: number
  perdida_total_ha: number
  perdida_total_pct: number
  perdida_por_year: YearlyLoss[]
  year_pico_perdida: number
  perdida_year_pico_ha: number
  tasa_anual_promedio_ha: number
  timelapse_url: string
  timelapse_embed_url: string
  fuente: string
}

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

interface PresetInfo {
  nombre: string
  bbox: HistoricoBBox
}

const BASE = '/api/v1'

// ── Helpers ────────────────────────────────────────────────────────────────
function formatNum(n: number, dec = 0): string {
  return n.toLocaleString('es-AR', {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  })
}

// ── Componente principal ───────────────────────────────────────────────────
export default function HistoricoPanel() {
  const [presets, setPresets] = useState<Record<string, PresetInfo>>({})
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [analisis, setAnalisis] = useState<HistoricoResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Chat
  const [chat, setChat] = useState<ChatMsg[]>([])
  const [pregunta, setPregunta] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatBottomRef = useRef<HTMLDivElement>(null)

  // BBox manual
  const [bboxManual, setBboxManual] = useState({
    lat_min: -27.0,
    lat_max: -26.5,
    lon_min: -65.5,
    lon_max: -65.0,
    nombre_zona: 'Mi predio',
  })
  const [showManual, setShowManual] = useState(false)

  // Cargar presets al inicio
  useEffect(() => {
    fetch(`${BASE}/historico/presets`)
      .then((r) => r.json())
      .then((d) => setPresets(d))
      .catch((e) => setError(`Error cargando presets: ${e.message}`))
  }, [])

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat])

  async function cargarPreset(key: string) {
    setLoading(true)
    setError(null)
    setChat([])
    try {
      const r = await fetch(`${BASE}/historico/preset/${key}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const d = await r.json()
      setAnalisis(d)
      setSelectedPreset(key)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error')
    } finally {
      setLoading(false)
    }
  }

  async function analizarManual() {
    setLoading(true)
    setError(null)
    setChat([])
    try {
      const r = await fetch(`${BASE}/historico/analizar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bboxManual),
      })
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail ?? `HTTP ${r.status}`)
      }
      const d = await r.json()
      setAnalisis(d)
      setSelectedPreset(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error')
    } finally {
      setLoading(false)
    }
  }

  async function enviarPregunta() {
    if (!pregunta.trim() || !analisis || chatLoading) return
    const q = pregunta.trim()
    setPregunta('')
    const newChat: ChatMsg[] = [...chat, { role: 'user', content: q }]
    setChat(newChat)
    setChatLoading(true)
    try {
      const r = await fetch(`${BASE}/historico/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pregunta: q,
          contexto: analisis,
          historial: newChat.slice(0, -1).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      })
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail ?? `HTTP ${r.status}`)
      }
      const d = await r.json()
      setChat([...newChat, { role: 'assistant', content: d.respuesta }])
    } catch (e) {
      setChat([
        ...newChat,
        {
          role: 'assistant',
          content: `⚠️ Error: ${e instanceof Error ? e.message : 'desconocido'}`,
        },
      ])
    } finally {
      setChatLoading(false)
    }
  }

  const sugerencias = [
    '¿Qué pasó entre 2008 y 2012?',
    '¿Cuáles fueron las principales causas?',
    '¿Sirve para bonos de carbono REDD+?',
    'Comparame contra el promedio nacional',
    'Generame un párrafo para reporte MRV',
  ]

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        width: '100%',
        maxWidth: 1280,
        margin: '0 auto',
        padding: '0 2rem 4rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
      }}
    >
      {/* HEADER */}
      <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
        <h2
          style={{
            fontSize: 'clamp(1.5rem, 3vw, 2.25rem)',
            fontWeight: 800,
            color: '#fff',
            margin: 0,
            letterSpacing: '-0.02em',
          }}
        >
          Análisis Histórico Satelital
        </h2>
        <p
          style={{
            color: '#94a3b8',
            fontSize: 14,
            marginTop: 8,
            maxWidth: 720,
            margin: '0.5rem auto 0',
          }}
        >
          40 años de cobertura forestal (1984-2023). Datos Hansen Global Forest
          Change · Google Earth Timelapse · Chat IA generativa.
        </p>
      </div>

      {/* PRESETS */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(34, 197, 94, 0.15)',
          borderRadius: 16,
          padding: '1.5rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem',
          }}
        >
          <h3
            style={{
              fontSize: 14,
              color: '#94a3b8',
              margin: 0,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            Zonas Argentinas Predefinidas
          </h3>
          <button
            onClick={() => setShowManual(!showManual)}
            style={{
              fontSize: 12,
              padding: '0.5rem 0.875rem',
              borderRadius: 8,
              background: showManual
                ? 'rgba(34, 197, 94, 0.15)'
                : 'rgba(30, 41, 59, 0.6)',
              border: `1px solid ${showManual ? '#22c55e' : '#334155'}`,
              color: showManual ? '#86efac' : '#cbd5e1',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            {showManual ? '✓ ' : ''}BBox manual
          </button>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '0.75rem',
          }}
        >
          {Object.entries(presets).map(([key, p]) => (
            <button
              key={key}
              onClick={() => cargarPreset(key)}
              disabled={loading}
              style={{
                padding: '0.875rem',
                borderRadius: 10,
                background:
                  selectedPreset === key
                    ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.08) 100%)'
                    : 'rgba(30, 41, 59, 0.5)',
                border: `1px solid ${
                  selectedPreset === key ? '#22c55e' : '#334155'
                }`,
                color: selectedPreset === key ? '#86efac' : '#e2e8f0',
                cursor: loading ? 'wait' : 'pointer',
                textAlign: 'left',
                fontSize: 13,
                fontWeight: 600,
                transition: 'all 0.2s',
                opacity: loading ? 0.5 : 1,
              }}
            >
              <div style={{ fontSize: 14, marginBottom: 4 }}>{p.nombre}</div>
              <div style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
                {p.bbox.lat_min.toFixed(1)}, {p.bbox.lon_min.toFixed(1)}
              </div>
            </button>
          ))}
        </div>

        {/* MANUAL BBOX */}
        {showManual && (
          <div
            style={{
              marginTop: '1.25rem',
              padding: '1rem',
              background: 'rgba(0,0,0,0.2)',
              borderRadius: 10,
              border: '1px dashed #334155',
            }}
          >
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
              Coordenadas WGS84 (lat, lon en grados decimales)
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '0.5rem',
              }}
            >
              {(['lat_min', 'lat_max', 'lon_min', 'lon_max'] as const).map(
                (k) => (
                  <label key={k} style={{ fontSize: 11, color: '#cbd5e1' }}>
                    {k}
                    <input
                      type="number"
                      step="0.01"
                      value={bboxManual[k]}
                      onChange={(e) =>
                        setBboxManual({
                          ...bboxManual,
                          [k]: parseFloat(e.target.value),
                        })
                      }
                      style={{
                        width: '100%',
                        padding: '0.5rem',
                        background: 'rgba(15,23,42,0.8)',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        color: '#fff',
                        fontSize: 12,
                        fontFamily: 'monospace',
                        marginTop: 4,
                      }}
                    />
                  </label>
                )
              )}
            </div>
            <input
              type="text"
              placeholder="Nombre de la zona"
              value={bboxManual.nombre_zona}
              onChange={(e) =>
                setBboxManual({ ...bboxManual, nombre_zona: e.target.value })
              }
              style={{
                width: '100%',
                padding: '0.5rem',
                background: 'rgba(15,23,42,0.8)',
                border: '1px solid #334155',
                borderRadius: 6,
                color: '#fff',
                fontSize: 12,
                marginTop: 8,
              }}
            />
            <button
              onClick={analizarManual}
              disabled={loading}
              style={{
                marginTop: 12,
                padding: '0.625rem 1.25rem',
                borderRadius: 8,
                background: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
                color: '#fff',
                border: 'none',
                cursor: loading ? 'wait' : 'pointer',
                fontWeight: 600,
                fontSize: 13,
              }}
            >
              {loading ? '⏳ Analizando...' : '▶  Analizar este BBox'}
            </button>
          </div>
        )}
      </div>

      {/* ERROR */}
      {error && (
        <div
          style={{
            padding: '1rem',
            borderRadius: 10,
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            color: '#fca5a5',
            fontSize: 13,
          }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* RESULTADOS */}
      {analisis && (
        <>
          {/* METRICAS */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '0.75rem',
            }}
          >
            <MetricCard
              label="Área analizada"
              value={`${formatNum(analisis.area_total_ha)} ha`}
              tone="neutral"
            />
            <MetricCard
              label="Cobertura año 2000"
              value={`${formatNum(analisis.cobertura_2000_ha)} ha`}
              sub={`${analisis.cobertura_2000_pct}% del área`}
              tone="ok"
            />
            <MetricCard
              label="Pérdida total 2001-2023"
              value={`${formatNum(analisis.perdida_total_ha)} ha`}
              sub={`${analisis.perdida_total_pct}% perdido`}
              tone="bad"
            />
            <MetricCard
              label="Año pico"
              value={`${analisis.year_pico_perdida}`}
              sub={`${formatNum(analisis.perdida_year_pico_ha)} ha`}
              tone="warn"
            />
            <MetricCard
              label="Tasa anual promedio"
              value={`${formatNum(analisis.tasa_anual_promedio_ha)} ha/año`}
              tone="neutral"
            />
          </div>

          {/* TIMELAPSE + CHAT side by side */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)',
              gap: '1rem',
              alignItems: 'stretch',
            }}
          >
            {/* TIMELAPSE EMBED */}
            <div
              style={{
                background: '#000',
                borderRadius: 12,
                overflow: 'hidden',
                border: '1px solid #334155',
                minHeight: 480,
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <div
                style={{
                  padding: '0.625rem 1rem',
                  background: 'rgba(15, 23, 42, 0.95)',
                  borderBottom: '1px solid #334155',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: 12,
                }}
              >
                <span style={{ color: '#cbd5e1', fontWeight: 600 }}>
                  🛰️  Google Earth Timelapse · 1984-2022
                </span>
                <a
                  href={analisis.timelapse_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: '#22c55e', fontSize: 11, textDecoration: 'none' }}
                >
                  Abrir en Earth ↗
                </a>
              </div>
              <iframe
                src={analisis.timelapse_embed_url}
                title="Google Earth Timelapse"
                style={{
                  flex: 1,
                  width: '100%',
                  border: 'none',
                  minHeight: 440,
                }}
                allow="fullscreen"
              />
            </div>

            {/* CHAT IA */}
            <div
              style={{
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(34, 197, 94, 0.15)',
                borderRadius: 12,
                display: 'flex',
                flexDirection: 'column',
                minHeight: 480,
              }}
            >
              <div
                style={{
                  padding: '0.875rem 1rem',
                  borderBottom: '1px solid #334155',
                  fontSize: 13,
                  color: '#cbd5e1',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#22c55e',
                    boxShadow: '0 0 8px #22c55e',
                  }}
                />
                Chat con los datos · Llama 3.3 70B
              </div>

              {/* MENSAJES */}
              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                  fontSize: 13,
                  minHeight: 0,
                  maxHeight: 360,
                }}
              >
                {chat.length === 0 && (
                  <div style={{ color: '#64748b', textAlign: 'center', padding: '2rem 1rem' }}>
                    <div style={{ fontSize: 32, marginBottom: 8 }}>💬</div>
                    <div style={{ fontSize: 12 }}>
                      Preguntale a la IA sobre estos datos.
                    </div>
                    <div
                      style={{
                        marginTop: '1rem',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 6,
                      }}
                    >
                      {sugerencias.map((s) => (
                        <button
                          key={s}
                          onClick={() => setPregunta(s)}
                          style={{
                            fontSize: 11,
                            padding: '0.5rem 0.75rem',
                            background: 'rgba(30, 41, 59, 0.6)',
                            border: '1px solid #334155',
                            borderRadius: 8,
                            color: '#94a3b8',
                            cursor: 'pointer',
                            textAlign: 'left',
                          }}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {chat.map((m, i) => (
                  <div
                    key={i}
                    style={{
                      alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                      maxWidth: '92%',
                      padding: '0.625rem 0.875rem',
                      borderRadius: 10,
                      background:
                        m.role === 'user'
                          ? 'rgba(34, 197, 94, 0.15)'
                          : 'rgba(30, 41, 59, 0.7)',
                      border: `1px solid ${
                        m.role === 'user'
                          ? 'rgba(34, 197, 94, 0.3)'
                          : '#334155'
                      }`,
                      color: m.role === 'user' ? '#d1fae5' : '#e2e8f0',
                      whiteSpace: 'pre-wrap',
                      lineHeight: 1.5,
                    }}
                  >
                    {m.content}
                  </div>
                ))}
                {chatLoading && (
                  <div style={{ color: '#64748b', fontSize: 12, fontStyle: 'italic' }}>
                    🤖 Pensando...
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* INPUT */}
              <div
                style={{
                  padding: '0.75rem',
                  borderTop: '1px solid #334155',
                  display: 'flex',
                  gap: 6,
                }}
              >
                <input
                  type="text"
                  placeholder="Preguntá..."
                  value={pregunta}
                  onChange={(e) => setPregunta(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && enviarPregunta()}
                  disabled={chatLoading}
                  style={{
                    flex: 1,
                    padding: '0.625rem 0.875rem',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid #334155',
                    borderRadius: 8,
                    color: '#fff',
                    fontSize: 13,
                  }}
                />
                <button
                  onClick={enviarPregunta}
                  disabled={chatLoading || !pregunta.trim()}
                  style={{
                    padding: '0 1rem',
                    borderRadius: 8,
                    background: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
                    color: '#fff',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: 13,
                    cursor:
                      chatLoading || !pregunta.trim() ? 'not-allowed' : 'pointer',
                    opacity: chatLoading || !pregunta.trim() ? 0.5 : 1,
                  }}
                >
                  ▶
                </button>
              </div>
            </div>
          </div>

          {/* TABLA AÑO POR AÑO */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(34, 197, 94, 0.15)',
              borderRadius: 12,
              padding: '1rem 1.25rem',
            }}
          >
            <h3
              style={{
                fontSize: 12,
                color: '#94a3b8',
                margin: '0 0 0.75rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}
            >
              Pérdida año por año
            </h3>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(85px, 1fr))',
                gap: 6,
              }}
            >
              {analisis.perdida_por_year.map((y) => {
                const isPico = y.year === analisis.year_pico_perdida
                const maxVal = Math.max(
                  ...analisis.perdida_por_year.map((x) => x.loss_ha)
                )
                const intensity = y.loss_ha / maxVal
                return (
                  <div
                    key={y.year}
                    style={{
                      padding: '0.5rem',
                      borderRadius: 6,
                      background: isPico
                        ? 'rgba(239, 68, 68, 0.18)'
                        : `rgba(34, 197, 94, ${intensity * 0.18 + 0.04})`,
                      border: isPico
                        ? '1px solid rgba(239, 68, 68, 0.4)'
                        : '1px solid rgba(34, 197, 94, 0.1)',
                      textAlign: 'center',
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: '#94a3b8',
                        fontFamily: 'monospace',
                      }}
                    >
                      {y.year}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        color: isPico ? '#fca5a5' : '#e2e8f0',
                        fontWeight: 600,
                        fontFamily: 'monospace',
                      }}
                    >
                      {formatNum(y.loss_ha / 1000, 1)}k
                    </div>
                  </div>
                )
              })}
            </div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: '0.75rem' }}>
              Valores en hectáreas (k = miles). · {analisis.fuente}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ── MetricCard ─────────────────────────────────────────────────────────────
function MetricCard({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: string
  sub?: string
  tone: 'ok' | 'bad' | 'warn' | 'neutral'
}) {
  const colors: Record<string, { bg: string; border: string; text: string }> = {
    ok: {
      bg: 'rgba(34, 197, 94, 0.08)',
      border: 'rgba(34, 197, 94, 0.3)',
      text: '#86efac',
    },
    bad: {
      bg: 'rgba(239, 68, 68, 0.08)',
      border: 'rgba(239, 68, 68, 0.3)',
      text: '#fca5a5',
    },
    warn: {
      bg: 'rgba(245, 158, 11, 0.08)',
      border: 'rgba(245, 158, 11, 0.3)',
      text: '#fcd34d',
    },
    neutral: {
      bg: 'rgba(30, 41, 59, 0.5)',
      border: '#334155',
      text: '#e2e8f0',
    },
  }
  const c = colors[tone]
  return (
    <div
      style={{
        padding: '0.875rem 1rem',
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 10,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: '#94a3b8',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          marginBottom: 4,
          fontWeight: 600,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 18,
          fontWeight: 700,
          color: c.text,
          fontFamily: 'monospace',
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{sub}</div>
      )}
    </div>
  )
}
