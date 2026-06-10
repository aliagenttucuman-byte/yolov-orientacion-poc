import { useEffect, useRef, useState } from 'react'
import { DetectionItem, ProcessResponse } from '../types'
import { tileUrl } from '../api/client'

interface Props {
  result: ProcessResponse
}

interface TileGroup {
  filename: string
  detections: DetectionItem[]
}

function groupByTile(detecciones: DetectionItem[]): TileGroup[] {
  const map = new Map<string, DetectionItem[]>()
  for (const d of detecciones) {
    const arr = map.get(d.tile_filename) ?? []
    arr.push(d)
    map.set(d.tile_filename, arr)
  }
  return Array.from(map.entries())
    .map(([filename, detections]) => ({ filename, detections }))
    .slice(0, 12)
}

interface TileCanvasProps {
  jobId: string
  tile: TileGroup
  onClick: () => void
}

function TileCanvas({ jobId, tile, onClick }: TileCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = tileUrl(jobId, tile.filename)

    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      ctx.drawImage(img, 0, 0)
      drawBoxes(ctx, tile.detections, img.naturalWidth, img.naturalHeight)
      setLoaded(true)
    }
    img.onerror = () => setError(true)
  }, [jobId, tile])

  return (
    <div
      className="relative bg-slate-800 rounded-xl overflow-hidden border border-slate-700 hover:border-blue-500 transition-all cursor-pointer group"
      onClick={onClick}
    >
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-auto block"
        style={{ aspectRatio: '1/1', objectFit: 'cover' }}
      />

      {/* Loading overlay */}
      {!loaded && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-800">
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-800 text-slate-500 text-xs">
          Error cargando imagen
        </div>
      )}

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-blue-900/0 group-hover:bg-blue-900/20 transition-all pointer-events-none" />

      {/* Badge */}
      <div className="absolute top-2 right-2 bg-green-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-md">
        {tile.detections.length} 🌲
      </div>

      {/* Filename */}
      <div className="px-2 py-1.5 text-[10px] text-slate-500 truncate border-t border-slate-700 bg-slate-900/60">
        {tile.filename}
      </div>
    </div>
  )
}

function drawBoxes(
  ctx: CanvasRenderingContext2D,
  detections: DetectionItem[],
  _w: number,
  _h: number
) {
  for (const d of detections) {
    const x = d.x1
    const y = d.y1
    const w = d.x2 - d.x1
    const h = d.y2 - d.y1

    // Box
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, w, h)

    // Fill (transparent)
    ctx.fillStyle = 'rgba(34, 197, 94, 0.08)'
    ctx.fillRect(x, y, w, h)

    // Label background
    const label = `${(d.confidence * 100).toFixed(0)}%`
    ctx.font = 'bold 11px system-ui'
    const textW = ctx.measureText(label).width + 6
    ctx.fillStyle = '#16a34a'
    ctx.fillRect(x, y - 16, textW, 16)

    // Label text
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, x + 3, y - 4)
  }
}

// ─── Modal ───────────────────────────────────────────────────────────────────

interface ModalProps {
  jobId: string
  tile: TileGroup
  onClose: () => void
}

function TileModal({ jobId, tile, onClose }: ModalProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = tileUrl(jobId, tile.filename)
    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      ctx.drawImage(img, 0, 0)
      drawBoxes(ctx, tile.detections, img.naturalWidth, img.naturalHeight)
    }
  }, [jobId, tile])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative max-w-3xl w-full bg-slate-900 rounded-2xl overflow-hidden border border-slate-700 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
          <div>
            <p className="text-sm font-semibold text-slate-200">{tile.filename}</p>
            <p className="text-xs text-slate-500">{tile.detections.length} detecciones</p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-xl leading-none px-2 py-1 rounded-lg hover:bg-slate-700 transition-colors"
          >
            ✕
          </button>
        </div>
        {/* Canvas */}
        <canvas ref={canvasRef} className="w-full h-auto block max-h-[70vh] object-contain" />
      </div>
    </div>
  )
}

// ─── Main TileViewer ──────────────────────────────────────────────────────────

export default function TileViewer({ result }: Props) {
  const [modalTile, setModalTile] = useState<TileGroup | null>(null)
  const tiles = groupByTile(result.detecciones)

  if (tiles.length === 0) {
    return (
      <div className="text-center py-12 text-slate-600 text-sm">
        No hay tiles con detecciones para mostrar
      </div>
    )
  }

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-200">
          Tiles con detecciones
        </h2>
        <span className="text-xs text-slate-500">
          Mostrando {tiles.length} de {result.tiles_with_detections}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {tiles.map((tile) => (
          <TileCanvas
            key={tile.filename}
            jobId={result.job_id}
            tile={tile}
            onClick={() => setModalTile(tile)}
          />
        ))}
      </div>

      {modalTile && (
        <TileModal
          jobId={result.job_id}
          tile={modalTile}
          onClose={() => setModalTile(null)}
        />
      )}
    </div>
  )
}
