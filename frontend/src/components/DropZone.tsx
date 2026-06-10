import { useRef, useState, DragEvent, ChangeEvent } from 'react'
import { uploadFile } from '../api/client'
import { UploadResponse } from '../types'

interface Props {
  onUploaded: (res: UploadResponse) => void
}

type DropState = 'idle' | 'dragover' | 'uploading' | 'uploaded' | 'error'

const ACCEPTED = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
const ACCEPTED_MIME = ['image/tiff', 'image/jpeg', 'image/png']

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function isValidFile(file: File): boolean {
  const name = file.name.toLowerCase()
  return (
    ACCEPTED.some((ext) => name.endsWith(ext)) ||
    ACCEPTED_MIME.includes(file.type)
  )
}

export default function DropZone({ onUploaded }: Props) {
  const [state, setState] = useState<DropState>('idle')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function pickFile(file: File) {
    if (!isValidFile(file)) {
      setError(`Formato no soportado: ${file.name}. Use TIF, JPG o PNG.`)
      setState('error')
      return
    }
    setSelectedFile(file)
    setState('idle')
    setError(null)
    setProgress(0)
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setState('dragover')
  }

  function onDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setState('idle')
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) pickFile(file)
    else setState('idle')
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) pickFile(file)
  }

  async function handleUpload() {
    if (!selectedFile) return
    setState('uploading')
    setProgress(0)
    setError(null)
    try {
      const res = await uploadFile(selectedFile, (pct) => setProgress(pct))
      setState('uploaded')
      onUploaded(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido')
      setState('error')
    }
  }

  const isDragOver = state === 'dragover'
  const isUploading = state === 'uploading'
  const isUploaded = state === 'uploaded'

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        className={[
          'relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed transition-all cursor-pointer',
          'min-h-[280px] p-8 text-center',
          isDragOver
            ? 'border-blue-400 bg-blue-950/40 scale-[1.01]'
            : isUploaded
            ? 'border-green-500 bg-green-950/20'
            : 'border-slate-600 bg-slate-800/40 hover:border-blue-500 hover:bg-slate-800/60',
          isUploading ? 'cursor-wait' : '',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          className="hidden"
          onChange={onInputChange}
          disabled={isUploading}
        />

        {/* Icon */}
        <div className={`text-5xl ${isUploaded ? 'text-green-400' : 'text-slate-500'}`}>
          {isUploaded ? '🌳' : isDragOver ? '📂' : '🛰️'}
        </div>

        {/* Text */}
        {!selectedFile ? (
          <>
            <p className="text-lg font-semibold text-slate-300">
              Arrastrá tu ortomosaico aquí
            </p>
            <p className="text-sm text-slate-500">
              Formatos aceptados: TIF, TIFF, JPG, PNG
            </p>
            <p className="text-xs text-slate-600">o hacé click para explorar</p>
          </>
        ) : (
          <div className="flex flex-col items-center gap-1">
            <p className="text-base font-semibold text-slate-200 break-all max-w-sm">
              {selectedFile.name}
            </p>
            <p className="text-sm text-slate-400">{formatBytes(selectedFile.size)}</p>
            {isUploaded && (
              <span className="mt-1 text-xs font-medium text-green-400 bg-green-900/40 px-2 py-0.5 rounded-full">
                ✓ Subido
              </span>
            )}
          </div>
        )}

        {/* Progress bar */}
        {isUploading && (
          <div className="w-full max-w-xs mt-2">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Subiendo...</span>
              <span>{progress}%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300 rounded-full"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-3 flex items-center gap-2 text-red-400 text-sm bg-red-950/30 border border-red-800 rounded-lg px-4 py-2">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Upload button */}
      {selectedFile && !isUploaded && (
        <button
          onClick={handleUpload}
          disabled={isUploading}
          className={[
            'mt-4 w-full py-3 rounded-xl font-semibold text-sm transition-all',
            isUploading
              ? 'bg-slate-700 text-slate-400 cursor-wait'
              : 'bg-blue-600 hover:bg-blue-500 text-white cursor-pointer active:scale-95',
          ].join(' ')}
        >
          {isUploading ? `Subiendo... ${progress}%` : '⬆ Subir ortomosaico'}
        </button>
      )}

      {/* Change file link */}
      {isUploaded && (
        <button
          onClick={() => {
            setState('idle')
            setSelectedFile(null)
            setProgress(0)
            if (inputRef.current) inputRef.current.value = ''
          }}
          className="mt-3 w-full text-sm text-slate-500 hover:text-slate-300 transition-colors"
        >
          Cambiar archivo
        </button>
      )}
    </div>
  )
}
