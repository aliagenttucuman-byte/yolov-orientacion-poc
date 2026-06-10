import { UploadResponse, ProcessRequest, ProcessResponse, CompareResponse } from '../types'

const BASE = '/api/v1'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const body = await res.json()
      msg = body.detail ?? body.message ?? msg
    } catch {
      // ignore parse error
    }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

export async function uploadFile(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/upload`)

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse)
        } catch {
          reject(new Error('Respuesta inválida del servidor'))
        }
      } else {
        let msg = `HTTP ${xhr.status}`
        try {
          const body = JSON.parse(xhr.responseText)
          msg = body.detail ?? body.message ?? msg
        } catch {
          // ignore
        }
        reject(new Error(msg))
      }
    })

    xhr.addEventListener('error', () => reject(new Error('Error de red al subir archivo')))
    xhr.addEventListener('abort', () => reject(new Error('Upload cancelado')))

    xhr.send(formData)
  })
}

export async function processJob(req: ProcessRequest): Promise<ProcessResponse> {
  const res = await fetch(`${BASE}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return handleResponse<ProcessResponse>(res)
}

export async function compareModels(
  job_id: string,
  models: string[]
): Promise<CompareResponse> {
  const res = await fetch(`${BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id, models }),
  })
  return handleResponse<CompareResponse>(res)
}

export function tileUrl(job_id: string, filename: string): string {
  return `${BASE}/tiles/${job_id}/${filename}`
}
