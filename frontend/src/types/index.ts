export interface UploadResponse {
  job_id: string
  filename: string
  size_bytes: number
  path: string
}

export interface ProcessRequest {
  job_id: string
  model_key: string
  conf: number
  iou: number
  tile_size: number
  overlap: number
}

export interface DetectionItem {
  tile_filename: string
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
}

export interface ProcessResponse {
  job_id: string
  model_key: string
  tree_count: number
  tiles_processed: number
  tiles_with_detections: number
  elapsed_sec: number
  tiles_per_sec: number
  detecciones: DetectionItem[]
}

export interface CompareResponse {
  job_id: string
  results: ProcessResponse[]
}

export type AppState =
  | 'idle'
  | 'uploading'
  | 'uploaded'
  | 'processing'
  | 'done'
  | 'comparing'
  | 'error'

export interface ModelInfo {
  key: string
  label: string
  description: string
  speed: string
}
