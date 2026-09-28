/**
 * API service — communicates with FastAPI backend
 */

const API_BASE = '/api';

export interface HealthResponse {
  status: string;
  device: string;
  device_info: Record<string, unknown>;
  llm_available: boolean;
  version: string;
  privacy: string;
}

export interface GarmentUploadResponse {
  success: boolean;
  garment_id: string;
  category: string;
  region: string;
  bbox: { x: number; y: number; w: number; h: number };
  processed_image: string | null;
  mask_image: string | null;
}

export interface PersonCaptureResponse {
  success: boolean;
  error?: string;
  suggestions?: string[];
  landmarks?: Array<{
    id: number;
    name: string;
    x: number;
    y: number;
    visibility: number;
  }>;
  measurements?: Record<string, number>;
  segmentation?: string;
  image_size?: { width: number; height: number };
}

export interface TryOnStartResponse {
  job_id: string;
  status: string;
}

export interface TryOnStatusResponse {
  job_id: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress: number;
  message: string;
  result_image?: string;
  garment_category?: string;
  garment_region?: string;
  elapsed_seconds?: number;
  error?: string;
}

export interface ModelsResponse {
  device: string;
  models: Record<string, { name: string; class: string; device: string; loaded: boolean }>;
  llm?: { available: boolean; endpoint: string | null; model: string | null };
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  async health(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
  }

  async getModels(): Promise<ModelsResponse> {
    const res = await fetch(`${this.baseUrl}/models`);
    if (!res.ok) throw new Error(`Failed to get models: ${res.status}`);
    return res.json();
  }

  async uploadGarment(file: File, category: string = 'auto'): Promise<GarmentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);

    const res = await fetch(`${this.baseUrl}/garment/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Garment upload failed');
    }

    return res.json();
  }

  async capturePerson(imageBase64: string): Promise<PersonCaptureResponse> {
    const res = await fetch(`${this.baseUrl}/person/capture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Capture failed' }));
      throw new Error(err.detail || 'Person capture failed');
    }

    return res.json();
  }

  async startTryOn(
    personImageBase64: string,
    garmentId: string,
    backgroundMode: string = 'preserve'
  ): Promise<TryOnStartResponse> {
    const res = await fetch(`${this.baseUrl}/try-on`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        person_image: personImageBase64,
        garment_id: garmentId,
        background_mode: backgroundMode,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Try-on failed' }));
      throw new Error(err.detail || 'Try-on start failed');
    }

    return res.json();
  }

  async getTryOnStatus(jobId: string): Promise<TryOnStatusResponse> {
    const res = await fetch(`${this.baseUrl}/try-on/${jobId}`);
    if (!res.ok) throw new Error(`Failed to get status: ${res.status}`);
    return res.json();
  }

  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/try-on/${jobId}/download`;
  }

  async pollTryOn(
    jobId: string,
    onProgress: (status: TryOnStatusResponse) => void,
    intervalMs: number = 500,
    maxAttempts: number = 120
  ): Promise<TryOnStatusResponse> {
    for (let i = 0; i < maxAttempts; i++) {
      const status = await this.getTryOnStatus(jobId);
      onProgress(status);

      if (status.status === 'COMPLETED' || status.status === 'FAILED') {
        return status;
      }

      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }

    throw new Error('Try-on timed out');
  }
}

export const api = new ApiService();
export default api;
