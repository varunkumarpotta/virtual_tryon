/**
 * TypeScript types for the virtual try-on application
 */

export type GarmentCategory =
  | 'auto' | 'shirt' | 't-shirt' | 'jacket' | 'pants'
  | 'jeans' | 'dress' | 'skirt' | 'saree' | 'other';

export type GarmentRegion = 'upper' | 'lower' | 'full';
export type BackgroundMode = 'preserve' | 'remove';
export type AppState = 'landing' | 'upload' | 'camera' | 'preview' | 'processing' | 'result';
export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface GarmentData {
  id: string;
  originalFile: File | null;
  originalUrl: string;
  processedUrl: string | null;
  maskUrl: string | null;
  category: GarmentCategory;
  region: GarmentRegion;
}

export interface ProcessingStatus {
  progress: number;
  message: string;
  status: JobStatus;
}

export interface DeviceInfo {
  device: string;
  platform: string;
  chip?: string;
  ram_total_gb: number;
  ram_available_gb: number;
  pytorch_version: string;
  mps_available: boolean;
  apple_silicon: boolean;
}

export const GARMENT_CATEGORIES: { value: GarmentCategory; label: string }[] = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'shirt', label: 'Shirt' },
  { value: 't-shirt', label: 'T-Shirt' },
  { value: 'jacket', label: 'Jacket' },
  { value: 'pants', label: 'Pants' },
  { value: 'jeans', label: 'Jeans' },
  { value: 'dress', label: 'Dress' },
  { value: 'skirt', label: 'Skirt' },
  { value: 'saree', label: 'Saree' },
  { value: 'other', label: 'Other' },
];

export const BACKGROUND_MODES: { value: BackgroundMode; label: string }[] = [
  { value: 'preserve', label: 'Preserve Camera' },
  { value: 'remove', label: 'Remove Background' },
];
