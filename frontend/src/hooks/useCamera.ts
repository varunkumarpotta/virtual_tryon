/**
 * useCamera hook — webcam access, preview, capture
 */
import { useRef, useState, useCallback, useEffect } from 'react';

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  isStreaming: boolean;
  error: string | null;
  devices: MediaDeviceInfo[];
  currentDeviceId: string | null;
  startCamera: (deviceId?: string) => Promise<void>;
  stopCamera: () => void;
  captureFrame: () => string | null;
  switchCamera: () => Promise<void>;
}

export function useCamera(): UseCameraReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState<string | null>(null);
  const [facingIndex, setFacingIndex] = useState(0);

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(devs => {
      const cameras = devs.filter(d => d.kind === 'videoinput');
      setDevices(cameras);
    }).catch(() => {});
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      setError(null);
      stopCamera();

      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          ...(deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: 'user' }),
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const track = stream.getVideoTracks()[0];
      setCurrentDeviceId(track.getSettings().deviceId || null);
      setIsStreaming(true);

      const devs = await navigator.mediaDevices.enumerateDevices();
      setDevices(devs.filter(d => d.kind === 'videoinput'));
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Camera access failed';
      if (msg.includes('NotAllowed') || msg.includes('Permission')) {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (msg.includes('NotFound') || msg.includes('DevicesNotFound')) {
        setError('No camera found. Please connect a camera.');
      } else {
        setError(`Camera error: ${msg}`);
      }
      setIsStreaming(false);
    }
  }, [stopCamera]);

  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    return canvas.toDataURL('image/jpeg', 0.9);
  }, [isStreaming]);

  const switchCamera = useCallback(async () => {
    if (devices.length <= 1) return;
    const nextIndex = (facingIndex + 1) % devices.length;
    setFacingIndex(nextIndex);
    await startCamera(devices[nextIndex].deviceId);
  }, [devices, facingIndex, startCamera]);

  useEffect(() => {
    return () => { stopCamera(); };
  }, [stopCamera]);

  return {
    videoRef, canvasRef, isStreaming, error, devices,
    currentDeviceId, startCamera, stopCamera, captureFrame, switchCamera,
  };
}
