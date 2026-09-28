/**
 * Main App Component — AI Virtual Try-On
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import { api, type TryOnStatusResponse } from './services/api';
import { useCamera } from './hooks/useCamera';
import type {
  AppState,
  GarmentData,
  GarmentCategory,
  BackgroundMode,
  ProcessingStatus,
  DeviceInfo,
} from './types';
import { GARMENT_CATEGORIES, BACKGROUND_MODES } from './types';

function App() {
  const [appState, setAppState] = useState<AppState>('landing');
  const [garment, setGarment] = useState<GarmentData | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [_resultJobId, setResultJobId] = useState<string | null>(null);
  const [processing, setProcessing] = useState<ProcessingStatus>({
    progress: 0, message: '', status: 'QUEUED',
  });
  const [selectedCategory, setSelectedCategory] = useState<GarmentCategory>('auto');
  const [backgroundMode, setBackgroundMode] = useState<BackgroundMode>('preserve');
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const _fileInputRef = useRef<HTMLInputElement>(null);
  const camera = useCamera();

  // Check backend health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.health();
        setBackendStatus('online');
        setDeviceInfo(health.device_info as unknown as DeviceInfo);
      } catch {
        setBackendStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle garment upload
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadLoading(true);

    try {
      const originalUrl = URL.createObjectURL(file);
      const result = await api.uploadGarment(file, selectedCategory);

      if (result.success) {
        setGarment({
          id: result.garment_id,
          originalFile: file,
          originalUrl,
          processedUrl: result.processed_image ? `data:image/jpeg;base64,${result.processed_image}` : null,
          maskUrl: result.mask_image ? `data:image/jpeg;base64,${result.mask_image}` : null,
          category: result.category as GarmentCategory,
          region: result.region as 'upper' | 'lower' | 'full',
        });
        setSelectedCategory(result.category as GarmentCategory);
        if (appState === 'landing') {
          setAppState('camera');
          camera.startCamera();
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploadLoading(false);
    }
  }, [selectedCategory, appState, camera]);

  const handleStartCamera = useCallback(async () => {
    await camera.startCamera();
    setAppState('camera');
  }, [camera]);

  const handleCapture = useCallback(() => {
    const frame = camera.captureFrame();
    if (frame) {
      setCapturedImage(frame);
      setAppState('preview');
    } else {
      setError('Failed to capture frame. Is the camera running?');
    }
  }, [camera]);

  const handleRetake = useCallback(() => {
    setCapturedImage(null);
    setResultImage(null);
    setResultJobId(null);
    setError(null);
    setAppState('camera');
  }, []);

  const handleTryOn = useCallback(async () => {
    if (!capturedImage || !garment) {
      setError('Please capture a photo and upload a garment first.');
      return;
    }
    setError(null);
    setAppState('processing');
    setProcessing({ progress: 5, message: 'Starting try-on...', status: 'QUEUED' });

    try {
      const startResult = await api.startTryOn(capturedImage, garment.id, backgroundMode);
      const finalStatus = await api.pollTryOn(
        startResult.job_id,
        (status: TryOnStatusResponse) => {
          setProcessing({ progress: status.progress, message: status.message, status: status.status });
        },
        500, 120
      );

      if (finalStatus.status === 'COMPLETED' && finalStatus.result_image) {
        setResultImage(`data:image/jpeg;base64,${finalStatus.result_image}`);
        setResultJobId(startResult.job_id);
        setAppState('result');
      } else {
        throw new Error(finalStatus.error || 'Try-on failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Try-on failed');
      setAppState('preview');
    }
  }, [capturedImage, garment, backgroundMode]);

  const handleDownload = useCallback(() => {
    if (!resultImage) return;
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `virtual_tryon_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [resultImage]);

  const handleTryAnother = useCallback(() => {
    setGarment(null);
    setResultImage(null);
    setResultJobId(null);
    setCapturedImage(null);
    setError(null);
    setAppState('landing');
    camera.stopCamera();
  }, [camera]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">✦</span>
            <h1>AI Virtual Try-On</h1>
          </div>
          <div className="header-right">
            <div className={`status-badge ${backendStatus}`}>
              <span className="status-dot" />
              {backendStatus === 'online' ? 'Local Processing' : backendStatus === 'checking' ? 'Connecting...' : 'Backend Offline'}
            </div>
            {deviceInfo && (
              <div className="device-badge">
                {deviceInfo.chip || deviceInfo.device.toUpperCase()}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠</span>
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <main className="app-main">
        {/* ---- LANDING ---- */}
        {appState === 'landing' && (
          <div className="landing">
            <div className="landing-hero">
              <div className="hero-glow" />
              <h2>Virtual Try-On</h2>
              <p className="hero-subtitle">
                Upload clothing and see how it looks on you — powered by AI, processed locally on your device.
              </p>

              <div className="landing-actions">
                <label className="btn btn-primary btn-lg" htmlFor="garment-upload-landing">
                  <span className="btn-icon">👔</span>
                  Upload Clothing
                  <input
                    id="garment-upload-landing"
                    type="file"
                    accept=".png,.jpg,.jpeg,.webp"
                    onChange={handleFileSelect}
                    hidden
                  />
                </label>

                <button
                  className="btn btn-secondary btn-lg"
                  onClick={handleStartCamera}
                  disabled={backendStatus !== 'online'}
                >
                  <span className="btn-icon">📷</span>
                  Open Camera
                </button>
              </div>

              {backendStatus === 'offline' && (
                <p className="warning-text">
                  Backend is offline. Start with <code>./start.sh</code>
                </p>
              )}

              <div className="category-selector">
                <label>Garment Type:</label>
                <div className="category-chips">
                  {GARMENT_CATEGORIES.map(cat => (
                    <button
                      key={cat.value}
                      className={`chip ${selectedCategory === cat.value ? 'active' : ''}`}
                      onClick={() => setSelectedCategory(cat.value)}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ---- WORKSPACE ---- */}
        {appState !== 'landing' && (
          <div className="workspace">
            {/* Main Panel */}
            <div className="panel panel-main">
              {(appState === 'camera' || appState === 'upload') && (
                <div className="camera-container">
                  <video ref={camera.videoRef} className="camera-video" autoPlay playsInline muted />
                  <canvas ref={camera.canvasRef} style={{ display: 'none' }} />

                  {!camera.isStreaming && (
                    <div className="camera-placeholder">
                      <span className="placeholder-icon">📷</span>
                      <p>Camera not active</p>
                      <button className="btn btn-primary" onClick={() => camera.startCamera()}>
                        Start Camera
                      </button>
                    </div>
                  )}

                  {camera.error && <div className="camera-error"><p>{camera.error}</p></div>}

                  {camera.isStreaming && (
                    <div className="camera-overlay">
                      <div className="camera-guide" />
                      {camera.devices.length > 1 && (
                        <button className="btn-cam-switch" onClick={camera.switchCamera}>🔄</button>
                      )}
                    </div>
                  )}
                </div>
              )}

              {appState === 'preview' && capturedImage && (
                <div className="preview-container">
                  <img src={capturedImage} alt="Captured" className="preview-image" />
                </div>
              )}

              {appState === 'processing' && (
                <div className="processing-container">
                  {capturedImage && <img src={capturedImage} alt="Processing" className="preview-image processing-bg" />}
                  <div className="processing-overlay">
                    <div className="spinner" />
                    <div className="progress-bar-container">
                      <div className="progress-bar" style={{ width: `${processing.progress}%` }} />
                    </div>
                    <p className="processing-message">{processing.message}</p>
                    <p className="processing-percent">{processing.progress}%</p>
                  </div>
                </div>
              )}

              {appState === 'result' && resultImage && (
                <div className="result-container">
                  <h3 className="result-title">Virtual Try-On Result</h3>
                  <img src={resultImage} alt="Try-On Result" className="result-image" />
                </div>
              )}
            </div>

            {/* Sidebar */}
            <div className="panel panel-sidebar">
              <div className="sidebar-section">
                <h3>Clothing</h3>
                {garment ? (
                  <div className="garment-preview">
                    <div className="garment-images">
                      <div className="garment-thumb">
                        <img src={garment.originalUrl} alt="Original" />
                        <span className="thumb-label">Original</span>
                      </div>
                      {garment.processedUrl && (
                        <div className="garment-thumb">
                          <img src={garment.processedUrl} alt="Processed" />
                          <span className="thumb-label">Processed</span>
                        </div>
                      )}
                    </div>
                    <div className="garment-info">
                      <span className="garment-badge">{garment.category}</span>
                      <span className="garment-region">{garment.region} body</span>
                    </div>
                  </div>
                ) : (
                  <p className="sidebar-hint">No clothing uploaded yet</p>
                )}
                <label className="btn btn-outline btn-block" htmlFor="garment-upload-sidebar">
                  {garment ? 'Change Clothing' : 'Upload Clothing'}
                  <input id="garment-upload-sidebar" type="file" accept=".png,.jpg,.jpeg,.webp" onChange={handleFileSelect} hidden />
                </label>
                {uploadLoading && <div className="upload-spinner" />}
              </div>

              <div className="sidebar-section">
                <h3>Garment Type</h3>
                <select className="select-input" value={selectedCategory} onChange={e => setSelectedCategory(e.target.value as GarmentCategory)}>
                  {GARMENT_CATEGORIES.map(cat => (
                    <option key={cat.value} value={cat.value}>{cat.label}</option>
                  ))}
                </select>
              </div>

              <div className="sidebar-section">
                <h3>Background</h3>
                <div className="radio-group">
                  {BACKGROUND_MODES.map(mode => (
                    <label key={mode.value} className="radio-label">
                      <input type="radio" name="bgMode" value={mode.value} checked={backgroundMode === mode.value} onChange={e => setBackgroundMode(e.target.value as BackgroundMode)} />
                      {mode.label}
                    </label>
                  ))}
                </div>
              </div>

              <div className="sidebar-section sidebar-actions">
                {appState === 'camera' && (
                  <>
                    {!camera.isStreaming && (
                      <button className="btn btn-primary btn-block" onClick={() => camera.startCamera()}>📷 Start Camera</button>
                    )}
                    {camera.isStreaming && (
                      <button className="btn btn-primary btn-block btn-capture" onClick={handleCapture} disabled={!garment}>📸 Capture</button>
                    )}
                  </>
                )}
                {appState === 'preview' && (
                  <>
                    <button className="btn btn-primary btn-block btn-tryon" onClick={handleTryOn} disabled={!garment || !capturedImage}>✨ Try On</button>
                    <button className="btn btn-outline btn-block" onClick={handleRetake}>🔄 Retake</button>
                  </>
                )}
                {appState === 'result' && (
                  <>
                    <button className="btn btn-primary btn-block" onClick={handleDownload}>💾 Download Result</button>
                    <button className="btn btn-outline btn-block" onClick={handleRetake}>🔄 Retake</button>
                    <button className="btn btn-secondary btn-block" onClick={handleTryAnother}>👔 Try Another Clothing</button>
                  </>
                )}
              </div>

              <div className="sidebar-section privacy-notice">
                <span className="privacy-icon">🔒</span>
                <span>All processing happens locally on your device</span>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
