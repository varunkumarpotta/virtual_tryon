# AI Virtual Try-On

A local AI-powered virtual try-on web application that lets you upload clothing items and virtually try them on using your webcam. All processing happens locally on your device — no data is sent to external servers.

![Status](https://img.shields.io/badge/status-functional%20prototype-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-green)
![Python](https://img.shields.io/badge/python-3.11-yellow)

## Features

- **Upload clothing** — shirts, t-shirts, jackets, pants, dresses, sarees, and more
- **Webcam integration** — capture your photo using your MacBook camera
- **AI pipeline** — pose estimation, body segmentation, garment processing, geometric warping
- **Real-time preview** — camera preview with body guide overlay
- **Background modes** — preserve camera background or remove it
- **Auto-classification** — automatically detects garment type
- **Privacy-first** — 100% local processing, no cloud APIs required
- **Download results** — save try-on images as JPEG
- **Model abstraction** — swap AI models via environment variables

## Requirements

| Requirement | Version |
|---|---|
| macOS | 13+ (Apple Silicon recommended) |
| Python | 3.11 |
| Node.js | 18+ (26.x tested) |
| npm | 9+ |
| RAM | 8 GB minimum |
| Disk | ~2 GB (for models) |

## Quick Start

### 1. Clone and enter the project

```bash
cd virtual_tryon
```

### 2. Create Python virtual environment

```bash
$(brew --prefix python@3.11)/bin/python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env if you need to configure LLM or HuggingFace tokens
```

### 6. Run the application

```bash
./start.sh
```

Open: **http://localhost:5173**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                          │
│  React + Vite + TypeScript                          │
│  Camera • Upload • Preview • Results                │
└─────────────────┬───────────────────────────────────┘
                  │ REST API (proxy via Vite)
┌─────────────────▼───────────────────────────────────┐
│                    Backend                           │
│  FastAPI + Python 3.11                              │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │   Pose   │ │  Segment │ │ Garment  │            │
│  │ MediaPipe│ │ MediaPipe│ │  rembg   │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │             │            │                   │
│  ┌────▼─────────────▼────────────▼─────┐            │
│  │         Try-On Pipeline             │            │
│  │    Geometric Warping + Composite    │            │
│  └─────────────────────────────────────┘            │
│                                                      │
│  ┌──────────────────────┐                           │
│  │  LLM Service (opt.)  │                           │
│  │  OmniRoute endpoint  │                           │
│  └──────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

## AI Pipeline

```
User Photo (webcam capture)
     │
     ▼
Person Detection (MediaPipe Pose)
     │
     ▼
Pose Estimation (33 body landmarks)
     │
     ▼
Body Segmentation (MediaPipe Selfie)
     │
     ▼
Garment Processing (rembg u2net_cloth_seg)
     │  ├── Background removal
     │  ├── Garment masking
     │  └── Auto-classification
     ▼
Geometric Warping (perspective transform)
     │  ├── Target region from pose landmarks
     │  ├── Perspective transform matrix
     │  └── Alpha blending with feathered edges
     ▼
Post-Processing (sharpening, refinement)
     │
     ▼
Result Image
```

## Models Used

| Component | Model | Size | Device | Notes |
|---|---|---|---|---|
| Pose Estimation | MediaPipe Pose | ~12MB | CPU | 33 body landmarks, real-time |
| Person Segmentation | MediaPipe Selfie | ~10MB | CPU | Binary person/background mask |
| Garment Processing | rembg (u2net_cloth_seg) | ~44MB | CPU | Cloth-specific segmentation |
| Try-On | Geometric Warping | N/A | CPU | Perspective transform + alpha blend |

### Why not diffusion-based models?

Full diffusion-based virtual try-on models (IDM-VTON, CatVTON, StableVITON) require 8-16+ GB VRAM. With 8 GB unified memory shared between macOS, applications, and GPU, these models are not practical on this hardware. The geometric warping approach provides instant results with reasonable quality.

## Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Inference device: auto, mps, cpu
INFERENCE_DEVICE=auto

# Model configuration
POSE_MODEL=mediapipe
SEGMENTATION_MODEL=mediapipe
GARMENT_PROCESSOR_MODEL=rembg
TRYON_MODEL=geometric

# HuggingFace token (only for gated models)
HUGGINGFACE_TOKEN=

# LLM Integration (optional)
ANTHROPIC_BASE_URL=http://localhost:20128
ANTHROPIC_AUTH_TOKEN=
ANTHROPIC_MODEL=

# Processing
MAX_INFERENCE_SIZE=512
RESULT_QUALITY=90
```

## Running Components Individually

### Backend only

```bash
source venv/bin/activate
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend only

```bash
cd frontend
npm run dev
```

### Health check

```bash
curl http://localhost:8000/api/health
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Backend health and device info |
| GET | `/api/models` | Loaded models information |
| POST | `/api/garment/upload` | Upload and process garment image |
| POST | `/api/person/capture` | Analyze person image (pose + segmentation) |
| POST | `/api/try-on` | Start virtual try-on (returns job_id) |
| GET | `/api/try-on/{job_id}` | Get try-on job status and result |
| GET | `/api/try-on/{job_id}/download` | Download result image |

## MPS (Apple Silicon GPU) Configuration

PyTorch MPS acceleration is automatically detected. The application will:

1. Check if MPS is available and built
2. Run a sanity check
3. Fall back to CPU if MPS fails

To force CPU:
```bash
INFERENCE_DEVICE=cpu
```

## Troubleshooting

### Backend won't start
- Ensure Python 3.11 is installed: `python3.11 --version`
- Activate venv: `source venv/bin/activate`
- Check dependencies: `pip install -r backend/requirements.txt`

### Camera not working
- Ensure browser has camera permission
- Use Chrome or Safari
- Check System Settings → Privacy & Security → Camera

### Models downloading slowly
- First run downloads models (~100MB total)
- Subsequent runs use cached models

### Out of memory
- Reduce `MAX_INFERENCE_SIZE` in `.env` (try 384 or 256)
- Close other applications
- Reduce image resolution

### MPS errors
- Set `INFERENCE_DEVICE=cpu` in `.env`
- Update macOS to latest version

## Expected Performance (M3 8GB)

| Operation | Time |
|---|---|
| Garment upload + processing | 2-4 seconds |
| Pose estimation | <1 second |
| Segmentation | <1 second |
| Geometric warping try-on | <1 second |
| **Total try-on** | **3-6 seconds** |

## Privacy

- ✅ All processing happens locally
- ✅ No images are sent to external servers
- ✅ Temporary files are auto-deleted
- ✅ No tracking or analytics
- ✅ API tokens never exposed to browser
- ⚠️ If LLM service is configured, text queries (not images) may be sent to your local OmniRoute endpoint

## Known Limitations

1. **Geometric warping** is less realistic than diffusion-based models — garment textures are preserved but may not perfectly adapt to body contours
2. **8 GB RAM** prevents running full generative try-on models locally
3. **Single-person only** — the pipeline is designed for one person in frame
4. **Lighting sensitivity** — extreme lighting may affect pose detection accuracy
5. **Pose coverage** — works best with front-facing, standing poses
6. **Garment classification** — auto-detection uses heuristics, may need manual override for unusual garments

## Next Steps

1. **Upgrade hardware** — 16+ GB M3/M4 would enable CatVTON or similar diffusion models
2. **TPS warping** — implement full Thin Plate Spline warping for better deformation
3. **Human parsing** — integrate SCHP for detailed body part segmentation
4. **GAN refinement** — add lightweight GAN post-processing for more realistic results
5. **Multi-garment** — support layering (shirt + pants + jacket)
6. **Real-time mode** — optimize pipeline for live camera feed processing

## License

For personal use only. Model licenses vary — see individual model pages.