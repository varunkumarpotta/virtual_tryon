"""
AI Virtual Try-On Backend — FastAPI Application
"""
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    logger.info("=" * 60)
    logger.info("  AI Virtual Try-On Backend Starting...")
    logger.info("=" * 60)

    from app.core.device import get_device_info
    device_info = get_device_info()
    logger.info(f"  Device: {device_info['device']}")
    logger.info(f"  Platform: {device_info['platform']}")
    logger.info(f"  RAM: {device_info['ram_total_gb']}GB total, {device_info['ram_available_gb']}GB available")
    if device_info.get("chip"):
        logger.info(f"  Chip: {device_info['chip']}")
    logger.info(f"  PyTorch: {device_info.get('pytorch_version', 'N/A')}")
    logger.info(f"  MPS: {'Available' if device_info.get('mps_available') else 'Not available'}")
    logger.info("=" * 60)

    # Pre-initialize pipeline (load models)
    logger.info("Pre-loading models...")
    from app.pipelines.tryon_pipeline import get_pipeline
    pipeline = get_pipeline()
    pipeline.initialize()
    logger.info("Models loaded successfully!")

    yield

    # Cleanup
    logger.info("Shutting down...")
    pipeline.cleanup()

    if settings.auto_delete_temp:
        import shutil
        for d in [settings.temp_dir, settings.uploads_dir, settings.results_dir]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"Cleaned up {d}")


app = FastAPI(
    title="AI Virtual Try-On",
    description="Local AI-powered virtual try-on application",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{settings.frontend_port}",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        f"http://127.0.0.1:{settings.frontend_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve results directory for image downloads
results_dir = settings.results_dir
results_dir.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
