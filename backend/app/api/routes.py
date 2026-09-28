"""
API Routes — All REST endpoints for the virtual try-on application.
"""
import uuid
import logging
import asyncio
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel as PydanticModel

from app.core.config import settings
from app.core.device import get_device_info
from app.pipelines.tryon_pipeline import get_pipeline
from app.services.llm_service import get_llm_service
from app.utils.image_utils import (
    decode_base64_image,
    encode_image_base64,
    bytes_to_cv2,
    validate_image_file,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# --- Pydantic Models ---

class TryOnRequest(PydanticModel):
    person_image: str  # base64
    garment_id: str
    background_mode: str = "preserve"


class CaptureRequest(PydanticModel):
    image: str  # base64


class HealthResponse(PydanticModel):
    status: str
    device: str
    models_loaded: bool
    llm_available: bool
    version: str = "1.0.0"


# --- Health ---

@router.get("/health")
async def health_check():
    """Check backend health and device info."""
    device_info = get_device_info()
    llm = get_llm_service()

    return {
        "status": "healthy",
        "device": device_info["device"],
        "device_info": device_info,
        "llm_available": llm.available,
        "version": "1.0.0",
        "privacy": "local_processing",
    }


# --- Models ---

@router.get("/models")
async def get_models():
    """Get information about loaded models."""
    pipeline = get_pipeline()
    try:
        info = pipeline.get_models_info()
        llm = get_llm_service()
        info["llm"] = llm.get_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e), "models": {}}


# --- Garment Upload ---

@router.post("/garment/upload")
async def upload_garment(
    file: UploadFile = File(...),
    category: str = Form("auto"),
):
    """
    Upload and process a garment image.
    Returns garment_id and processing results.
    """
    # Validate
    valid, msg = validate_image_file(file.filename or "unknown.jpg", file.size or 0)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    try:
        # Read file
        contents = await file.read()
        image = bytes_to_cv2(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file")

        # Process garment
        pipeline = get_pipeline()
        result = pipeline.process_garment(image, category=category)

        # Encode processed image for frontend
        garment_data = pipeline._garment_cache.get(result["garment_id"])
        processed_b64 = None
        mask_b64 = None
        if garment_data:
            processed_b64 = encode_image_base64(garment_data["processed_image"])
            mask_b64 = encode_image_base64(garment_data["mask"])

        return {
            "success": True,
            "garment_id": result["garment_id"],
            "category": result["category"],
            "region": result["region"],
            "bbox": result["bbox"],
            "processed_image": processed_b64,
            "mask_image": mask_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Garment upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Garment processing failed: {str(e)}")


# --- Person Capture ---

@router.post("/person/capture")
async def capture_person(request: CaptureRequest):
    """
    Analyze a captured person image (from webcam).
    Returns pose and segmentation data.
    """
    try:
        image = decode_base64_image(request.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        pipeline = get_pipeline()
        result = pipeline.process_person(image)

        if not result["detected"]:
            return {
                "success": False,
                "error": result.get("error", "No person detected"),
                "suggestions": [
                    "Make sure you are visible in the camera",
                    "Try better lighting",
                    "Stand further from the camera",
                    "Face the camera directly",
                ],
            }

        # Encode segmentation mask
        seg_b64 = None
        if result.get("segmentation_mask") is not None:
            seg_b64 = encode_image_base64(result["segmentation_mask"])

        # Simplify pose data for frontend
        landmarks_simple = []
        for lm in result["pose"]["landmarks"]:
            landmarks_simple.append({
                "id": lm["id"],
                "name": lm["name"],
                "x": round(lm["x_norm"], 4),
                "y": round(lm["y_norm"], 4),
                "visibility": round(lm["visibility"], 2),
            })

        return {
            "success": True,
            "landmarks": landmarks_simple,
            "measurements": result["pose"]["measurements"],
            "segmentation": seg_b64,
            "image_size": result["image_size"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Person capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"Person analysis failed: {str(e)}")


# --- Try-On ---

# Store for async job results
_tryon_results = {}


@router.post("/try-on")
async def start_tryon(request: TryOnRequest, background_tasks: BackgroundTasks):
    """
    Start a virtual try-on job. Returns job_id for polling.
    """
    try:
        image = decode_base64_image(request.person_image)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode person image")

        pipeline = get_pipeline()

        # Check garment exists
        if request.garment_id not in pipeline._garment_cache:
            raise HTTPException(status_code=404, detail="Garment not found. Please upload a garment first.")

        job_id = str(uuid.uuid4())

        # Store initial status
        pipeline._jobs[job_id] = {
            "status": "QUEUED",
            "progress": 0,
            "message": "Queued for processing...",
        }

        # Run in background
        background_tasks.add_task(
            _run_tryon_task,
            job_id, image, request.garment_id, request.background_mode
        )

        return {
            "job_id": job_id,
            "status": "QUEUED",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Try-on start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_tryon_task(
    job_id: str,
    person_image: np.ndarray,
    garment_id: str,
    background_mode: str,
):
    """Background task for try-on processing."""
    pipeline = get_pipeline()

    # Run synchronous pipeline in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.try_on(
            person_image=person_image,
            garment_id=garment_id,
            background_mode=background_mode,
            job_id=job_id,
        ),
    )

    # Store result
    _tryon_results[job_id] = result


@router.get("/try-on/{job_id}")
async def get_tryon_status(job_id: str):
    """Get try-on job status and result."""
    pipeline = get_pipeline()
    status = pipeline.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job_id,
        "status": status["status"],
        "progress": status["progress"],
        "message": status["message"],
    }

    # If completed, include result image
    if status["status"] == "COMPLETED" and job_id in _tryon_results:
        result = _tryon_results[job_id]
        if result.get("success") and result.get("result_path"):
            # Read result image and encode
            result_image = cv2.imread(result["result_path"])
            if result_image is not None:
                response["result_image"] = encode_image_base64(result_image)
                response["garment_category"] = result.get("garment_category")
                response["garment_region"] = result.get("garment_region")
                response["elapsed_seconds"] = result.get("elapsed_seconds")
        elif result.get("error"):
            response["status"] = "FAILED"
            response["error"] = result["error"]

    return response


@router.get("/try-on/{job_id}/download")
async def download_result(job_id: str):
    """Download the try-on result image."""
    result_path = settings.results_dir / f"{job_id}.jpg"
    if not result_path.exists():
        # Check _tryon_results
        if job_id in _tryon_results and _tryon_results[job_id].get("result_path"):
            result_path = _tryon_results[job_id]["result_path"]

    from pathlib import Path
    result_path = Path(result_path)

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    return FileResponse(
        str(result_path),
        media_type="image/jpeg",
        filename=f"virtual_tryon_{job_id[:8]}.jpg",
    )
