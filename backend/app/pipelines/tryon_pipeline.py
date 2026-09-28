"""
Try-On Pipeline — Orchestrates the full virtual try-on workflow.
"""
import numpy as np
import cv2
import uuid
import time
import logging
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from app.core.config import settings
from app.core.device import detect_device
from app.models.pose import create_pose_model
from app.models.segmentation import create_segmentation_model
from app.models.garment_parser import create_garment_processor
from app.models.virtual_tryon import create_tryon_model

logger = logging.getLogger(__name__)


class TryOnPipeline:
    """
    Full virtual try-on pipeline:
    Person image → Pose → Segmentation → Garment warp → Composite → Result
    """

    def __init__(self):
        self.device = detect_device()
        self._pose_model = None
        self._segmentation_model = None
        self._garment_processor = None
        self._tryon_model = None
        self._initialized = False

        # Cache
        self._garment_cache: Dict[str, Dict] = {}
        self._jobs: Dict[str, Dict] = {}

    def initialize(self) -> None:
        """Lazy-load all models."""
        if self._initialized:
            return

        logger.info(f"Initializing pipeline on device: {self.device}")

        self._pose_model = create_pose_model(settings.pose_model, self.device)
        self._segmentation_model = create_segmentation_model(settings.segmentation_model, self.device)
        self._garment_processor = create_garment_processor(settings.garment_processor_model, self.device)
        self._tryon_model = create_tryon_model(settings.tryon_model, self.device)

        self._initialized = True
        logger.info("Pipeline initialized")

    def get_models_info(self) -> Dict[str, Any]:
        """Get info about loaded models."""
        self.initialize()
        return {
            "device": self.device,
            "models": {
                "pose": self._pose_model.get_info(),
                "segmentation": self._segmentation_model.get_info(),
                "garment_processor": self._garment_processor.get_info(),
                "tryon": self._tryon_model.get_info(),
            },
        }

    def process_garment(
        self,
        image: np.ndarray,
        category: str = "auto",
        garment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process and cache a garment image.
        Returns garment_id and processing results.
        """
        self.initialize()

        if garment_id is None:
            garment_id = str(uuid.uuid4())

        logger.info(f"Processing garment {garment_id}, category={category}")

        # Resize if needed
        image = self._resize_for_inference(image)

        # Process garment
        self._garment_processor.ensure_loaded()
        result = self._garment_processor.predict(image, category=category)

        # Cache processed garment
        self._garment_cache[garment_id] = {
            "processed_image": result["processed_image"],
            "mask": result["mask"],
            "category": result["category"],
            "region": result["region"],
            "bbox": result["bbox"],
            "original_size": result["original_size"],
        }

        return {
            "garment_id": garment_id,
            "category": result["category"],
            "region": result["region"],
            "bbox": result["bbox"],
        }

    def process_person(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a person image: pose + segmentation.
        """
        self.initialize()

        image = self._resize_for_inference(image)

        # Pose estimation
        self._pose_model.ensure_loaded()
        pose_data = self._pose_model.predict(image)

        if not pose_data["detected"]:
            return {"detected": False, "error": "No person detected in the image"}

        # Person segmentation
        self._segmentation_model.ensure_loaded()
        seg_data = self._segmentation_model.predict(image)

        return {
            "detected": True,
            "pose": pose_data,
            "segmentation_mask": seg_data["mask"],
            "image_size": pose_data["image_size"],
        }

    def try_on(
        self,
        person_image: np.ndarray,
        garment_id: str,
        background_mode: str = "preserve",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full try-on pipeline:
        1. Analyze person (pose + segmentation)
        2. Load cached garment
        3. Warp garment
        4. Composite
        """
        self.initialize()
        if job_id is None:
            job_id = str(uuid.uuid4())
        start_time = time.time()

        def update_progress(pct: int, msg: str):
            self._jobs[job_id] = {
                "status": "PROCESSING",
                "progress": pct,
                "message": msg,
            }
            if progress_callback:
                progress_callback(pct, msg)

        try:
            self._jobs[job_id] = {"status": "QUEUED", "progress": 0, "message": "Starting..."}

            # Step 1: Get garment data
            update_progress(10, "Loading garment data...")
            if garment_id not in self._garment_cache:
                return {
                    "job_id": job_id,
                    "success": False,
                    "error": "Garment not found. Please upload a garment first.",
                }

            garment_data = self._garment_cache[garment_id]

            # Step 2: Resize person image
            update_progress(15, "Preparing person image...")
            person_image = self._resize_for_inference(person_image)

            # Step 3: Pose estimation
            update_progress(25, "Analyzing pose...")
            self._pose_model.ensure_loaded()
            pose_data = self._pose_model.predict(person_image)

            if not pose_data["detected"]:
                self._jobs[job_id] = {"status": "FAILED", "progress": 0, "message": "No person detected"}
                return {
                    "job_id": job_id,
                    "success": False,
                    "error": "No person detected. Please ensure you are visible in the camera.",
                }

            # Step 4: Person segmentation
            update_progress(40, "Segmenting person...")
            self._segmentation_model.ensure_loaded()
            seg_data = self._segmentation_model.predict(person_image)

            # Step 5: Generate try-on
            update_progress(60, "Generating virtual try-on...")
            self._tryon_model.ensure_loaded()
            tryon_result = self._tryon_model.predict(
                person_image=person_image,
                garment_image=garment_data["processed_image"],
                garment_mask=garment_data["mask"],
                pose_data=pose_data,
                garment_region=garment_data["region"],
                person_mask=seg_data["mask"],
                background_mode=background_mode,
            )

            if not tryon_result.get("success", False):
                self._jobs[job_id] = {
                    "status": "FAILED",
                    "progress": 0,
                    "message": tryon_result.get("error", "Try-on failed"),
                }
                return {
                    "job_id": job_id,
                    "success": False,
                    "error": tryon_result.get("error", "Try-on generation failed"),
                }

            # Step 6: Post-processing
            update_progress(85, "Refining result...")
            result_image = self._refine_result(tryon_result["result_image"])

            # Step 7: Save result
            update_progress(95, "Saving result...")
            result_path = self._save_result(result_image, job_id)

            elapsed = time.time() - start_time

            self._jobs[job_id] = {
                "status": "COMPLETED",
                "progress": 100,
                "message": f"Completed in {elapsed:.1f}s",
            }

            update_progress(100, f"Completed in {elapsed:.1f}s")

            return {
                "job_id": job_id,
                "success": True,
                "result_path": str(result_path),
                "elapsed_seconds": round(elapsed, 1),
                "garment_category": garment_data["category"],
                "garment_region": garment_data["region"],
            }

        except Exception as e:
            logger.exception(f"Try-on pipeline failed: {e}")
            self._jobs[job_id] = {"status": "FAILED", "progress": 0, "message": str(e)}
            return {"job_id": job_id, "success": False, "error": str(e)}

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def _resize_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        max_dim = settings.max_inference_size
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image

        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _refine_result(self, image: np.ndarray) -> np.ndarray:
        """Post-processing refinement."""
        # Subtle sharpening
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ]) / 1.0
        sharpened = cv2.filter2D(image, -1, kernel)
        # Blend original with sharpened
        result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
        return result

    def _save_result(self, image: np.ndarray, job_id: str) -> Path:
        """Save result image to results directory."""
        result_path = settings.results_dir / f"{job_id}.jpg"
        cv2.imwrite(
            str(result_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, settings.result_quality],
        )
        return result_path

    def cleanup(self) -> None:
        """Cleanup temp files and caches."""
        self._garment_cache.clear()
        self._jobs.clear()
        logger.info("Pipeline cache cleared")


# Singleton pipeline
_pipeline: Optional[TryOnPipeline] = None


def get_pipeline() -> TryOnPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TryOnPipeline()
    return _pipeline
