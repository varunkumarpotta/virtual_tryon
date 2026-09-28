"""
Person/body segmentation model — MediaPipe + rembg
"""
import numpy as np
import cv2
import logging
from typing import Any, Dict
from app.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MediaPipeSegmentationModel(BaseModel):
    """MediaPipe Selfie Segmentation — fast, lightweight."""

    def __init__(self, device: str = "cpu"):
        super().__init__("mediapipe_segmentation", device="cpu")

    def load(self) -> None:
        import mediapipe as mp
        self._mp_selfie = mp.solutions.selfie_segmentation
        self._segmenter = self._mp_selfie.SelfieSegmentation(model_selection=1)
        self._loaded = True
        logger.info("MediaPipe Selfie Segmentation loaded")

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._segmenter.process(rgb)

        mask = results.segmentation_mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Refine mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Smooth edges
        binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
        binary_mask = (binary_mask > 128).astype(np.uint8) * 255

        return {
            "mask": binary_mask,
            "confidence_map": mask,
        }


class RembgSegmentationModel(BaseModel):
    """rembg-based background removal / segmentation using U2-Net."""

    def __init__(self, device: str = "cpu"):
        super().__init__("rembg_segmentation", device="cpu")

    def load(self) -> None:
        from rembg import new_session
        self._session = new_session("u2net")
        self._loaded = True
        logger.info("rembg (U2-Net) segmentation loaded")

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.ensure_loaded()
        from rembg import remove
        from PIL import Image
        import io

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = remove(pil_img, session=self._session, only_mask=True)
        mask = np.array(result)

        return {
            "mask": mask,
            "confidence_map": mask.astype(np.float32) / 255.0,
        }


def create_segmentation_model(model_name: str = "mediapipe", device: str = "cpu") -> BaseModel:
    """Factory function for segmentation models."""
    if model_name == "mediapipe":
        return MediaPipeSegmentationModel(device)
    elif model_name == "rembg":
        return RembgSegmentationModel(device)
    raise ValueError(f"Unknown segmentation model: {model_name}")
