"""
Garment processing — background removal, masking, classification
"""
import numpy as np
import cv2
import logging
from typing import Any, Dict, Tuple
from app.models.base_model import BaseModel

logger = logging.getLogger(__name__)

GARMENT_CATEGORIES = [
    "auto", "shirt", "t-shirt", "jacket", "pants", "jeans",
    "dress", "skirt", "saree", "other"
]

# Upper-body vs lower-body vs full-body garments
UPPER_BODY = {"shirt", "t-shirt", "jacket"}
LOWER_BODY = {"pants", "jeans", "skirt"}
FULL_BODY = {"dress", "saree"}


class GarmentProcessor(BaseModel):
    """Process garment images: background removal, masking, classification."""

    def __init__(self, device: str = "cpu"):
        super().__init__("garment_processor", device="cpu")
        self._rembg_session = None

    def load(self) -> None:
        from rembg import new_session
        self._rembg_session = new_session("u2net_cloth_seg")
        self._loaded = True
        logger.info("Garment processor (u2net_cloth_seg) loaded")

    def predict(self, image: np.ndarray, category: str = "auto", **kwargs) -> Dict[str, Any]:
        """
        Process a garment image:
        1. Remove background
        2. Create garment mask
        3. Classify garment type
        4. Extract garment region
        """
        self.ensure_loaded()
        from rembg import remove
        from PIL import Image

        h, w = image.shape[:2]

        # Convert to PIL for rembg
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Remove background — get RGBA result
        result_rgba = remove(pil_img, session=self._rembg_session)
        result_np = np.array(result_rgba)

        # Extract mask from alpha channel
        if result_np.shape[2] == 4:
            garment_mask = result_np[:, :, 3]
            garment_rgb = cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            # Fallback: create mask from difference
            garment_rgb = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(garment_rgb, cv2.COLOR_BGR2GRAY)
            _, garment_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Get bounding box of garment
        bbox = self._get_bounding_box(garment_mask)

        # Auto-classify if needed
        if category == "auto":
            category = self._classify_garment(garment_mask, image, bbox)

        # Determine garment region type
        region = "upper"
        if category in LOWER_BODY:
            region = "lower"
        elif category in FULL_BODY:
            region = "full"

        return {
            "processed_image": garment_rgb,
            "mask": garment_mask,
            "category": category,
            "region": region,
            "bbox": bbox,
            "original_size": {"width": w, "height": h},
        }

    def _get_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """Get bounding box of non-zero region in mask."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            h, w = mask.shape[:2]
            return {"x": 0, "y": 0, "w": w, "h": h}
        x, y, w, h = cv2.boundingRect(coords)
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    def _classify_garment(self, mask: np.ndarray, image: np.ndarray, bbox: Dict) -> str:
        """Simple heuristic-based garment classification using aspect ratio and position."""
        h_img, w_img = mask.shape[:2]
        bw, bh = bbox["w"], bbox["h"]

        if bw == 0 or bh == 0:
            return "other"

        aspect_ratio = bw / bh
        # Position: where is the center of mass relative to image height?
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cy = moments["m01"] / moments["m00"]
            cy_ratio = cy / h_img
        else:
            cy_ratio = 0.5

        # Coverage ratio
        mask_area = cv2.countNonZero(mask)
        bbox_area = bw * bh
        fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0

        # Heuristic classification
        if aspect_ratio > 1.5 and cy_ratio < 0.4:
            return "t-shirt"
        elif aspect_ratio > 0.6 and aspect_ratio < 1.5 and cy_ratio < 0.5:
            if bh > h_img * 0.5:
                return "jacket"
            return "shirt"
        elif aspect_ratio < 0.7 and cy_ratio > 0.4:
            if fill_ratio > 0.6:
                return "pants"
            return "skirt"
        elif bh > h_img * 0.6:
            return "dress"
        else:
            return "shirt"  # Default to shirt


def create_garment_processor(model_name: str = "rembg", device: str = "cpu") -> BaseModel:
    """Factory function for garment processors."""
    return GarmentProcessor(device)
