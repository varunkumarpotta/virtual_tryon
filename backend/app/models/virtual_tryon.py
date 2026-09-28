"""
Virtual Try-On model — Geometric warping pipeline
Uses TPS (Thin Plate Spline) warping to deform garments onto detected body poses.
"""
import numpy as np
import cv2
import logging
from typing import Any, Dict, Optional, Tuple
from app.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GeometricTryOnModel(BaseModel):
    """
    Geometric warping-based virtual try-on.
    Uses TPS warping + alpha blending for garment placement.
    Fast, works on CPU, no heavy ML model needed.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__("geometric_tryon", device="cpu")

    def load(self) -> None:
        self._loaded = True
        logger.info("Geometric Try-On model ready")

    def predict(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_mask: np.ndarray,
        pose_data: Dict[str, Any],
        garment_region: str = "upper",
        person_mask: Optional[np.ndarray] = None,
        background_mode: str = "preserve",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform geometric try-on:
        1. Define target region on person from pose
        2. Warp garment to match body shape
        3. Composite with alpha blending
        """
        self.ensure_loaded()
        h, w = person_image.shape[:2]
        landmark_dict = pose_data.get("landmark_dict", {})
        measurements = pose_data.get("measurements", {})

        if not landmark_dict:
            return {"success": False, "error": "No pose landmarks detected"}

        # Get target region based on garment type
        target_points, target_size = self._get_target_region(
            landmark_dict, measurements, garment_region, w, h
        )

        if target_points is None:
            return {"success": False, "error": "Could not determine target region"}

        # Warp the garment
        warped_garment, warped_mask = self._warp_garment(
            garment_image, garment_mask, target_points, target_size, (w, h)
        )

        # Create the composite
        result = self._composite(
            person_image, warped_garment, warped_mask,
            person_mask, background_mode
        )

        return {
            "success": True,
            "result_image": result,
            "warped_garment": warped_garment,
            "warped_mask": warped_mask,
        }

    def _get_target_region(
        self,
        landmarks: Dict,
        measurements: Dict,
        region: str,
        img_w: int,
        img_h: int,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Calculate target quadrilateral for garment placement."""

        if region == "upper":
            return self._get_upper_body_region(landmarks, measurements, img_w, img_h)
        elif region == "lower":
            return self._get_lower_body_region(landmarks, measurements, img_w, img_h)
        elif region == "full":
            return self._get_full_body_region(landmarks, measurements, img_w, img_h)

        return None, None

    def _get_upper_body_region(
        self, lm: Dict, meas: Dict, img_w: int, img_h: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Get upper body region (shirt, t-shirt, jacket)."""
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in lm for k in required):
            return None, None

        ls = lm["left_shoulder"]
        rs = lm["right_shoulder"]
        lh = lm["left_hip"]
        rh = lm["right_hip"]

        # Expand shoulders slightly for garment overhang
        shoulder_w = abs(rs["x"] - ls["x"])
        expand = shoulder_w * 0.2

        # Account for shoulder angle
        angle = meas.get("shoulder_angle", 0)

        # Define quadrilateral corners
        # Top-left, Top-right, Bottom-right, Bottom-left
        tl_x = ls["x"] - expand
        tl_y = ls["y"] - shoulder_w * 0.15  # Slightly above shoulders for collar
        tr_x = rs["x"] + expand
        tr_y = rs["y"] - shoulder_w * 0.15

        # Hip region - extend slightly below hips
        hip_expand = abs(rh["x"] - lh["x"]) * 0.15
        bl_x = lh["x"] - hip_expand
        bl_y = lh["y"] + shoulder_w * 0.1
        br_x = rh["x"] + hip_expand
        br_y = rh["y"] + shoulder_w * 0.1

        # Clamp to image bounds
        points = np.array([
            [max(0, tl_x), max(0, tl_y)],
            [min(img_w, tr_x), max(0, tr_y)],
            [min(img_w, br_x), min(img_h, br_y)],
            [max(0, bl_x), min(img_h, bl_y)],
        ], dtype=np.float32)

        # Target size for warped garment
        target_w = int(max(abs(tr_x - tl_x), abs(br_x - bl_x)))
        target_h = int(max(abs(bl_y - tl_y), abs(br_y - tr_y)))

        return points, (target_w, target_h)

    def _get_lower_body_region(
        self, lm: Dict, meas: Dict, img_w: int, img_h: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Get lower body region (pants, jeans, skirt)."""
        required = ["left_hip", "right_hip"]
        if not all(k in lm for k in required):
            return None, None

        lh = lm["left_hip"]
        rh = lm["right_hip"]

        hip_w = abs(rh["x"] - lh["x"])
        expand = hip_w * 0.25

        # Bottom: use ankles if available, otherwise estimate
        if "left_ankle" in lm and "right_ankle" in lm:
            la = lm["left_ankle"]
            ra = lm["right_ankle"]
            bottom_y = max(la["y"], ra["y"]) + hip_w * 0.1
        elif "left_knee" in lm and "right_knee" in lm:
            lk = lm["left_knee"]
            rk = lm["right_knee"]
            # Estimate ankle from knee
            knee_y = max(lk["y"], rk["y"])
            bottom_y = knee_y + (knee_y - min(lh["y"], rh["y"]))
        else:
            bottom_y = min(lh["y"], rh["y"]) + hip_w * 2

        points = np.array([
            [lh["x"] - expand, lh["y"] - hip_w * 0.05],
            [rh["x"] + expand, rh["y"] - hip_w * 0.05],
            [rh["x"] + expand * 0.8, min(img_h, bottom_y)],
            [lh["x"] - expand * 0.8, min(img_h, bottom_y)],
        ], dtype=np.float32)

        target_w = int(hip_w + expand * 2)
        target_h = int(bottom_y - min(lh["y"], rh["y"]))

        return points, (target_w, target_h)

    def _get_full_body_region(
        self, lm: Dict, meas: Dict, img_w: int, img_h: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Get full body region (dress, saree)."""
        required = ["left_shoulder", "right_shoulder"]
        if not all(k in lm for k in required):
            return None, None

        ls = lm["left_shoulder"]
        rs = lm["right_shoulder"]
        shoulder_w = abs(rs["x"] - ls["x"])
        expand = shoulder_w * 0.25

        # Bottom
        if "left_ankle" in lm and "right_ankle" in lm:
            la = lm["left_ankle"]
            ra = lm["right_ankle"]
            bottom_y = max(la["y"], ra["y"]) + shoulder_w * 0.1
        elif "left_knee" in lm and "right_knee" in lm:
            lk = lm["left_knee"]
            rk = lm["right_knee"]
            bottom_y = max(lk["y"], rk["y"]) + shoulder_w * 0.5
        else:
            bottom_y = ls["y"] + shoulder_w * 3

        points = np.array([
            [ls["x"] - expand, ls["y"] - shoulder_w * 0.15],
            [rs["x"] + expand, rs["y"] - shoulder_w * 0.15],
            [rs["x"] + expand * 1.2, min(img_h, bottom_y)],
            [ls["x"] - expand * 1.2, min(img_h, bottom_y)],
        ], dtype=np.float32)

        target_w = int(shoulder_w + expand * 2.4)
        target_h = int(bottom_y - min(ls["y"], rs["y"]))

        return points, (target_w, target_h)

    def _warp_garment(
        self,
        garment: np.ndarray,
        mask: np.ndarray,
        target_points: np.ndarray,
        target_size: Tuple[int, int],
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp garment image to fit target body region using perspective transform."""
        gh, gw = garment.shape[:2]
        out_w, out_h = output_size

        # Source points: corners of garment image
        src_points = np.array([
            [0, 0],
            [gw, 0],
            [gw, gh],
            [0, gh],
        ], dtype=np.float32)

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_points, target_points)

        # Warp garment
        warped_garment = cv2.warpPerspective(
            garment, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Warp mask
        warped_mask = cv2.warpPerspective(
            mask, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped_garment, warped_mask

    def _composite(
        self,
        person: np.ndarray,
        warped_garment: np.ndarray,
        warped_mask: np.ndarray,
        person_mask: Optional[np.ndarray],
        background_mode: str,
    ) -> np.ndarray:
        """Composite garment onto person image."""
        result = person.copy()

        # Normalize mask to float [0, 1]
        alpha = warped_mask.astype(np.float32) / 255.0

        # Feather edges for smoother blending
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        # Expand alpha to 3 channels
        if len(alpha.shape) == 2:
            alpha_3 = np.stack([alpha] * 3, axis=-1)
        else:
            alpha_3 = alpha

        # Apply garment with alpha blending
        h, w = result.shape[:2]
        gh, gw = warped_garment.shape[:2]
        # Ensure sizes match
        if gh != h or gw != w:
            warped_garment = cv2.resize(warped_garment, (w, h))
            alpha_3 = cv2.resize(alpha_3, (w, h))

        result = (warped_garment.astype(np.float32) * alpha_3 +
                  result.astype(np.float32) * (1 - alpha_3)).astype(np.uint8)

        # Handle background
        if background_mode == "remove" and person_mask is not None:
            bg_mask = person_mask.astype(np.float32) / 255.0
            if len(bg_mask.shape) == 2:
                bg_mask_3 = np.stack([bg_mask] * 3, axis=-1)
            else:
                bg_mask_3 = bg_mask
            # White background
            white_bg = np.ones_like(result) * 255
            # Combine garment mask with person mask
            combined_mask = np.maximum(bg_mask_3, alpha_3)
            result = (result.astype(np.float32) * combined_mask +
                      white_bg.astype(np.float32) * (1 - combined_mask)).astype(np.uint8)

        return result


def create_tryon_model(model_name: str = "geometric", device: str = "cpu") -> BaseModel:
    """Factory function for try-on models."""
    if model_name == "geometric":
        return GeometricTryOnModel(device)
    raise ValueError(f"Unknown try-on model: {model_name}")
