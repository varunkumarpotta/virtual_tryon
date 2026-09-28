"""
Pose estimation model — MediaPipe Pose
Detects body landmarks for try-on positioning.
"""
import numpy as np
import logging
from typing import Any, Dict, List, Optional
from app.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for key body parts
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


class MediaPipePoseModel(BaseModel):
    """MediaPipe Pose estimation — lightweight, runs on CPU."""

    def __init__(self, device: str = "cpu"):
        super().__init__("mediapipe_pose", device="cpu")  # Always CPU for MediaPipe
        self._pose = None

    def load(self) -> None:
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )
        self._pose_video = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._loaded = True
        logger.info("MediaPipe Pose loaded")

    def predict(self, image: np.ndarray, video_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Detect pose landmarks.
        Returns dict with landmarks, segmentation mask, and body measurements.
        """
        self.ensure_loaded()
        import cv2

        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        pose_model = self._pose_video if video_mode else self._pose
        results = pose_model.process(rgb)

        if not results.pose_landmarks:
            return {"detected": False, "landmarks": [], "segmentation": None}

        h, w = image.shape[:2]
        landmarks = []
        landmark_dict = {}

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            point = {
                "id": idx,
                "name": LANDMARK_NAMES.get(idx, f"point_{idx}"),
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,
                "visibility": lm.visibility,
                "x_norm": lm.x,
                "y_norm": lm.y,
            }
            landmarks.append(point)
            if idx in LANDMARK_NAMES:
                landmark_dict[LANDMARK_NAMES[idx]] = point

        # Calculate body measurements
        measurements = self._calculate_measurements(landmark_dict, w, h)

        # Segmentation mask
        seg_mask = None
        if results.segmentation_mask is not None:
            seg_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

        return {
            "detected": True,
            "landmarks": landmarks,
            "landmark_dict": landmark_dict,
            "measurements": measurements,
            "segmentation": seg_mask,
            "image_size": {"width": w, "height": h},
        }

    def _calculate_measurements(self, lm: Dict, w: int, h: int) -> Dict[str, float]:
        """Calculate body proportions from landmarks."""
        measurements = {}

        try:
            if "left_shoulder" in lm and "right_shoulder" in lm:
                ls, rs = lm["left_shoulder"], lm["right_shoulder"]
                measurements["shoulder_width"] = abs(rs["x"] - ls["x"])
                measurements["shoulder_center_x"] = (ls["x"] + rs["x"]) / 2
                measurements["shoulder_center_y"] = (ls["y"] + rs["y"]) / 2

            if "left_hip" in lm and "right_hip" in lm:
                lh, rh = lm["left_hip"], lm["right_hip"]
                measurements["hip_width"] = abs(rh["x"] - lh["x"])
                measurements["hip_center_x"] = (lh["x"] + rh["x"]) / 2
                measurements["hip_center_y"] = (lh["y"] + rh["y"]) / 2

            if "shoulder_center_y" in measurements and "hip_center_y" in measurements:
                measurements["torso_height"] = abs(
                    measurements["hip_center_y"] - measurements["shoulder_center_y"]
                )

            if "left_hip" in lm and "left_knee" in lm:
                measurements["upper_leg_length"] = abs(lm["left_knee"]["y"] - lm["left_hip"]["y"])

            if "left_knee" in lm and "left_ankle" in lm:
                measurements["lower_leg_length"] = abs(lm["left_ankle"]["y"] - lm["left_knee"]["y"])

            # Shoulder angle for pose-aware warping
            if "left_shoulder" in lm and "right_shoulder" in lm:
                ls, rs = lm["left_shoulder"], lm["right_shoulder"]
                dx = rs["x"] - ls["x"]
                dy = rs["y"] - ls["y"]
                measurements["shoulder_angle"] = float(np.degrees(np.arctan2(dy, dx)))

        except Exception as e:
            logger.warning(f"Measurement calculation error: {e}")

        return measurements


def create_pose_model(model_name: str = "mediapipe", device: str = "cpu") -> BaseModel:
    """Factory function for pose models."""
    if model_name == "mediapipe":
        return MediaPipePoseModel(device)
    raise ValueError(f"Unknown pose model: {model_name}")
