"""
Image utility functions — encoding, decoding, validation
"""
import numpy as np
import cv2
import base64
import io
import logging
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def decode_base64_image(data: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if "," in data:
            data = data.split(",", 1)[1]

        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def encode_image_base64(image: np.ndarray, format: str = "jpeg", quality: int = 90) -> str:
    """Encode OpenCV image to base64 string."""
    if format == "jpeg":
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif format == "png":
        _, buffer = cv2.imencode(".png", image)
    elif format == "webp":
        _, buffer = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        _, buffer = cv2.imencode(".jpg", image)

    return base64.b64encode(buffer).decode("utf-8")


def bytes_to_cv2(data: bytes) -> Optional[np.ndarray]:
    """Convert raw bytes to OpenCV image."""
    try:
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Failed to convert bytes to image: {e}")
        return None


def validate_image_file(filename: str, size: int) -> Tuple[bool, str]:
    """Validate image file by extension and size."""
    import os
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    if size > MAX_FILE_SIZE:
        return False, f"File too large: {size / (1024*1024):.1f}MB. Maximum: {MAX_FILE_SIZE / (1024*1024):.0f}MB"

    return True, "OK"
