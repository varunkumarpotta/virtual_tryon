"""
Device detection — MPS / CPU with automatic fallback
"""
import platform
import logging

logger = logging.getLogger(__name__)

_device = None
_device_info = None


def detect_device() -> str:
    """Detect best available compute device."""
    global _device
    if _device is not None:
        return _device

    from app.core.config import settings

    if settings.inference_device.value != "auto":
        _device = settings.inference_device.value
        logger.info(f"Using configured device: {_device}")
        return _device

    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Quick sanity check
            test = torch.zeros(1, device="mps")
            del test
            _device = "mps"
            logger.info("MPS (Apple Silicon GPU) detected and available")
            return _device
    except Exception as e:
        logger.warning(f"MPS check failed: {e}")

    _device = "cpu"
    logger.info("Using CPU for inference")
    return _device


def get_device_info() -> dict:
    """Get detailed device information."""
    global _device_info
    if _device_info is not None:
        return _device_info

    import psutil

    info = {
        "device": detect_device(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["mps_available"] = False
        info["mps_built"] = False

    # Check for Apple Silicon
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        info["apple_silicon"] = True
        try:
            import subprocess
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True
            ).strip()
            info["chip"] = chip
        except Exception:
            info["chip"] = "Apple Silicon (unknown model)"
    else:
        info["apple_silicon"] = False

    _device_info = info
    return info


def get_torch_device():
    """Get torch device object."""
    import torch
    device_name = detect_device()
    return torch.device(device_name)
