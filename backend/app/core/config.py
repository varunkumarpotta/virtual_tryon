"""
AI Virtual Try-On Backend — Core Configuration
"""
import os
from pathlib import Path
from enum import Enum
from pydantic_settings import BaseSettings
from typing import Optional


class InferenceDevice(str, Enum):
    AUTO = "auto"
    MPS = "mps"
    CPU = "cpu"


class TryOnModel(str, Enum):
    GEOMETRIC = "geometric"
    CATVTON = "catvton"


class Settings(BaseSettings):
    # Inference
    inference_device: InferenceDevice = InferenceDevice.AUTO
    max_inference_size: int = 512
    result_quality: int = 90

    # Models
    pose_model: str = "mediapipe"
    segmentation_model: str = "mediapipe"
    human_parsing_model: str = "schp"
    garment_processor_model: str = "rembg"
    tryon_model: str = "geometric"

    # HuggingFace
    huggingface_token: Optional[str] = None

    # LLM
    anthropic_base_url: Optional[str] = None
    anthropic_auth_token: Optional[str] = None
    anthropic_model: Optional[str] = None

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 5173

    # Privacy
    auto_delete_temp: bool = True
    log_level: str = "INFO"

    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def models_dir(self) -> Path:
        p = self.base_dir / "models" / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def temp_dir(self) -> Path:
        p = self.base_dir / "tmp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads_dir(self) -> Path:
        p = self.base_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = self.base_dir / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
