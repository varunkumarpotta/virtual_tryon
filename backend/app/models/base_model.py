"""
Base model interface — All AI models must implement this.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all AI models in the pipeline."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._model = None
        logger.info(f"Initializing {self.__class__.__name__} with model={model_name}, device={device}")

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> Any:
        """Run inference on an image."""
        pass

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._loaded = False
        logger.info(f"Unloaded {self.__class__.__name__}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "class": self.__class__.__name__,
            "device": self.device,
            "loaded": self._loaded,
        }
