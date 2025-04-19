"""Base class for Whisper implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import numpy as np


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    segments: List = field(default_factory=list)
    language: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    implementation: str
    model_name: str
    model_params: Dict[str, Any]
    transcription_time: float
    text: str = ""


class WhisperImplementation(ABC):
    """Base class for Whisper implementations."""

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the model with the given name."""
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {}

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        pass
