"""Faster Whisper implementation."""

from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class FasterWhisperImplementation(WhisperImplementation):
    """Whisper implementation using Faster Whisper."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.beam_size = 1
        self.language = None
        self.cpu_threads = 4

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            from faster_whisper import WhisperModel
            self._faster_whisper = WhisperModel
        except ImportError:
            self.log.error("Failed to import faster_whisper. Make sure it's installed.")
            raise

        self.model_name = model_name
        self.log.info(f"Loading Faster Whisper model {self.model_name}")

        # Get models directory in project root
        models_dir = get_models_dir()
        self.log.info(f"Using models directory: {models_dir}")

        # Load the model
        self._model = self._faster_whisper(
            model_size_or_path=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(models_dir),  # Use models directory in project root
            cpu_threads=self.cpu_threads,
        )

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with Faster Whisper using model {self.model_name}")

        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Convert segments to list (it's a generator)
        segments_list = list(segments)
        text = " ".join([segment.text for segment in segments_list])

        return TranscriptionResult(
            text=text,
            segments=segments_list,
            language=info.language,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "cpu_threads": self.cpu_threads,
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed for Faster Whisper
        self._model = None
