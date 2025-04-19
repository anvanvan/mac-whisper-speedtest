"""WhisperCpp with CoreML implementation."""

import os
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class WhisperCppCoreMLImplementation(WhisperImplementation):
    """Whisper implementation using pywhispercpp with CoreML support."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.models_dir = get_models_dir()  # use models folder in the project root
        self.n_threads = 4
        self.coreml_enabled = False

        # Set environment variable to enable CoreML
        os.environ["WHISPER_COREML"] = "1"

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        import pywhispercpp.model
        self._pywhispercpp = pywhispercpp
        pywhispercpp.model.logging = self.log

        models_map = {
            "tiny": "tiny-q5_1",
            "base": "base-q5_1",
            # "small": "small-q5_1",
            "small": "small",
            "medium": "medium-q5_0",
            "large": "large-v3-turbo-q5_0",
        }
        self.model_name = models_map.get(model_name, model_name)

        # Check if CoreML is enabled
        self.coreml_enabled = os.environ.get("WHISPER_COREML", "0") == "1"
        if self.coreml_enabled:
            self.log.info("CoreML support is enabled for whisper.cpp")

            # Check if CoreML model files exist
            coreml_model_path = self.models_dir / f"ggml-{model_name}-encoder.mlmodelc"
            if coreml_model_path.exists():
                self.log.info(f"Found CoreML model at {coreml_model_path}")
            else:
                self.log.warning(f"CoreML model not found at {coreml_model_path}. Will use CPU fallback.")
                self.coreml_enabled = False

        # Load the model
        self._model = self._pywhispercpp.model.Model(
            self.model_name,
            models_dir=str(self.models_dir),
            n_threads=self.n_threads,
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

        self.log.info(f"Transcribing with whisper.cpp using model {self.model_name}")
        if self.coreml_enabled:
            self.log.info("Using CoreML acceleration")

        segments = self._model.transcribe(audio, language=None)
        text = " ".join([segment.text for segment in segments])

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=None,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "coreml": self.coreml_enabled,
            "n_threads": self.n_threads,
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # Free the model if possible
        if hasattr(self, "_model") and hasattr(self._model, "free"):
            self._model.free()
