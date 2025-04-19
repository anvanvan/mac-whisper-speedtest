"""MLX Whisper implementation."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class MLXWhisperImplementation(WhisperImplementation):
    """Whisper implementation using MLX Whisper."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.language = None

        # Check if we're on macOS (MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("MLX is only supported on macOS with Apple Silicon")

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            # Import mlx-whisper package
            try:
                from huggingface_hub import snapshot_download
                from mlx_whisper.load_models import load_model
            except ImportError:
                self.log.error("Failed to import mlx-whisper. Make sure it's installed.")
                raise
        except ImportError:
            self.log.error("Failed to import MLX. Make sure it's installed.")
            raise

        self.model_name = model_name
        self.log.info(f"Loading MLX Whisper model {self.model_name}")

        # Map model name to the format expected by mlx-whisper
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx-q4",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-turbo"
        }

        # Get the appropriate model path
        self.hf_repo = model_map.get(self.model_name, self.model_name)

        # Get the models directory from the utility function
        models_dir = str(get_models_dir())

        # Download the model to the models directory
        self.log.info(f"Downloading model {self.hf_repo} to {models_dir}")
        model_path = snapshot_download(
            repo_id=self.hf_repo,
            cache_dir=models_dir,
        )

        # Load the model
        self.log.info(f"Loading model from {model_path}")
        self._model = load_model(model_path)
        self._model_path = model_path

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with MLX Whisper using model {self.model_name}")

        # Import here to avoid circular imports
        from mlx_whisper import transcribe

        # Run transcription directly on the audio array
        # mlx_whisper can handle numpy arrays directly
        result = transcribe(
            audio=audio,
            path_or_hf_repo=self._model_path,
            temperature=0.0,  # Use deterministic decoding
            language=self.language
        )

        # Extract the text from the output
        text = result.get("text", "")

        return TranscriptionResult(
            text=text,
            segments=result.get("segments", []),
            language=result.get("language"),
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self.hf_repo,
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed for MLX
        self._model = None
        self._model_path = None
