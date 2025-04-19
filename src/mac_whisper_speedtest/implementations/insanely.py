"""Insanely Fast Whisper implementation."""

import platform
import tempfile
from typing import Any, Dict

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation


class InsanelyFastWhisperImplementation(WhisperImplementation):
    """Whisper implementation using Insanely Fast Whisper."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.device_id = "mps" if platform.system() == "Darwin" else "cpu"
        self.batch_size = 24
        self.compute_type = "float16"

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            import torch
            from transformers import pipeline
            from transformers.utils import is_flash_attn_2_available
        except ImportError:
            self.log.error("Failed to import required packages. Make sure transformers is installed.")
            raise

        self.model_name = self._map_model_name(model_name)
        self.log.info(f"Loading Insanely Fast Whisper model {self.model_name}")

        # Load the model using transformers pipeline
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        self.log.info(f"Using attention implementation: {attn_implementation}")

        self._model = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch.float16,
            device=self.device_id,
            model_kwargs={"attn_implementation": attn_implementation},
        )

    def _map_model_name(self, model_name: str) -> str:
        """Map the model name to one supported by Insanely Fast Whisper."""
        # Insanely Fast Whisper uses HuggingFace model names
        model_map = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }
        return model_map.get(model_name, f"openai/whisper-{model_name}")

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with Insanely Fast Whisper using model {self.model_name}")

        # Convert numpy array to a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Save the audio data to a temporary file
            sf.write(temp_file.name, audio, 16000, format="WAV")

            # Run transcription using transformers pipeline
            result = self._model(
                temp_file.name,
                chunk_length_s=30,
                batch_size=self.batch_size,
                return_timestamps=False,
            )

        # Extract the text from the output
        text = result.get("text", "")

        # Format segments to match expected output
        segments = []
        if "chunks" in result:
            segments = result["chunks"]
        elif "segments" in result:
            segments = result["segments"]

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=None,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "device_id": self.device_id,
            "batch_size": self.batch_size,
            "compute_type": self.compute_type,
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed
        self._model = None
