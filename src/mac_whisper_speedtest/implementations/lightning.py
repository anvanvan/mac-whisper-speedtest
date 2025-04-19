"""Lightning Whisper MLX implementation."""

import platform
import tempfile
from typing import Any, Dict

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class LightningWhisperMLXImplementation(WhisperImplementation):
    """Whisper implementation using Lightning Whisper MLX."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.batch_size = 24  # Default batch size for transcribe_audio is 6
        self.quant = None # Set to None to disable quantization by default
        self.language = None  # Language code for transcription

        # Check if we're on macOS (MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("LightningWhisperMLX is only supported on macOS with Apple Silicon")

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            # Always use our custom loader
            self._custom_loader = True
            self.log.info("Using custom loader for LightningWhisperMLX")

            # Import our custom loader here to make sure it's available
            try:
                from mac_whisper_speedtest.implementations.lightning_loader import load_model
                self._custom_load_model = load_model
            except Exception as e:
                self.log.error(f"Failed to import custom loader: {e}")
                self.log.info("Falling back to no quantization")
                self.quant = None

            try:
                from lightning_whisper_mlx import LightningWhisperMLX
                self._lightning_whisper_mlx = LightningWhisperMLX
                self._package_name = "lightning_whisper_mlx"
            except ImportError:
                # Try the alternate package name
                from lighting_whisper_mlx import LightningWhisperMLX
                self._lightning_whisper_mlx = LightningWhisperMLX
                self._package_name = "lighting_whisper_mlx"
        except ImportError:
            self.log.error("Failed to import lightning_whisper_mlx. Make sure it's installed.")
            raise

        self.model_name = self._map_model_name(model_name)
        self.log.info(f"Loading LightningWhisperMLX model {self.model_name}")

        # Load the model
        if not self._custom_loader:
            # Use the original implementation
            self._model = self._lightning_whisper_mlx(
                model=self.model_name,
                batch_size=self.batch_size,
                quant=self.quant
            )
        else:
            # Use our custom loader
            import os
            from huggingface_hub import hf_hub_download
            # Import the models and transcribe_audio from the correct package
            if self._package_name == "lightning_whisper_mlx":
                from lightning_whisper_mlx.lightning import models
            else:
                from lighting_whisper_mlx.lightning import models

            # Get the repo ID based on the model name and quantization
            repo_id = ""
            if self.quant and "distil" not in self.model_name:
                repo_id = models[self.model_name][self.quant]
            else:
                repo_id = models[self.model_name]['base']

            # Set up the models directory
            models_dir = get_models_dir()
            model_dir = os.path.join(str(models_dir), f"lightning-{self.model_name}-{self.quant}")
            os.makedirs(model_dir, exist_ok=True)

            # Download the model files
            self.log.info(f"Downloading model {repo_id} to {model_dir}")
            hf_hub_download(repo_id=repo_id, filename="weights.npz", local_dir=model_dir)
            hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=model_dir)

            # Load the model using our custom loader
            if self._package_name == "lightning_whisper_mlx":
                from lightning_whisper_mlx.transcribe import transcribe_audio
                # Import our custom loader
                from mac_whisper_speedtest.implementations.lightning_loader import load_model
            else:
                from lighting_whisper_mlx.transcribe import transcribe_audio
                # Import our custom loader
                from mac_whisper_speedtest.implementations.lightning_loader import load_model

            # Load the model with our custom loader
            self._model_dir = model_dir
            self._transcribe_audio = transcribe_audio

            # Load the model directly using our custom loader
            try:
                if hasattr(self, "_custom_load_model"):
                    self._whisper_model = self._custom_load_model(model_dir)
                    self.log.info("Successfully loaded model with custom loader")
            except Exception as e:
                self.log.error(f"Failed to load model with custom loader: {e}")
                self.log.info("Continuing without custom loader")

    def _map_model_name(self, model_name: str) -> str:
        """Map the model name to one supported by LightningWhisperMLX."""
        # LightningWhisperMLX supports these models:
        # ["tiny", "small", "distil-small.en", "base", "medium", "distil-medium.en",
        #  "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"]

        # Direct mappings
        if model_name in ["tiny", "small", "base", "medium", "large", "large-v2", "large-v3"]:
            return model_name

        # Handle special cases
        if "small" in model_name and ".en" in model_name:
            return "distil-small.en"
        if "medium" in model_name and ".en" in model_name:
            return "distil-medium.en"
        if "large-v2" in model_name and ("distil" in model_name or ".en" in model_name):
            return "distil-large-v2"
        if "large-v3" in model_name and ("distil" in model_name or ".en" in model_name):
            return "distil-large-v3"

        # Default to large-v3 for unknown models
        self.log.warning(f"Unknown model name: {model_name}, defaulting to small")
        return "small"

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model") and not hasattr(self, "_model_dir"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with LightningWhisperMLX using model {self.model_name}")

        # Convert numpy array to a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Save the audio data to a temporary file
            sf.write(temp_file.name, audio, 16000, format="WAV")

            # Run transcription
            if hasattr(self, "_model"):
                # Using original implementation
                result = self._model.transcribe(audio_path=temp_file.name)
            else:
                # Using custom loader
                # The transcribe_audio function expects the first parameter to be audio, not audio_path
                # Pass the correct parameters to transcribe_audio
                result = self._transcribe_audio(
                    audio=temp_file.name,  # First parameter is audio
                    path_or_hf_repo=self._model_dir,
                    batch_size=self.batch_size,
                    temperature=0.0,  # Use deterministic decoding
                    language=self.language  # Pass language if specified
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
        params = {
            "batch_size": self.batch_size,
            "quant": self.quant or "none",
        }

        if self.language:
            params["language"] = self.language

        return params

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # MLX models don't need explicit cleanup
        if hasattr(self, "_model"):
            self._model = None
        if hasattr(self, "_model_dir"):
            self._model_dir = None
            self._transcribe_audio = None
        if hasattr(self, "_whisper_model"):
            self._whisper_model = None
