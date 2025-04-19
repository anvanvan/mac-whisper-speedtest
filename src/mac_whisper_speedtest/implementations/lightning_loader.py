"""Custom loader for Lightning Whisper MLX models."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

# Import the whisper module from lightning_whisper_mlx
try:
    # Try both package names
    try:
        from lightning_whisper_mlx import whisper
    except ImportError:
        from lighting_whisper_mlx import whisper
except Exception as e:
    # If we can't import whisper, try to import it from mlx_whisper
    # This is a fallback in case the lightning_whisper_mlx package is not compatible
    from mlx_whisper import whisper


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    """Load a Whisper model from a local path or Hugging Face repo.

    This is a fixed version of the loader that uses nn.quantize instead of
    QuantizedLinear.quantize_module.

    Args:
        path_or_hf_repo: Local path or Hugging Face repo ID
        dtype: Data type for the model

    Returns:
        Loaded Whisper model
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        try:
            # Use nn.quantize instead of QuantizedLinear.quantize_module
            class_predicate = (
                lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                and f"{p}.scales" in weights
            )
            nn.quantize(model, **quantization, class_predicate=class_predicate)
        except Exception as e:
            # If quantization fails, log the error and continue without quantization
            print(f"Warning: Quantization failed: {e}")
            print("Continuing without quantization.")

    model.update(weights)
    mx.eval(model.parameters())
    return model
