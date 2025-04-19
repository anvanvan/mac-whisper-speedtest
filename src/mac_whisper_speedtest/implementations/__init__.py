"""Implementation registry for Whisper implementations."""

import importlib.util
import logging
from typing import List, Type

from mac_whisper_speedtest.implementations.base import WhisperImplementation
from mac_whisper_speedtest.implementations.faster import FasterWhisperImplementation
from mac_whisper_speedtest.implementations.mlx import MLXWhisperImplementation

logger = logging.getLogger(__name__)

# Conditionally import implementations based on available packages
available_implementations = [
    FasterWhisperImplementation,
    MLXWhisperImplementation,
]

# Try to import WhisperCppCoreMLImplementation
try:
    if importlib.util.find_spec("pywhispercpp"):
        from mac_whisper_speedtest.implementations.coreml import WhisperCppCoreMLImplementation
        available_implementations.append(WhisperCppCoreMLImplementation)
    else:
        logger.warning("pywhispercpp not found, WhisperCppCoreMLImplementation will not be available")
except ImportError:
    logger.warning("Failed to import WhisperCppCoreMLImplementation")

# Try to import InsanelyFastWhisperImplementation
try:
    if importlib.util.find_spec("insanely_fast_whisper"):
        from mac_whisper_speedtest.implementations.insanely import InsanelyFastWhisperImplementation
        available_implementations.append(InsanelyFastWhisperImplementation)
    else:
        logger.warning("insanely-fast-whisper not found, InsanelyFastWhisperImplementation will not be available")
except ImportError:
    logger.warning("Failed to import InsanelyFastWhisperImplementation")

# Try to import LightningWhisperMLXImplementation
try:
    if importlib.util.find_spec("lightning_whisper_mlx"):
        from mac_whisper_speedtest.implementations.lightning import LightningWhisperMLXImplementation
        available_implementations.append(LightningWhisperMLXImplementation)
    else:
        logger.warning("lightning-whisper-mlx not found, LightningWhisperMLXImplementation will not be available")
except ImportError:
    logger.warning("Failed to import LightningWhisperMLXImplementation")


def get_all_implementations() -> List[Type[WhisperImplementation]]:
    """Get all available Whisper implementations.

    Returns:
        List of WhisperImplementation classes
    """
    return available_implementations


def get_implementation_by_name(name: str) -> Type[WhisperImplementation]:
    """Get a Whisper implementation by name.

    Args:
        name: The name of the implementation

    Returns:
        The WhisperImplementation class

    Raises:
        ValueError: If the implementation is not found
    """
    for impl in get_all_implementations():
        if impl.__name__ == name:
            return impl

    raise ValueError(f"Implementation not found: {name}")
