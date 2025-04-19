"""Audio recording and processing utilities."""

import asyncio
import numpy as np
import pyaudio
import structlog

log = structlog.get_logger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2


def to_whisper_ndarray(frames, *, sample_rate, channels, sample_width):
    """Convert raw audio frames to the format expected by Whisper models."""
    assert (sample_rate, channels, sample_width) == (16000, 1, 2), "16kHz 16bit mono"
    return (
        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        / np.iinfo(np.int16).max
    )


def get_default_device() -> tuple[int, str]:
    """Retrieve the default input sound device index and name."""
    p = pyaudio.PyAudio()
    try:
        val = p.get_default_input_device_info()
        return int(val["index"]), str(val["name"])
    finally:
        p.terminate()
    # This line should never be reached, but just in case
    return None, "default"


async def record_audio(
    stop_event,
    channels=1,
    sample_rate=SAMPLE_RATE,
    format=pyaudio.paInt16,
    convert=to_whisper_ndarray,
    device=None,
):
    """Record audio until the stop event is set.

    Args:
        stop_event: An asyncio.Event that signals when to stop recording
        channels: Number of audio channels (1 for mono, 2 for stereo)
        sample_rate: Audio sample rate in Hz
        format: PyAudio format constant
        convert: Function to convert raw audio data to desired format
        device: Input device index

    Returns:
        Converted audio data (format depends on the convert function)
    """
    frames_per_buffer = 1024
    p = pyaudio.PyAudio()
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        frames_per_buffer=frames_per_buffer,
        input=True,
        input_device_index=device,
    )
    try:
        frames = []
        log.info("Recording started, waiting for stop event")

        # Record until the stop event is set
        while not stop_event.is_set():
            data = stream.read(frames_per_buffer)
            frames.append(data)
            await asyncio.sleep(0.0)

        log.info("Recording stopped")
        samples = b"".join(frames)

        # Check if recording is too short (less than 1 second)
        if len(samples) / sample_rate / SAMPLE_WIDTH < 1.0:
            raise ValueError("Recording too short (less than 1 second)")

        return convert(
            samples,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=p.get_sample_size(format),
        )
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
