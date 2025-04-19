"""Benchmark runner for Whisper implementations."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import BenchmarkResult, WhisperImplementation

log = structlog.get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str
    implementations: List[Type[WhisperImplementation]]
    num_runs: int = 3
    audio_data: np.ndarray = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    model_name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def print_summary(self):
        """Print a summary of the benchmark results."""
        print(f"\n=== Benchmark Summary for '{self.model_name}' model ===\n")
        print(f"{'Implementation':<22} {'Avg Time (s)':<15} {'Parameters'}")
        print("-" * 80)

        # Sort results by average time
        sorted_results = sorted(
            self.results,
            key=lambda r: r.transcription_time
        )

        # Map implementation names to shorter versions
        name_map = {
            "WhisperCppCoreMLImplementation": "whisper.cpp",
            "MLXWhisperImplementation": "mlx-whisper",
            "InsanelyFastWhisperImplementation": "insanely-fast-whisper",
            "LightningWhisperMLXImplementation": "lightning-whisper-mlx",
            "FasterWhisperImplementation": "faster-whisper"
        }

        for result in sorted_results:
            # Use the short name if available, otherwise use the original
            short_name = name_map.get(result.implementation, result.implementation)
            params_str = ", ".join([f"{k}={v}" for k, v in result.model_params.items()])
            print(f"{short_name:<22} {result.transcription_time:<15.4f} {params_str}")


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkSummary:
    """Run the benchmark with the given configuration.

    Args:
        config: The benchmark configuration

    Returns:
        A summary of the benchmark results
    """
    summary = BenchmarkSummary(model_name=config.model_name)

    for impl_class in config.implementations:
        impl_name = impl_class.__name__
        log.info(f"Benchmarking {impl_name} with model {config.model_name}")

        try:
            # Create implementation instance
            implementation = impl_class()

            # Load the model (not timed)
            log.info(f"Loading model for {impl_name}")
            implementation.load_model(config.model_name)

            # Run multiple times and average
            total_time = 0.0
            for run in range(config.num_runs):
                log.info(f"Run {run+1}/{config.num_runs} for {impl_name}")

                # Time the transcription
                start_time = time.time()
                result = await implementation.transcribe(config.audio_data)
                end_time = time.time()

                run_time = end_time - start_time
                total_time += run_time

                log.info(f"Run {run+1} completed in {run_time:.4f} seconds")
                log.info(f"Transcription: {result.text[:50]}...")

            # Calculate average time
            avg_time = total_time / config.num_runs

            # Add result to summary
            summary.results.append(BenchmarkResult(
                implementation=impl_name,
                model_name=config.model_name,
                model_params=implementation.get_params(),
                transcription_time=avg_time,
                text=result.text
            ))

            log.info(f"Average time for {impl_name}: {avg_time:.4f} seconds")

            # Clean up
            if hasattr(implementation, 'cleanup') and callable(implementation.cleanup):
                implementation.cleanup()

        except Exception as e:
            log.error(f"Error benchmarking {impl_name}: {e}", exc_info=True)
            # Add failed result
            summary.results.append(BenchmarkResult(
                implementation=impl_name,
                model_name=config.model_name,
                model_params={"error": str(e)},
                transcription_time=float('inf'),
                text=""
            ))

    return summary
