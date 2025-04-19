# Whisper Benchmark for Apple Silicon

A benchmarking tool to compare different Whisper implementations optimized for Apple Silicon, focusing on speed while maintaining accuracy.

## Example output

```
=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
whisper.cpp            0.4762          coreml=True, n_threads=4
mlx-whisper            0.4897          model=mlx-community/whisper-small-mlx-q4
insanely-fast-whisper  0.8733          device_id=mps, batch_size=24, compute_type=float16
lightning-whisper-mlx  1.1470          batch_size=24, quant=none
faster-whisper         2.2725          device=cpu, compute_type=int8, beam_size=1, cpu_threads=4
```

## Overview

This tool measures transcription performance across different implementations of the same base model (e.g., all variants of "small"). It helps you find the fastest Whisper implementation on your Apple Silicon Mac for a given base model.

## Features

- Input: Live speech recording
- Base model selection (tiny, small, base, medium, large)
- Automatic testing of all available optimized variants
- Consistent audio preprocessing across all implementations
- Console output with timing and variant-specific parameters

## Implementations Tested

1. **pywhispercpp + CoreML**

   - Source: https://github.com/abdeladim-s/pywhispercpp
   - Key params: WHISPER_COREML=1

2. **faster-whisper**

   - Source: https://github.com/SYSTRAN/faster-whisper
   - Key params: compute_type="float16"

3. **insanely-fast-whisper**

   - Source: https://github.com/Vaibhavs10/insanely-fast-whisper
   - Key params: device_id="mps"

4. **mlx-whisper**

   - Source: https://github.com/ml-explore/mlx-examples
   - Key params: model_size="small"

5. **lightning-whisper-mlx**
   - Source: https://github.com/lightning-AI/lightning-whisper
   - Key params: quant="4bit", model="mall"

## Installation

```bash
# Clone the repository
git clone https://github.com//mac-whisper-speedtest.git
cd mac-whisper-speedtest

# Install dependencies
uv sync
```

## Usage

```bash
# Run benchmark with default settings (small model)
.venv/bin/mac-whisper-speedtest

# Run benchmark with a specific model
.venv/bin/mac-whisper-speedtest --model small

# Run benchmark with specific implementations
.venv/bin/mac-whisper-speedtest --model small --implementations "WhisperCppCoreMLImplementation,LightningWhisperMLXImplementation"

# Run benchmark with a specific number of runs per implementation
.venv/bin/mac-whisper-speedtest --model small --num-runs 5
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- PyAudio and its dependencies
- Various Whisper implementations (installed automatically)

## Project Structure

```
mac-whisper-speedtest/
├── pyproject.toml
├── src/
│   └── mac_whisper_speedtest/
│       ├── __init__.py
│       ├── audio.py           # Audio recording/processing
│       ├── benchmark.py       # Core benchmarking logic
│       ├── implementations/   # Individual impl. wrappers
│       │   ├── __init__.py    # Implementation registry
│       │   ├── base.py        # Abstract base class
│       │   ├── coreml.py      # WhisperCpp with CoreML
│       │   ├── faster.py      # Faster Whisper
│       │   ├── insanely.py    # Insanely Fast Whisper
│       │   ├── mlx.py         # MLX Whisper
│       │   └── lightning.py   # Lightning Whisper MLX
│       └── cli.py             # Command line interface
└── README.md
```

## License

MIT
