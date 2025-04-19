"""Utility functions for the Whisper benchmark tool."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # Start from the current file and go up until we find the project root
    current_path = Path(os.path.abspath(__file__))
    
    # Go up until we find the src directory or reach the filesystem root
    while current_path.name != "src" and current_path != current_path.parent:
        current_path = current_path.parent
    
    # If we found the src directory, go up one more level to get the project root
    if current_path.name == "src":
        return current_path.parent
    
    # Fallback to the current working directory if we couldn't find the project root
    return Path(os.getcwd())


def get_models_dir() -> Path:
    """Get the models directory in the project root.
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Path to the models directory
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir
