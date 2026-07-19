"""Shared project paths used across the dehazer package."""

from pathlib import Path

# Repository root (two levels up: dehazer/config.py -> dehazer/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output directory for dehazed images
OUTPUT_DIR = PROJECT_ROOT / "seriespicturesoutput"


def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
