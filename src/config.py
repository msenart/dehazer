from pathlib import Path

# root library path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# output directory for dehazed images
OUTPUT_DIR = PROJECT_ROOT / "seriespicturesoutput"


def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
