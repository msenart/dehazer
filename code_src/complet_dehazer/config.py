import os
from pathlib import Path

# root library path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# output directory for dehazed images
OUTPUT_DIR = PROJECT_ROOT / "dehazer" / "seriespicturesoutput"

# create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)