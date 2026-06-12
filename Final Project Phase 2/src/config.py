from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLE_RATE = 16000
N_MFCC = 13
TRIM_DB = 20

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

DATASET_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "ML Dataset",
    PROJECT_ROOT / "ML Dataset-20260204T153843Z-3-001" / "ML Dataset",
]


def resolve_dataset_dir(explicit_path=None):
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists():
            return candidate
    for candidate in DATASET_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset directory not found. Set an explicit path or place data in data/raw/ML Dataset."
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
