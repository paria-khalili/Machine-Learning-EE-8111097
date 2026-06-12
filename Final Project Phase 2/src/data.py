from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .config import AUDIO_EXTS, DEFAULT_SAMPLE_RATE, N_MFCC, TRIM_DB, ensure_dir


def find_audio_files(dataset_dir):
    dataset_dir = Path(dataset_dir)
    language_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    records = []
    for lang_dir in language_dirs:
        for path in lang_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                records.append({"path": path, "language": lang_dir.name})
    return records


def load_audio(path, target_sr=DEFAULT_SAMPLE_RATE, mono=True):
    y, sr = librosa.load(path, sr=target_sr, mono=mono)
    return y, sr


def clean_audio(y, sr, trim_db=TRIM_DB):
    if y.size == 0:
        return y
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    return y


def _stats_from_vector(prefix, vec):
    return {
        f"{prefix}_mean": float(np.mean(vec)),
        f"{prefix}_std": float(np.std(vec)),
    }


def _stats_from_matrix(prefix, mat):
    stats = {}
    for idx, row in enumerate(mat, start=1):
        stats[f"{prefix}{idx}_mean"] = float(np.mean(row))
        stats[f"{prefix}{idx}_std"] = float(np.std(row))
    return stats


def extract_features(y, sr, n_mfcc=N_MFCC):
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.update(_stats_from_matrix("mfcc_", mfcc))

    delta = librosa.feature.delta(mfcc)
    features.update(_stats_from_matrix("mfcc_delta_", delta))

    delta2 = librosa.feature.delta(mfcc, order=2)
    features.update(_stats_from_matrix("mfcc_delta2_", delta2))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.update(_stats_from_vector("zcr", zcr))

    rms = librosa.feature.rms(y=y)
    features.update(_stats_from_vector("rms", rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.update(_stats_from_vector("spec_centroid", centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.update(_stats_from_vector("spec_bandwidth", bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.update(_stats_from_vector("spec_rolloff", rolloff))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.update(_stats_from_matrix("spec_contrast_", contrast))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update(_stats_from_matrix("chroma_", chroma))

    return features


def build_feature_dataframe(
    dataset_dir,
    cache_path=None,
    limit=None,
    verbose=True,
):
    records = find_audio_files(dataset_dir)
    if limit:
        records = records[:limit]

    rows = []
    failures = []

    for idx, record in enumerate(records, start=1):
        path = record["path"]
        language = record["language"]
        try:
            y, sr = load_audio(path)
            y = clean_audio(y, sr)
            if y.size == 0:
                raise ValueError("empty audio after cleaning")
            feats = extract_features(y, sr)
            feats["path"] = str(path)
            feats["language"] = language
            rows.append(feats)
        except Exception as exc:
            failures.append((str(path), str(exc)))

        if verbose and idx % 100 == 0:
            print(f"Processed {idx}/{len(records)} files")

    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["path", "language"] + [c for c in df.columns if c not in {"path", "language"}]
        df = df[cols]

    if cache_path:
        cache_path = Path(cache_path)
        ensure_dir(cache_path.parent)
        df.to_csv(cache_path, index=False)

    return df, failures


def summarize_by_language(df):
    return df["language"].value_counts().sort_index()
