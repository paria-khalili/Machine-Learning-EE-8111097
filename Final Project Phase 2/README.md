# ML Phase 2 - Multilingual Speech Classification

This workspace completes the Phase 2 pipeline: data cleaning, feature extraction, classification, clustering, and evaluation. The detailed process write-up lives in `reports/README.md`.

## Structure

- `Data_Cleaning_and_Feature_Extraction.ipynb`
- `Classification.ipynb`
- `Clustering.ipynb`
- `Evaluation.ipynb`
- `src/config.py` (paths and constants)
- `src/data.py` (loading, cleaning, feature extraction)
- `src/modeling.py` (classification, clustering, metrics, plots)
- `data/raw/` (raw dataset or symlink to it)
- `data/processed/` (features and metrics outputs)
- `reports/README.md` (report narrative)
- `reports/figures/` (saved plots)
- `requirements.txt`

## Process overview

1. Data cleaning
   - Resample audio to 16 kHz, convert to mono, normalize amplitude, and trim leading/trailing silence.
2. Feature extraction
   - MFCCs (with deltas), spectral centroid, bandwidth, rolloff, RMS, zero crossing rate, spectral contrast, and chroma.
   - Aggregate with mean and standard deviation to produce fixed-length vectors.
3. Classification
   - Stratified 80/20 split.
   - Standardization applied to features.
   - Models: SVM (RBF), Random Forest, Logistic Regression, KNN.
4. Clustering
   - KMeans and GMM (plus Agglomerative as a second perspective).
   - Silhouette scores used to pick k.
   - PCA visualization saved to `reports/figures/`.
5. Evaluation
   - Accuracy, precision, recall, and F1 (macro and weighted).
   - Confusion matrices saved for each classifier.
   - Silhouette score and cluster purity for clustering.

## How to run

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure the dataset is available. The loader checks these locations (in order):
   - `data/raw/ML Dataset`
   - `ML Dataset-20260204T153843Z-3-001/ML Dataset`

3. Run the notebooks in order:
   1) `Data_Cleaning_and_Feature_Extraction.ipynb`
   2) `Classification.ipynb`
   3) `Clustering.ipynb`
   4) `Evaluation.ipynb`

Outputs are written to `data/processed/` and plots to `reports/figures/`.

## Notes

- MP3 decoding uses `ffmpeg` via `librosa`.
- If you re-run feature extraction, delete `data/processed/features.csv` first to avoid stale data.
- The notebooks are ready to execute but are not run in this environment; run them locally to populate metrics, plots, and final outputs.
