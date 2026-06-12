import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SPEAKER_ID_REGEX = re.compile(r"(\d+)_")
_LEAKAGE_COL_PATTERN = re.compile(r"(^|_)(language|path|speaker|group|label|target|book|source)(_|$)")
_GENERIC_GROUP_FOLDERS = {
    "audio",
    "audios",
    "clip",
    "clips",
    "data",
    "dataset",
    "datasets",
    "female",
    "female90",
    "file",
    "files",
    "flac",
    "m",
    "male",
    "male90",
    "mp3",
    "ogg",
    "raw",
    "test",
    "train",
    "val",
    "validation",
    "wav",
}


def _is_usable_group_folder(folder_name):
    if not folder_name:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "", str(folder_name).strip().lower())
    if not normalized:
        return False
    if normalized.endswith("dataset"):
        return False
    return normalized not in _GENERIC_GROUP_FOLDERS


def _select_feature_matrix(df, label_col="language", path_col="path", group_col=None):
    """
    Build leakage-safe feature matrix:
    - remove label/path/group and obvious leakage-named columns
    - keep only numeric features
    """
    drop_cols = {label_col, path_col}
    if group_col:
        drop_cols.add(group_col)

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in {"id"} or col_lower.endswith("_id"):
            drop_cols.add(col)
            continue
        if _LEAKAGE_COL_PATTERN.search(col_lower):
            drop_cols.add(col)

    X = df.drop(columns=sorted(drop_cols), errors="ignore")
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols, errors="ignore")
    if X.empty:
        raise ValueError("No numeric feature columns remain after leakage-safe filtering.")
    return X


def split_data(df, label_col="language", test_size=0.2, random_state=42, group_col=None):
    X = _select_feature_matrix(df, label_col=label_col, path_col="path", group_col=group_col)
    y = df[label_col]
    if group_col and group_col in df.columns:
        groups = df[group_col]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def speaker_group_from_path(path_value):
    """
    Build a group key as '<language>/<speaker_or_book_id>'.

    Prefer folder-level grouping from '<language>/<speaker_or_book_folder>/<file>'.
    If folder-level ID is missing/generic, fall back to filename-based ID extraction.
    """
    path = Path(str(path_value))
    folder_candidate = path.parent.name if path.parent else ""
    language_candidate = path.parent.parent.name if len(path.parents) >= 2 else ""

    if _is_usable_group_folder(folder_candidate) and _is_usable_group_folder(language_candidate):
        language = language_candidate
        speaker_id = folder_candidate
    else:
        match = _SPEAKER_ID_REGEX.match(path.name)
        speaker_id = match.group(1) if match else path.stem.split("_")[0]
        if _is_usable_group_folder(folder_candidate):
            language = folder_candidate
        elif _is_usable_group_folder(language_candidate):
            language = language_candidate
        elif len(path.parents) >= 1:
            language = path.parent.name
        else:
            language = "unknown"
    return f"{language}/{speaker_id}"


def split_data_group_stratified(
    df,
    label_col="language",
    path_col="path",
    group_col="speaker_group",
    test_size=0.2,
    random_state=42,
):
    """
    Split data per label while keeping groups disjoint across train/test.

    This avoids leakage from the same speaker appearing in both sets and
    preserves class balance by selecting test groups independently per label.
    """
    work_df = df.copy()
    if group_col not in work_df.columns:
        if path_col not in work_df.columns:
            raise ValueError(f"'{group_col}' not found and '{path_col}' column is missing.")
        work_df[group_col] = work_df[path_col].apply(speaker_group_from_path)

    train_idx = []
    test_idx = []
    rng = np.random.default_rng(random_state)

    for label in sorted(work_df[label_col].unique()):
        label_rows = work_df[work_df[label_col] == label]
        groups = np.array(sorted(label_rows[group_col].unique()))
        if len(groups) < 2:
            raise ValueError(
                f"Need at least 2 groups for label '{label}' to do group-aware splitting."
            )

        n_test_groups = max(1, int(round(len(groups) * test_size)))
        n_test_groups = min(n_test_groups, len(groups) - 1)
        test_groups = rng.choice(groups, size=n_test_groups, replace=False)

        label_test_mask = (work_df[label_col] == label) & (work_df[group_col].isin(test_groups))
        test_idx.extend(work_df.index[label_test_mask])
        train_idx.extend(work_df.index[(work_df[label_col] == label) & (~label_test_mask)])

    X = _select_feature_matrix(work_df, label_col=label_col, path_col=path_col, group_col=group_col)
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = work_df[label_col].loc[train_idx]
    y_test = work_df[label_col].loc[test_idx]

    return X_train, X_test, y_train, y_test


def _build_pipeline(model, use_pca=False, pca_components=0.95, random_state=42):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)


def build_classification_models(use_pca=False, pca_components=0.95, random_state=42):
    models = {
        "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
        "RandomForest": None,
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    models["RandomForest"] = (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        if not use_pca
        else _build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
            ),
            use_pca=use_pca,
            pca_components=pca_components,
            random_state=random_state,
        )
    )

    pipelines = {}
    for name, model in models.items():
        if isinstance(model, Pipeline):
            pipelines[name] = model
        else:
            pipelines[name] = _build_pipeline(
                model,
                use_pca=use_pca,
                pca_components=pca_components,
                random_state=random_state,
            )
    return pipelines


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    metrics = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics[name] = report
    return metrics, predictions


def summarize_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_confusion(y_true, y_pred, labels, title=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def kmeans_silhouette_scores(X, k_values, random_state=42):
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores


def fit_kmeans(X, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def fit_agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels


def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def cluster_purity(y_true, cluster_labels):
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    total = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        labels, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return total / len(y_true)


def reduce_to_2d(X, method="pca", random_state=42):
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
        return reducer.fit_transform(X)

    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(X)


def plot_clusters(points_2d, labels, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=points_2d[:, 0], y=points_2d[:, 1], hue=labels, ax=ax, palette="tab10")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    ax.legend(title="Cluster", bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax
