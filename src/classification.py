from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.loaders import load_samples


def create_pca_svc_pipeline(
    pca_variance: float = 0.99,
    pca_whiten: bool = False,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = False,
    random_state: int = 100,
) -> Pipeline:
    """
    Create a PCA-SVC classification pipeline.

    Parameters
    ----------
    pca_variance : float, default=0.99
        Fraction of explained variance retained by PCA.
    pca_whiten : bool, default=False
        If True, apply PCA whitening.
    C : float, default=1.0
        Regularisation parameter of the support vector classifier.
    kernel : str, default="rbf"
        Kernel type used by the support vector classifier.
    gamma : str, default="scale"
        Kernel coefficient for the RBF kernel.
    probability : bool, default=False
        If True, enable probability estimates.
    random_state : int, default=100
        Random seed used by the support vector classifier.

    Returns
    -------
    Pipeline
        PCA-SVC classification pipeline.
    """

    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    whiten=pca_whiten,
                ),
            ),
            (
                "svc",
                SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    probability=probability,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_logo_predictions(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    pipeline: Pipeline,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Evaluate predictions using leave-one-group-out cross-validation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    groups : array-like of shape (n_samples,)
        Group labels used for leave-one-group-out splitting.
    pipeline : Pipeline
        Classification pipeline.
    n_jobs : int, default=1
        Number of parallel jobs.

    Returns
    -------
    dict[str, Any]
        Dictionary containing predicted labels and the
        classification report.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    logo = LeaveOneGroupOut()

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        groups=groups,
        cv=logo,
        method="predict",
        n_jobs=n_jobs,
    )

    report = classification_report(
        y,
        y_pred,
        zero_division=0,
        digits=4,
    )

    return {
        "y_pred": y_pred,
        "classification_report": report,
    }


def get_logo_pca_component_counts(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    pipeline: Pipeline,
    n_jobs: int = 1,
) -> list[int]:
    """
    Return the number of retained PCA components for each
    leave-one-group-out split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    groups : array-like of shape (n_samples,)
        Group labels used for leave-one-group-out splitting.
    pipeline : Pipeline
        PCA-SVC classification pipeline.
    n_jobs : int, default=1
        Number of parallel jobs.

    Returns
    -------
    list[int]
        Number of retained PCA components for each split.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    logo = LeaveOneGroupOut()

    results = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=logo,
        return_estimator=True,
        n_jobs=n_jobs,
    )

    return [
        estimator.named_steps["pca"].n_components_
        for estimator in results["estimator"]
    ]


def add_sample_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Add sample-level metadata to a spectrum metadata table.

    Parameters
    ----------
    metadata : pd.DataFrame
        Spectrum metadata table.

    Returns
    -------
    pd.DataFrame
        Metadata table enriched with sample information.
    """

    samples = load_samples()[
        [
            "sample_id",
            "display_label",
            "material_group",
            "material_type",
            "polymer_type",
        ]
    ]

    return metadata.merge(
        samples,
        on="sample_id",
        how="left",
    )


def make_class_labels(metadata: pd.DataFrame) -> np.ndarray:
    """
    Create class labels from the display labels.

    Parameters
    ----------
    metadata : pd.DataFrame
        Spectrum metadata table.

    Returns
    -------
    np.ndarray
        Class labels.
    """

    return metadata["display_label"].to_numpy()


def make_polymer_family_labels(metadata: pd.DataFrame) -> np.ndarray:
    """
    Create polymer-family-level labels.

    Plastic samples are grouped by polymer type, while
    non-plastic samples retain their display labels.

    Parameters
    ----------
    metadata : pd.DataFrame
        Spectrum metadata table.

    Returns
    -------
    np.ndarray
        Polymer-family-level labels.
    """

    labels = metadata["display_label"].copy()

    plastic_mask = metadata["material_group"].eq("plastic")

    labels.loc[plastic_mask] = metadata.loc[
        plastic_mask,
        "polymer_type",
    ]

    labels = labels.replace(
        {
            "PE-HD": "PE",
        }
    )

    return labels.to_numpy()


def make_binary_plastic_labels(metadata: pd.DataFrame) -> np.ndarray:
    """
    Create binary plastic versus non-plastic labels.

    Parameters
    ----------
    metadata : pd.DataFrame
        Spectrum metadata table.

    Returns
    -------
    np.ndarray
        Binary class labels.
    """

    return np.where(
        metadata["material_group"].eq("plastic"),
        "Plastic",
        "Non-plastic",
    )


def export_prediction_results(
    metadata: pd.DataFrame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    output_path: Path,
    y_proba: ArrayLike | None = None,
    class_labels: ArrayLike | None = None,
) -> pd.DataFrame:
    """
    Export prediction results together with metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata associated with the spectra.
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    output_path : Path
        Output CSV file path.
    y_proba : array-like of shape (n_samples, n_classes), optional
        Probability matrix.
    class_labels : array-like of shape (n_classes,), optional
        Class labels corresponding to the probability columns.

    Returns
    -------
    pd.DataFrame
        Exported prediction dataframe.

    Raises
    ------
    ValueError
        If probability values are provided without class labels.
    """

    results_df = metadata.copy()

    results_df["y_true"] = np.asarray(y_true)
    results_df["y_pred"] = np.asarray(y_pred)

    if y_proba is not None:
        if class_labels is None:
            raise ValueError(
                "class_labels must be provided when y_proba is used."
            )

        proba_df = pd.DataFrame(
            np.asarray(y_proba),
            columns=[
                f"proba_{label}"
                for label in np.asarray(class_labels)
            ],
        )

        results_df = pd.concat(
            [results_df.reset_index(drop=True), proba_df],
            axis=1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    return results_df


def evaluate_generalisable_batch_effects(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    random_state: int = 100,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Evaluate whether batch effects are consistently detectable
    within each material class.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Material labels.
    groups : array-like of shape (n_samples,)
        Measurement-series labels used as target labels for the
        batch-effect classifier.
    random_state : int, default=100
        Random seed used for cross-validation and classification.
    n_splits : int, default=5
        Maximum number of stratified cross-validation splits.

    Returns
    -------
    pd.DataFrame
        Batch-effect evaluation summary for each material class.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    results = []

    for material in np.unique(y):
        material_mask = y == material

        X_mat = X[material_mask]
        groups_mat = groups[material_mask]

        unique_groups, group_counts = np.unique(
            groups_mat,
            return_counts=True,
        )

        num_groups = len(unique_groups)

        if num_groups <= 1:
            continue

        min_group_count = group_counts.min()
        effective_splits = min(n_splits, min_group_count)

        if effective_splits < 2:
            continue

        group_clf = SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            random_state=random_state,
        )

        cv = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )

        cv_scores = cross_val_score(
            group_clf,
            X_mat,
            groups_mat,
            cv=cv,
            n_jobs=-1,
        )

        results.append(
            {
                "Material": material,
                "Num_Series": num_groups,
                "Random_Baseline": 1.0 / num_groups,
                "Batch_Prediction_Accuracy": cv_scores.mean(),
                "Batch_Prediction_Std": cv_scores.std(ddof=1),
                "Support": len(X_mat),
            }
        )

    report_df = pd.DataFrame(results)

    if not report_df.empty:
        weighted_acc = np.average(
            report_df["Batch_Prediction_Accuracy"],
            weights=report_df["Support"],
        )

        print(
            "Dataset-Wide Batch Prediction Accuracy "
            f"(Unseen Data): {weighted_acc:.4f}\n"
        )

    return report_df