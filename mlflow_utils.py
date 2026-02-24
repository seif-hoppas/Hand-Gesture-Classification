"""
mlflow_utils.py
---------------
Centralized MLflow tracking helpers for the Hand Gesture Classification project.
All MLflow interactions are routed through the functions below so the notebook
stays clean and experiment logging is consistent.
"""

import os
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def init_experiment(experiment_name: str = "Hand-Gesture-Classification") -> None:
    """Create or retrieve a named MLflow experiment."""
    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def start_run(run_name: str):
    """Start a new MLflow run with a descriptive name."""
    return mlflow.start_run(run_name=run_name)


def end_run() -> None:
    """End the current active MLflow run."""
    mlflow.end_run()


def get_run_id() -> str:
    """Return the run-id of the currently active run."""
    return mlflow.active_run().info.run_id


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_params(params: dict) -> None:
    """Log a dictionary of parameters to the active run."""
    mlflow.log_params(params)


def log_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics, log them, and return the dict."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    mlflow.log_metrics(metrics)
    return metrics


def log_dataset_info(df: pd.DataFrame, csv_path: str) -> None:
    """Log dataset metadata and the CSV file itself as an artifact."""
    info = {
        "source_file": csv_path,
        "num_samples": int(len(df)),
        "num_features": int(df.shape[1] - 1),
        "labels": sorted(df["label"].unique().tolist()),
        "num_classes": int(df["label"].nunique()),
    }

    os.makedirs("artifacts", exist_ok=True)
    info_path = os.path.join("artifacts", "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    mlflow.log_artifact(info_path)
    mlflow.log_artifact(csv_path)
    mlflow.log_param("dataset_samples", info["num_samples"])
    mlflow.log_param("dataset_features", info["num_features"])
    mlflow.log_param("dataset_classes", info["num_classes"])


def log_model(model, artifact_path: str) -> None:
    """Log a scikit-learn compatible model to the active run."""
    mlflow.sklearn.log_model(model, artifact_path)


def log_figure(fig, filename: str) -> None:
    """Save a matplotlib figure and log it as an artifact."""
    os.makedirs("artifacts", exist_ok=True)
    filepath = os.path.join("artifacts", filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    mlflow.log_artifact(filepath)


def log_artifact(filepath: str) -> None:
    """Log an arbitrary file as an artifact."""
    mlflow.log_artifact(filepath)


def log_confusion_matrix(y_true, y_pred, labels, filename: str = "confusion_matrix.png"):
    """Create, save, and log a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    log_figure(fig, filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def register_model(run_id: str, artifact_path: str, registry_name: str):
    """Register a logged model in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, registry_name)
    return result


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Convert a {model_name: {metric: value}} dict into a DataFrame."""
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df


def plot_comparison_chart(results_df: pd.DataFrame, filename: str = "model_comparison.png"):
    """Bar chart comparing models across all logged metrics."""
    ax = results_df.plot(kind="bar", figsize=(12, 6), rot=30)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig = ax.get_figure()
    log_figure(fig, filename)
    plt.close(fig)
    return fig
