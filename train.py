"""
train.py
--------
Standalone training pipeline for the Hand Gesture Classification project.
Mirrors the notebook workflow and logs every step with MLflow via mlflow_utils.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import loguniform, randint
import joblib

import mlflow_utils as mlu


# ── 1. Load data ──────────────────────────────────────────────────────────────

DATA_PATH = "hand_landmarks_data.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples, {df.shape[1]-1} features, {df['label'].nunique()} classes")


# ── 2. MLflow experiment & dataset logging ────────────────────────────────────

mlu.init_experiment("Hand-Gesture-Classification")

with mlu.start_run("dataset-logging"):
    mlu.log_dataset_info(df, DATA_PATH)
    mlu.end_run()


# ── 3. Preprocessing ─────────────────────────────────────────────────────────

df_processed = df.copy()
wrist_x = df_processed["x1"].copy()
wrist_y = df_processed["y1"].copy()

for i in range(1, 22):
    df_processed[f"x{i}"] -= wrist_x
    df_processed[f"y{i}"] -= wrist_y

mid_finger_dist = np.sqrt(df_processed["x13"] ** 2 + df_processed["y13"] ** 2)
for i in range(1, 22):
    df_processed[f"x{i}"] /= mid_finger_dist
    df_processed[f"y{i}"] /= mid_finger_dist

y = df_processed["label"]
X = df_processed.drop(columns=["label"])
labels = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ── 4. Baseline models ───────────────────────────────────────────────────────

baseline_models = {
    "Logistic-Regression-Baseline": LogisticRegression(max_iter=1000),
    "SVM-Baseline": SVC(),
    "KNN-Baseline": KNeighborsClassifier(),
    "Random-Forest-Baseline": RandomForestClassifier(),
}

baseline_results = {}

for name, model in baseline_models.items():
    with mlu.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mlu.log_params(model.get_params())
        metrics = mlu.log_metrics(y_test, y_pred)
        mlu.log_model(model, artifact_path="model")
        mlu.log_confusion_matrix(y_test, y_pred, labels=labels, filename=f"{name}_confusion_matrix.png")
        baseline_results[name] = metrics
        mlu.end_run()
    print(f"{name}: accuracy={metrics['accuracy']:.4f}")


# ── 5. Hyperparameter tuning ─────────────────────────────────────────────────

tuned_results = {}

# SVM
with mlu.start_run(run_name="SVM-Tuned"):
    svm_search = RandomizedSearchCV(
        SVC(class_weight="balanced"),
        param_distributions={"kernel": ["rbf"], "C": loguniform(1e-3, 1e3), "gamma": loguniform(1e-4, 1e1)},
        n_iter=40, cv=5, random_state=42,
    )
    svm_search.fit(X_train, y_train)
    y_pred_svm = svm_search.predict(X_test)
    mlu.log_params(svm_search.best_params_)
    tuned_results["SVM-Tuned"] = mlu.log_metrics(y_test, y_pred_svm)
    mlu.log_model(svm_search.best_estimator_, artifact_path="model")
    mlu.log_confusion_matrix(y_test, y_pred_svm, labels=labels, filename="SVM-Tuned_confusion_matrix.png")
    svm_run_id = mlu.get_run_id()
    mlu.end_run()
print(f"SVM-Tuned: {svm_search.best_params_}, accuracy={tuned_results['SVM-Tuned']['accuracy']:.4f}")

# KNN
with mlu.start_run(run_name="KNN-Tuned"):
    knn_search = RandomizedSearchCV(
        KNeighborsClassifier(),
        param_distributions={"n_neighbors": randint(1, 30), "weights": ["uniform", "distance"], "p": [1, 2]},
        n_iter=30, cv=5, random_state=42,
    )
    knn_search.fit(X_train, y_train)
    y_pred_knn = knn_search.predict(X_test)
    mlu.log_params(knn_search.best_params_)
    tuned_results["KNN-Tuned"] = mlu.log_metrics(y_test, y_pred_knn)
    mlu.log_model(knn_search.best_estimator_, artifact_path="model")
    mlu.log_confusion_matrix(y_test, y_pred_knn, labels=labels, filename="KNN-Tuned_confusion_matrix.png")
    mlu.end_run()
print(f"KNN-Tuned: {knn_search.best_params_}, accuracy={tuned_results['KNN-Tuned']['accuracy']:.4f}")

# Random Forest
with mlu.start_run(run_name="Random-Forest-Tuned"):
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions={
            "n_estimators": randint(50, 500), "max_depth": randint(2, 20),
            "min_samples_split": randint(2, 20), "min_samples_leaf": randint(1, 20),
            "bootstrap": [True, False],
        },
        n_iter=30, cv=5, random_state=42,
    )
    rf_search.fit(X_train, y_train)
    y_pred_rf = rf_search.predict(X_test)
    mlu.log_params(rf_search.best_params_)
    tuned_results["Random-Forest-Tuned"] = mlu.log_metrics(y_test, y_pred_rf)
    mlu.log_model(rf_search.best_estimator_, artifact_path="model")
    mlu.log_confusion_matrix(y_test, y_pred_rf, labels=labels, filename="Random-Forest-Tuned_confusion_matrix.png")
    mlu.end_run()
print(f"Random-Forest-Tuned: {rf_search.best_params_}, accuracy={tuned_results['Random-Forest-Tuned']['accuracy']:.4f}")


# ── 6. Comparison ────────────────────────────────────────────────────────────

all_results = {**baseline_results, **tuned_results}
comparison_df = mlu.build_comparison_table(all_results)
print("\n=== Model Comparison ===")
print(comparison_df.to_string())

with mlu.start_run(run_name="model-comparison-chart"):
    mlu.plot_comparison_chart(comparison_df, filename="model_comparison.png")
    mlu.end_run()


# ── 7. Register best model ───────────────────────────────────────────────────

result = mlu.register_model(
    run_id=svm_run_id,
    artifact_path="model",
    registry_name="Hand-Gesture-SVM-Champion",
)
print(f"\nRegistered model: {result.name}, version: {result.version}")


# ── 8. Save locally ──────────────────────────────────────────────────────────

joblib.dump(svm_search, "svm_winner.pkl")
print("Best model saved to svm_winner.pkl")
