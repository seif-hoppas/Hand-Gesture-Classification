# Hand Gesture Classification

Classify hand gestures from MediaPipe hand-landmark coordinates using classical ML models. The project covers data exploration, preprocessing, model comparison, hyperparameter tuning, MLflow experiment tracking, and real-time webcam inference.

## Overview

1. **Data** -- `hand_landmarks_data.csv` contains (x, y, z) coordinates for 21 hand landmarks per sample, along with a gesture label.
2. **Preprocessing** -- Landmarks are normalized relative to the wrist (landmark 1) and scaled by the distance to the middle-finger tip (landmark 13).
3. **Model training** -- Four classifiers are compared: Logistic Regression, SVM, KNN, and Random Forest. RandomizedSearchCV is used for hyperparameter tuning on SVM, KNN, and Random Forest.
4. **Evaluation** -- Models are evaluated with accuracy, precision, recall, and F1-score.
5. **Experiment tracking** -- Every run is logged to MLflow with parameters, metrics, models, confusion matrices, and the dataset. A comparison chart is generated and the best model is registered in the MLflow Model Registry.
6. **Real-time inference** -- The best model (SVM) is saved with joblib and used for live gesture recognition through a webcam via OpenCV and MediaPipe.

## MLflow Integration

All MLflow logic lives in `mlflow_utils.py`. The notebook and `train.py` import it to:

- Create a named experiment (`Hand-Gesture-Classification`).
- Log the dataset CSV and metadata as artifacts.
- Open a separate run for each model variant (baseline and tuned) with descriptive run names.
- Log hyperparameters, weighted classification metrics, confusion matrix plots, and the trained model artifact for every run.
- Generate a bar chart comparing all models and log it as an artifact.
- Register the champion model under `Hand-Gesture-SVM-Champion` in the Model Registry.

To launch the MLflow UI and inspect the runs:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in a browser.

Screenshots of the MLflow UI (runs, charts, model registry) are stored in the `screenshots/` folder.

## Results

### Before hyperparameter tuning

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.7864   | 0.7859    | 0.7864 | 0.7835   |
| SVM                 | 0.9320   | 0.9383    | 0.9320 | 0.9326   |
| KNN                 | 0.9550   | 0.9553    | 0.9550 | 0.9551   |
| Random Forest       | 0.9562   | 0.9565    | 0.9562 | 0.9562   |

### After hyperparameter tuning

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| SVM           | 0.9846   | 0.9847    | 0.9846 | 0.9846   |
| KNN           | 0.9577   | 0.9581    | 0.9577 | 0.9577   |
| Random Forest | 0.9449   | 0.9454    | 0.9449 | 0.9449   |

### Model comparison chart

See `artifacts/model_comparison.png` (also logged as an MLflow artifact).

## Model Choice

The **Tuned SVM (RBF kernel)** is selected as the champion model for the following reasons:

- **Highest accuracy (0.9846)** among all tested models, baseline and tuned.
- **Best F1-score (0.9846)** indicating strong and balanced precision/recall across all gesture classes.
- It clearly outperforms KNN-Tuned (0.9577) and Random-Forest-Tuned (0.9449) after hyperparameter search.
- SVM with RBF kernel generalizes well to the high-dimensional landmark feature space while being relatively efficient at inference time for real-time webcam use.

The tuned SVM model is registered in MLflow as `Hand-Gesture-SVM-Champion` and also exported locally as `svm_winner.pkl`.

## Files

- `ML1_project.ipynb` -- Main Jupyter notebook (exploration, training, evaluation, MLflow logging, inference).
- `train.py` -- Standalone Python script that reproduces the full training pipeline with MLflow logging.
- `mlflow_utils.py` -- Centralized MLflow utility functions used by both the notebook and the script.
- `hand_landmarks_data.csv` -- Hand-landmark dataset.
- `requirements.txt` -- Python dependencies.
- `screenshots/` -- Screenshots from the MLflow UI (runs, charts, model registry).
- `mlruns/` -- MLflow tracking data (generated after running the pipeline).

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Key libraries: NumPy, Pandas, scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn, OpenCV, MediaPipe, joblib, MLflow.

## Running

### Notebook

1. Install dependencies (see above).
2. Open `ML1_project.ipynb` in Jupyter or VS Code.
3. Run all cells sequentially to reproduce the training pipeline with MLflow tracking.
4. The final cell launches real-time webcam inference -- press **q** to quit the window.

### Python script

```bash
python train.py
```

This runs the full pipeline (data loading, preprocessing, baseline training, tuning, comparison, model registration) and logs everything to MLflow.

### MLflow UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000 to view experiments, compare runs, and inspect the model registry.

## Notes

- The webcam inference cell requires a working camera and a display environment.
- The trained model is exported as `svm_winner.pkl`.
- The `mlruns/` folder and `screenshots/` folder should be included when pushing to the `research` branch.
