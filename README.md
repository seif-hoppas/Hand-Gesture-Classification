# Hand Gesture Classification

Classify hand gestures from MediaPipe hand-landmark coordinates using classical ML models. The project covers data exploration, preprocessing, model comparison, hyperparameter tuning, and real-time webcam inference.

## Overview

1. **Data** -- `hand_landmarks_data.csv` contains (x, y, z) coordinates for 21 hand landmarks per sample, along with a gesture label.
2. **Preprocessing** -- Landmarks are normalized relative to the wrist (landmark 1) and scaled by the distance to the middle-finger tip (landmark 13).
3. **Model training** -- Four classifiers are compared: Logistic Regression, SVM, KNN, and Random Forest. RandomizedSearchCV is used for hyperparameter tuning on SVM, KNN, and Random Forest.
4. **Evaluation** -- Models are evaluated with accuracy, precision, recall, and F1-score.
5. **Real-time inference** -- The best model (SVM) is saved with joblib and used for live gesture recognition through a webcam via OpenCV and MediaPipe.

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

## Files

- `ML1_project.ipynb` -- Main Jupyter notebook (exploration, training, evaluation, inference).
- `hand_landmarks_data.csv` -- Hand-landmark dataset.
- `requirements.txt` -- Python dependencies.

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Key libraries: NumPy, Pandas, scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn, OpenCV, MediaPipe, joblib.

## Running

1. Install dependencies (see above).
2. Open `ML1_project.ipynb` in Jupyter or VS Code.
3. Run all cells sequentially to reproduce the training pipeline.
4. The final cell launches real-time webcam inference -- press **q** to quit the window.

## Notes

- The webcam inference cell requires a working camera and a display environment.
- The trained model is exported as `svm_winner.pkl`.
