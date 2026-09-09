# Ensemble Learning Projects (XGBoost & AdaBoost)

## Overview
While bagging methods (like Random Forest) train models independently, Boosting methods train models sequentially, with each new model focusing on correcting the errors of the previous ones. This folder contains advanced predictive projects using state-of-the-art boosting algorithms.

## Real-World Projects Implemented
* **`heart_disease_detection_xgboost.py`**: Uses XGBoost to determine feature importance (e.g., Chest Pain Type, Max Heart Rate) and predict heart disease on the Cleveland dataset.
* **`titanic_survival_xgboost.py`**: Predicts passenger survival using Extreme Gradient Boosting, handling missing values and evaluating logloss.
* **`bank_marketing_adaboost.py`**: Predicts client term deposit subscriptions using AdaBoost. Implements sample weighting to handle severe class imbalance and compares ROC-AUC curves.
* **`california_housing_xgboost.py`**: Performs an XGBoost regression on California housing prices. Generates hyperparameter heatmaps (RMSE, R²) to analyze learning rates and max depths.
* **`iris_classification_adaboost.py`**: Visualizes AdaBoost decision boundary evolution as the number of estimators increases on the Iris dataset.

## Key Concepts Demonstrated
* Gradient Boosting (XGBoost) and Adaptive Boosting (AdaBoost).
* Class Imbalance Handling via sample weights.
* Advanced Visualizations: ROC-AUC Curves, Confusion Matrices, Decision Boundaries, and Hyperparameter Heatmaps.
