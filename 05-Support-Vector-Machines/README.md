# Support Vector Machines (SVM)

## Overview
Support Vector Machines are powerful models for classification and regression tasks. They work by finding the optimal hyperplane that maximizes the margin between classes. This module demonstrates SVM fundamentals, kernel usage, and hyperparameter tuning.

## Key Concepts Implemented
* **Linear SVM**: Finding optimal decision boundaries for linearly separable data.
* **Kernel Trick**: 
  * Using `rbf` (Radial Basis Function) and `poly` (Polynomial) kernels to classify non-linear datasets (e.g., interlocking "moons" or blobs).
* **Support Vector Regression (SVR)**: Adapting SVMs for continuous value prediction.
* **Hyperparameter Tuning**: Using `GridSearchCV` to exhaustively search for the best `C` (regularization), `kernel`, and `gamma` combinations.
* **Decision Boundary Visualizations**: Plotting hyperplanes, margins, and the specific support vectors that define the boundary.

## Datasets Used
* Synthetic datasets: `make_blobs`, `make_moons`, `make_regression` (Scikit-Learn built-in)
* `digits` Dataset
