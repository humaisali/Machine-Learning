# Data Preprocessing

## Overview
Data preprocessing is the foundational step in any machine learning pipeline. Raw data is often messy, incomplete, and strictly not in a format that models can interpret. This module focuses on the techniques required to clean and transform raw data into high-quality features.

## Key Concepts Implemented
* **Missing Value Imputation**: Handling `NaN` values intelligently (e.g., filling missing ages with the median and embarked ports with the mode).
* **Categorical Encoding**:
  * **Label Encoding**: Converting ordinal categorical variables into integers.
  * **One-Hot Encoding**: Transforming nominal categorical variables into binary dummy variables to prevent the model from inferring a false order.
* **Feature Scaling**:
  * **Min-Max Scaling**: Normalizing data to a specific range (usually `[0, 1]`).
  * **Standardization (Z-Scoring)**: Centering data around a mean of 0 with a standard deviation of 1, which is crucial for distance-based algorithms.
* **Correlation Analysis**: Generating heatmaps to identify relationships between features before modeling.

## Dataset Used
* `Titanic-Dataset.csv`
