# ============================================================
# Lab 10 - Task 1: XGBoost on Titanic Dataset
# Predict survival using XGBoost
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── 1. Load Dataset ──────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== Titanic Dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── 2. Preprocessing ─────────────────────────────────────────

# Drop columns not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)          # Median age
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Most frequent port

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked']) # C=0, Q=1, S=2

print("\n=== Preprocessed Data ===")
print(df.head())
print(f"\nMissing Values After Preprocessing:\n{df.isnull().sum()}")

# ── 3. Feature / Target Split ────────────────────────────────
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ── 4. Train XGBoost Model ───────────────────────────────────
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
