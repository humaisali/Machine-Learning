# ============================================================
# Lab 10 - Task 5: XGBoost on Heart Disease Dataset
# Feature importance analysis + held-out test evaluation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

# ── 1. Load Dataset ──────────────────────────────────────────
# UCI Heart Disease dataset (Cleveland)
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/heart-disease/processed.cleveland.data")

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=columns, na_values='?')

print("=== Heart Disease Dataset (Cleveland) ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── 2. Preprocessing ─────────────────────────────────────────
# Drop rows with missing values (only 6 rows)
df.dropna(inplace=True)

# Binarize target: 0 = no disease, 1 = disease
df['target'] = (df['target'] > 0).astype(int)

print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
print(f"Classes — 0: No Disease, 1: Disease")

X = df.drop(columns=['target'])
y = df['target']

feature_names = list(X.columns)

# ── 3. Train / Validation / Test Split ───────────────────────
# 60% train | 20% val | 20% held-out test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 4. Train XGBoost Model ───────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ── 5. Held-Out Test Evaluation ──────────────────────────────
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n=== Held-Out Test Set Performance ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['No Disease', 'Disease']))

# ── 6. Feature Importance Analysis ───────────────────────────
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
sorted_feats = [feature_names[i] for i in sorted_idx]
sorted_imps  = importances[sorted_idx]

# Human-readable feature descriptions
feat_desc = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
    'trestbps': 'Resting BP', 'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG', 'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina', 'oldpeak': 'ST Depression',
    'slope': 'Slope of ST', 'ca': 'Num Major Vessels', 'thal': 'Thalassemia'
}
sorted_labels = [feat_desc.get(f, f) for f in sorted_feats]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Horizontal bar chart
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_feats)))
bars = ax1.barh(sorted_labels[::-1], sorted_imps[::-1], color=colors[::-1])
ax1.set_xlabel('Feature Importance Score', fontsize=11)
ax1.set_title('Feature Importance — XGBoost\nHeart Disease Dataset', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for bar, imp in zip(bars, sorted_imps[::-1]):
    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{imp:.3f}', va='center', fontsize=8)

# Confusion Matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Disease', 'Disease'])
disp.plot(ax=ax2, colorbar=True, cmap='Blues')
ax2.set_title(f'Confusion Matrix — Held-Out Test\nAcc={acc:.3f} | F1={f1:.3f}',
              fontweight='bold')

plt.suptitle('XGBoost Feature Importance & Evaluation — Heart Disease',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('/home/claude/task5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFeature importance & confusion matrix plot saved.")

# ── 7. Top Features Summary ──────────────────────────────────
print("\n=== Top 5 Most Important Features ===")
for rank, (feat, imp) in enumerate(zip(sorted_feats[:5], sorted_imps[:5]), 1):
    print(f"  {rank}. {feat_desc.get(feat, feat):20s}: {imp:.4f}")
