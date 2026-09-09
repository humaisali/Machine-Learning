# ============================================================
# Lab 10 - Task 4: AdaBoost on Bank Marketing Dataset
# Predict term deposit subscription — handle class imbalance
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import urllib.request

# ── 1. Load Dataset ──────────────────────────────────────────
# Bank Marketing dataset from UCI (semicolon-separated)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

import io, zipfile, urllib.request

response = urllib.request.urlopen(url)
zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
with zip_data.open('bank.csv') as f:
    df = pd.read_csv(f, sep=';')

print("=== Bank Marketing Dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nTarget Distribution:\n{df['y'].value_counts()}")
print(f"\nClass Imbalance Ratio: {df['y'].value_counts()['no'] / df['y'].value_counts()['yes']:.2f}:1")

# ── 2. Preprocessing ─────────────────────────────────────────
df_enc = df.copy()

# Encode all categorical columns
le = LabelEncoder()
cat_cols = df_enc.select_dtypes(include='object').columns
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop(columns=['y'])
y = df_enc['y']  # 0=no, 1=yes

print(f"\nFeatures: {list(X.columns)}")
print(f"Encoded target — 0: no, 1: yes")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Handle Imbalance via Sample Weights ───────────────────
# Compute per-sample weights: minority class gets higher weight
class_counts = y_train.value_counts()
n_total = len(y_train)
class_weight = {cls: n_total / (2 * count) for cls, count in class_counts.items()}
sample_weights = y_train.map(class_weight).values

print(f"\nClass weights: {class_weight}")

# ── 4. Train Two Models (without vs with sample weighting) ───
models = {
    'Without Sample Weights': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, random_state=42
    ),
    'With Sample Weights': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, random_state=42
    )
}

fit_args = {
    'Without Sample Weights': {},
    'With Sample Weights': {'sample_weight': sample_weights}
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

roc_ax = axes[0]
all_results = {}

for name, model in models.items():
    model.fit(X_train, y_train, **fit_args[name])
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    all_results[name] = {
        'y_pred': y_pred, 'roc_auc': roc_auc,
        'fpr': fpr, 'tpr': tpr
    }
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# ── 5. ROC Curves ────────────────────────────────────────────
for name, res in all_results.items():
    roc_ax.plot(res['fpr'], res['tpr'],
                label=f"{name} (AUC={res['roc_auc']:.3f})")

roc_ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')
roc_ax.set_title('ROC Curve Comparison', fontweight='bold')
roc_ax.legend(fontsize=8)
roc_ax.grid(alpha=0.3)

# ── 6. Confusion Matrices ────────────────────────────────────
for ax, (name, res) in zip(axes[1:], all_results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No', 'Yes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold')

plt.suptitle('AdaBoost on Bank Marketing — Effect of Sample Weighting',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('/home/claude/task4_roc_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nROC & Confusion Matrix plot saved.")
