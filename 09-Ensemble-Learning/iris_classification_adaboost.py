# ============================================================
# Lab 10 - Task 2: AdaBoost on Iris Dataset
# Classify 3 species & visualize decision boundary changes
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import label_binarize

# ── 1. Load Dataset ──────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

# Use only 2 features for 2D decision boundary visualization
X_2d = X[:, :2]  # sepal length & sepal width

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_2d, X_test_2d, _, _ = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Iris Dataset ===")
print(f"Classes : {iris.target_names}")
print(f"Features: {iris.feature_names}")
print(f"Samples : {X.shape[0]}")

# ── 2. Compare Different Number of Estimators ────────────────
estimator_counts = [1, 5, 10, 50, 100, 200]
results = []

print("\n=== Performance vs Number of Estimators ===")
print(f"{'Estimators':>12} | {'Accuracy':>9} | {'Precision':>10} | {'Recall':>7}")
print("-" * 50)

for n in estimator_counts:
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    results.append((n, acc, prec, rec))
    print(f"{n:>12} | {acc:>9.4f} | {prec:>10.4f} | {rec:>7.4f}")

# ── 3. Decision Boundary Visualization ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cmap   = plt.cm.get_cmap('Set1', 3)

for i, n in enumerate(estimator_counts):
    clf_2d = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    clf_2d.fit(X_train_2d, y_train)

    # Create mesh grid
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = axes[i]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    for cls, color in zip(range(3), colors):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=iris.target_names[cls],
                   edgecolors='k', s=40, alpha=0.8)

    ax.set_title(f'AdaBoost — {n} Estimator(s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Decision Boundary Evolution — AdaBoost on Iris Dataset',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/task2_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDecision boundary plot saved.")

# ── 4. Performance Metrics Bar Chart ─────────────────────────
ns, accs, precs, recs = zip(*results)

x = np.arange(len(ns))
width = 0.25

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(x - width, accs,  width, label='Accuracy',  color='#4ECDC4')
ax2.bar(x,         precs, width, label='Precision', color='#45B7D1')
ax2.bar(x + width, recs,  width, label='Recall',    color='#FF6B6B')

ax2.set_xlabel('Number of Estimators')
ax2.set_ylabel('Score')
ax2.set_title('AdaBoost Performance vs Number of Estimators (Iris)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(ns)
ax2.set_ylim(0, 1.1)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/task2_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Metrics comparison plot saved.")

# ── 5. Final Model Report ────────────────────────────────────
best_n = max(results, key=lambda x: x[1])[0]
clf_best = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=best_n,
    random_state=42
)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

print(f"\n=== Best Model (n_estimators={best_n}) ===")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))
