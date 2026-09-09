# ============================================================
# Lab 10 - Task 3: XGBoost on California Housing Dataset
# Predict house prices — experiment with max_depth & lr
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import itertools

# ── 1. Load Dataset ──────────────────────────────────────────
housing = fetch_california_housing()
X, y    = housing.data, housing.target

df = pd.DataFrame(X, columns=housing.feature_names)
df['MedHouseVal'] = y

print("=== California Housing Dataset ===")
print(df.head())
print(f"\nShape  : {df.shape}")
print(f"Target : MedHouseVal (median house value in $100,000s)")
print(f"\nMissing: {df.isnull().sum().sum()} values")

# ── 2. Split & Scale ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. Hyperparameter Experiment ─────────────────────────────
max_depths     = [3, 5, 7]
learning_rates = [0.01, 0.1, 0.3]

results = []
print("\n=== Hyperparameter Experiment ===")
print(f"{'Max Depth':>10} | {'Learning Rate':>14} | {'RMSE':>8} | {'R² Score':>9}")
print("-" * 55)

for depth, lr in itertools.product(max_depths, learning_rates):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=depth,
        learning_rate=lr,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results.append((depth, lr, rmse, r2))
    print(f"{depth:>10} | {lr:>14.2f} | {rmse:>8.4f} | {r2:>9.4f}")

# ── 4. Visualization — RMSE Heatmap ──────────────────────────
rmse_matrix = np.zeros((len(max_depths), len(learning_rates)))
r2_matrix   = np.zeros((len(max_depths), len(learning_rates)))

for d, depth in enumerate(max_depths):
    for l, lr in enumerate(learning_rates):
        for row in results:
            if row[0] == depth and row[1] == lr:
                rmse_matrix[d][l] = row[2]
                r2_matrix[d][l]   = row[3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.imshow(rmse_matrix, cmap='YlOrRd')
ax1.set_xticks(range(len(learning_rates)))
ax1.set_yticks(range(len(max_depths)))
ax1.set_xticklabels([str(lr) for lr in learning_rates])
ax1.set_yticklabels([str(d) for d in max_depths])
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Max Depth')
ax1.set_title('RMSE Heatmap\n(Lower is Better)', fontweight='bold')
plt.colorbar(im1, ax=ax1)
for i in range(len(max_depths)):
    for j in range(len(learning_rates)):
        ax1.text(j, i, f"{rmse_matrix[i, j]:.3f}", ha='center', va='center', fontsize=10)

im2 = ax2.imshow(r2_matrix, cmap='YlGn')
ax2.set_xticks(range(len(learning_rates)))
ax2.set_yticks(range(len(max_depths)))
ax2.set_xticklabels([str(lr) for lr in learning_rates])
ax2.set_yticklabels([str(d) for d in max_depths])
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Max Depth')
ax2.set_title('R² Score Heatmap\n(Higher is Better)', fontweight='bold')
plt.colorbar(im2, ax=ax2)
for i in range(len(max_depths)):
    for j in range(len(learning_rates)):
        ax2.text(j, i, f"{r2_matrix[i, j]:.3f}", ha='center', va='center', fontsize=10)

plt.suptitle('XGBoost Hyperparameter Analysis — California Housing', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/task3_hyperparameter_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nHeatmap saved.")

# ── 5. Best Model — Actual vs Predicted ──────────────────────
best = min(results, key=lambda x: x[2])
print(f"\n=== Best Config: depth={best[0]}, lr={best[1]} ===")
print(f"RMSE: {best[2]:.4f} | R²: {best[3]:.4f}")

best_model = XGBRegressor(
    n_estimators=200,
    max_depth=best[0],
    learning_rate=best[1],
    random_state=42,
    verbosity=0
)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

fig2, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_best, alpha=0.3, s=10, color='#4ECDC4')
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
ax.set_xlabel('Actual House Value ($100k)')
ax.set_ylabel('Predicted House Value ($100k)')
ax.set_title(f'Actual vs Predicted (depth={best[0]}, lr={best[1]})\nRMSE={best[2]:.4f}, R²={best[3]:.4f}',
             fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/task3_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("Actual vs Predicted plot saved.")
