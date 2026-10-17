from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Create folder for results
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 1. Load public dataset
print("Loading Student Performance dataset...")
dataset = fetch_ucirepo(id=320)

X = dataset.data.features.copy()
y = dataset.data.targets["G3"].copy()

# Exclude earlier grades to avoid target leakage
X = X.drop(columns=["G1", "G2"], errors="ignore")

# Remove rows with missing target values
y = pd.to_numeric(y, errors="coerce")
valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].copy()

print("Dataset loaded:", X.shape)

# 2. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Identify feature types
numeric = X_train.select_dtypes(include="number").columns.tolist()
categorical = X_train.select_dtypes(exclude="number").columns.tolist()

# 4. Preprocess numeric and categorical features
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical)
])

# 5. Choose models
models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# 6. Train and compare models
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
trained = {}

for name, estimator in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])

    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="neg_mean_absolute_error"
    )

    cv_mae = -scores.mean()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "Model": name,
        "CV_MAE": cv_mae,
        "Test_MAE": mae,
        "Test_RMSE": rmse
    })

    trained[name] = model

results_df = pd.DataFrame(results).sort_values("CV_MAE")
results_df.to_csv(output_dir / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# Select model using training cross-validation only
best_name = results_df.iloc[0]["Model"]
best_model = trained[best_name]

# 7. Final evaluation on test data
predictions = best_model.predict(X_test)
residuals = y_test.to_numpy() - predictions

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

baseline_mae = results_df.loc[
    results_df["Model"] == "Baseline", "Test_MAE"
].iloc[0]

improvement = (
    (baseline_mae - mae) / baseline_mae * 100
    if baseline_mae > 0 else 0
)

print("\nFINAL TEST RESULTS")
print("Selected model:", best_name)
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Baseline MAE: {baseline_mae:.3f}")
print(f"Improvement over baseline: {improvement:.2f}%")

# Save predictions
pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Residual": residuals
}).to_csv(output_dir / "test_predictions.csv", index=False)

# 8. Actual vs predicted chart
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([0, 20], [0, 20], "--")
plt.xlabel("Actual final grade")
plt.ylabel("Predicted final grade")
plt.title("Actual vs Predicted Grades")
plt.tight_layout()
plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
plt.close()

# 9. Residual plot
plt.figure(figsize=(7, 5))
plt.scatter(predictions, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted grade")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(output_dir / "residual_plot.png", dpi=150)
plt.close()

# 10. Feature importance
importance = permutation_importance(
    best_model, X_test, y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5, random_state=42
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
}).sort_values("Importance", ascending=False)

importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

top = importance_df.head(10).sort_values("Importance")
plt.figure(figsize=(8, 6))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Increase in MAE when feature is shuffled")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance.png", dpi=150)
plt.close()

# 11. Save evaluation summary
with open(output_dir / "evaluation_summary.txt", "w") as file:
    file.write("Student Performance Prediction\n")
    file.write(f"Selected model: {best_name}\n")
    file.write(f"Test MAE: {mae:.3f}\n")
    file.write(f"Test RMSE: {rmse:.3f}\n")
    file.write(f"Baseline MAE: {baseline_mae:.3f}\n")
    file.write(f"Improvement: {improvement:.2f}%\n")
    file.write("G1 and G2 excluded from model inputs.\n")

print("\nProject completed successfully!")
print("Check the outputs folder for charts and results.")