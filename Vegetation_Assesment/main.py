import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/lidar_ndvi_dataset.csv')
X = df[['CHM', 'NDVI', 'CanopyDensity', 'Elevation']]
y = df['VHI']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = [r2, rmse]
    print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")

# Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title("Correlation among CHM, NDVI and VHI")
plt.show()

# Scatter plot for best model
best_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
plt.scatter(y_test, y_pred, c='green', alpha=0.7)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("Observed VHI")
plt.ylabel("Predicted VHI")
plt.title("Observed vs Predicted Vegetation Health (GBR)")
plt.show()

# SHAP Explainability
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
