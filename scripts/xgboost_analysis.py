import xgboost as xgb
import shap
import pandas as pd

# Load processed data
df = pd.read_csv('banana_sse_provincial_data.csv') # Siguraduhin na tama ang filename

# Features base sa paper
features = ['TS', 'CWDD', 'CEI', 'DPI', 'Fertilizer', 'GAP']
X = df[features]
y = df['SSE']

# Train XGBoost
model = xgb.XGBRegressor(n_estimators=1000, max_depth=6, eta=0.05)
model.fit(X, y)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("Analysis complete.")
