
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "energy_data_raw.csv"

# 1. Load
df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 2. Data cleaning — same approach as Review-2
numeric_cols = ["load_kw","temperature_c","humidity_pct","prev_24h_avg_load"]
categorical_cols = ["day_type","is_holiday","feeder_id"]

for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 3. Feature engineering
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# 15-minute data: shift by 4 rows = next hour
df["next_hour_load"] = df["load_kw"].shift(-4)
df = df.dropna(subset=["next_hour_load"]).reset_index(drop=True)

features = [
    "temperature_c","humidity_pct","prev_24h_avg_load",
    "hour","day_of_week","month","day_type","is_holiday","feeder_id"
]

num_features = [
    "temperature_c","humidity_pct","prev_24h_avg_load",
    "hour","day_of_week","month"
]
cat_features = ["day_type","is_holiday","feeder_id"]

# 4. Chronological train/test split
split = int(len(df) * 0.80)
train = df.iloc[:split]
test = df.iloc[split:]

X_train, y_train = train[features], train["next_hour_load"]
X_test, y_test = test[features], test["next_hour_load"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=16, random_state=42, n_jobs=-1
    ),
    "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance")
}

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print(f"MAE : {mae:.2f} kW")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"R2  : {r2:.4f}")

# 5. Main KNN model
knn = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=7, weights="distance"))
])
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# 6. Adaptive peak threshold: 90th percentile of training load
threshold = train["load_kw"].quantile(0.90)

def classify(load):
    if load >= threshold * 1.10:
        return "Critical"
    elif load >= threshold:
        return "Warning"
    return "Normal"

output = test[["timestamp","load_kw"]].copy()
output["predicted_next_hour_kw"] = pred
output["risk"] = [classify(x) for x in pred]
output.to_csv("knn_predictions.csv", index=False)

print("\nPeak threshold:", round(threshold, 2), "kW")
print(output.head(20).to_string(index=False))
