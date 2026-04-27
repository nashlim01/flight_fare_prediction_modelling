import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

FEATURES = [
    "price", "market_price", "days_to_departure", "route_id", "seat_capacity", 
    "seats_remaining", "user_views", "conversion_rate", "weather_score", 
    "holiday_flag", "day_of_week"
]

def train_demand_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            min_child_samples=10, random_state=42, verbosity=-1
        )
        model_name = "LightGBM"
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
        )
        model_name = "GradientBoostingRegressor (fallback)"

    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"✅ {model_name} RMSE on test set: {rmse:.2f}")
    return model, rmse
