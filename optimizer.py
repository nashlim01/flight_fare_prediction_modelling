import numpy as np
import pandas as pd
import model as mdl

def _refresh_behavioral_features(row: pd.Series, price: float) -> pd.Series:
    """Recompute demand-driving behavioral features for a given price point."""
    days = max(1, float(row["days_to_departure"]))
    weather = float(row["weather_score"])
    holiday = int(row["holiday_flag"])
    seats_remaining = max(1.0, float(row["seats_remaining"]))
    seat_capacity = max(seats_remaining, float(row["seat_capacity"]))

    base_views = 450 * np.exp(-days / 22.0)
    holiday_mult = 1.5 if holiday else 1.0
    weather_mult = 1.2 * weather + 0.8
    views = max(1.0, base_views * holiday_mult * weather_mult)

    scarcity = 1 + 0.6 * (1 - seats_remaining / seat_capacity)
    price_penalty = np.exp(-price / 220.0)
    conversion_rate = float(np.clip(price_penalty * scarcity, 0.01, 0.85))

    row["user_views"] = int(round(views))
    row["conversion_rate"] = conversion_rate
    return row

def optimize_price(scenario_features: pd.Series, trained_model, price_min=100, price_max=1000, step=5):
    """Sweep price to maximize expected revenue."""
    prices = np.arange(price_min, price_max + step, step)
    revenues = []
    predicted_demands = []
    user_views_history = []
    conversion_rate_history = []

    if isinstance(scenario_features, pd.DataFrame):
        base_row = scenario_features.iloc[0].copy()
    else:
        base_row = scenario_features.copy()
    
    for p in prices:
        row = base_row.copy()
        row["price"] = p
        row = _refresh_behavioral_features(row, p)
        user_views_history.append(float(row["user_views"]))
        conversion_rate_history.append(float(row["conversion_rate"]))
        X_input = pd.DataFrame([row], columns=mdl.FEATURES)
        demand_pred = trained_model.predict(X_input)[0]
        demand_capped = max(0, min(demand_pred, row["seats_remaining"]))
        rev = p * demand_capped
        revenues.append(rev)
        predicted_demands.append(demand_capped)
        
    best_idx = np.argmax(revenues)
    return {
        "optimal_price": prices[best_idx],
        "expected_demand": predicted_demands[best_idx],
        "expected_revenue": revenues[best_idx],
        "price_history": prices,
        "demand_history": predicted_demands,
        "revenue_history": revenues,
        "user_views_history": user_views_history,
        "conversion_rate_history": conversion_rate_history
    }
