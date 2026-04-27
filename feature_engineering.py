import numpy as np
import pandas as pd
from typing import Optional

CONDITION_MAP = {
    "Clear": 1.0, "Clouds": 0.9, "Mist": 0.7, "Rain": 0.5, 
    "Thunderstorm": 0.2, "Snow": 0.3, "Haze": 0.8
}

def compute_weather_score(weather: dict, dest_weather: Optional[dict] = None) -> float:
    t1 = weather["temp"]; t2 = dest_weather["temp"] if dest_weather else t1
    temp_avg = (t1 + t2) / 2
    precip_avg = (weather["precip_mm"] + (dest_weather.get("precip_mm", 0) if dest_weather else 0)) / 2
    wind_avg = (weather["wind_speed"] + (dest_weather.get("wind_speed", 0) if dest_weather else 0)) / 2
    
    temp_score = np.exp(-((temp_avg - 25)**2) / 100)
    precip_score = max(0, 1 - precip_avg / 10)
    wind_score = max(0, 1 - wind_avg / 15)
    cond = weather.get("condition", "Clear")
    cond_score = CONDITION_MAP.get(cond, 0.4)
    dest_cond = dest_weather.get("condition", "Clear") if dest_weather else cond
    cond_score = (CONDITION_MAP.get(cond, 0.4) + CONDITION_MAP.get(dest_cond, 0.4)) / 2
    
    score = 0.3 * temp_score + 0.3 * precip_score + 0.2 * wind_score + 0.2 * cond_score
    
    # Disruption penalty for extreme conditions
    if weather.get("condition") in ["Thunderstorm", "Snow"] or precip_avg > 8:
        score *= 0.6
    
    return np.clip(score, 0, 1)

def engineer_features(df_raw: pd.DataFrame, depart_dates: list, route_data: list, holiday_flag: list) -> pd.DataFrame:
    """Transform raw inputs into exactly 10 features + target."""
    routes_df = pd.DataFrame(route_data)
    df = pd.DataFrame({
        "route_id": [r["route_id"] for r in route_data],
        "seat_capacity": [r["capacity"] for r in route_data],
        "depart_date": pd.to_datetime(depart_dates)
    })
    df = df.merge(routes_df, on="route_id", how="left")
    df["holiday_flag"] = holiday_flag
    
    today = pd.Timestamp.now().normalize()
    df["days_to_departure"] = (df["depart_date"] - today).dt.days.clip(lower=1)
    df["day_of_week"] = df["depart_date"].dt.dayofweek
    
    # Initialize other features with defaults (will be populated by data_generator)
    for col in ["price", "seats_remaining", "user_views", "conversion_rate", "weather_score"]:
        df[col] = 0.0
    df["demand"] = 0
    
    return df[df.columns.intersection([
        "price", "days_to_departure", "route_id", "seat_capacity", 
        "seats_remaining", "user_views", "conversion_rate", "weather_score", 
        "holiday_flag", "day_of_week", "demand"
    ])]
