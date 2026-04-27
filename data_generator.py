import numpy as np
import pandas as pd
from feature_engineering import engineer_features, compute_weather_score
from data_ingestion import fetch_weather, fetch_holiday, get_routes

def simulate_behavioral_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    route_data = get_routes()
    n_routes = len(route_data)
    
    rows = []
    for i in range(n_samples):
        route = route_data[i % n_routes]
        days = int(rng.integers(2, 60))
        depart_date = pd.Timestamp.now() + pd.Timedelta(days=days)
        
        # Real API calls (cached)
        w_orig = fetch_weather(route["origin"], depart_date.strftime("%Y-%m-%d"))
        w_dest = fetch_weather(route["destination"], depart_date.strftime("%Y-%m-%d"))
        holiday = fetch_holiday(depart_date.strftime("%Y-%m-%d"))
        
        w_score = compute_weather_score(w_orig, w_dest)
        price = rng.uniform(120, 1800)
        cap = route["capacity"]
        seats_rem = rng.integers(10, cap + 1)
        
        # Behavioral Simulation (Core Relationships)
        # user_views: ↑ as days→0, ↑ holidays, ↓ bad weather
        base_views = 450 * np.exp(-days / 22)
        holiday_mult = 1.5 if holiday else 1.0
        weather_mult = 1.2 * w_score + 0.8
        views = base_views * holiday_mult * weather_mult
        views *= rng.uniform(0.85, 1.15) # noise
        
        # conversion_rate: ↓ as price ↑, ↑ as seats_remaining ↓ (urgency)
        scarcity = 1 + 0.6 * (1 - seats_rem / cap)
        price_penalty = np.exp(-price / 220)
        conv_rate = price_penalty * scarcity
        conv_rate = np.clip(conv_rate * rng.uniform(0.8, 1.1), 0.01, 0.85)
        
        demand = views * conv_rate
        demand = min(demand, seats_rem)
        demand = int(demand * rng.uniform(0.95, 1.05))
        demand = max(0, min(demand, seats_rem))
        
        rows.append({
            "route_id": route["route_id"],
            "price": price,
            "seat_capacity": cap,
            "seats_remaining": seats_rem,
            "user_views": int(views),
            "conversion_rate": round(conv_rate, 4),
            "weather_score": round(w_score, 3),
            "holiday_flag": int(holiday),
            "days_to_departure": days,
            "day_of_week": depart_date.dayofweek,
            "demand": demand
        })
        
    return pd.DataFrame(rows)
