import numpy as np
import pandas as pd
import model as mdl

def _estimate_cost_per_seat(row: pd.Series, market_price: float) -> float:
    """Simple operating-cost proxy used as a pricing floor."""
    seat_capacity = max(1.0, float(row.get("seat_capacity", 150.0)))
    base_cost = 70.0 + 0.08 * seat_capacity
    return max(base_cost, 0.45 * market_price)

def _refresh_behavioral_features(row: pd.Series, price: float, market_price: float) -> pd.Series:
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
    mid_price = max(80.0, float(market_price))
    price_penalty = np.exp(-price / mid_price)
    conversion_rate = float(np.clip(price_penalty * scarcity, 0.01, 0.85))

    row["user_views"] = int(round(views))
    row["conversion_rate"] = conversion_rate
    row["market_price"] = float(market_price)
    return row

def optimize_price(
    scenario_features: pd.Series,
    trained_model,
    price_min=None,
    price_max=None,
    step=5,
    market_price: float = None,
    max_iterations=5,
    expand_factor=1.5,
):
    """Adaptive sweep price optimization with dynamic range expansion."""
    if isinstance(scenario_features, pd.DataFrame):
        base_row = scenario_features.iloc[0].copy()
    else:
        base_row = scenario_features.copy()

    if market_price is None:
        if "market_price" in base_row and pd.notna(base_row["market_price"]):
            market_price = float(base_row["market_price"])
        else:
            market_price = float(base_row.get("price", 300.0))

    cost_per_seat = _estimate_cost_per_seat(base_row, market_price)
    anchor_min = max(cost_per_seat, market_price * 0.7)
    anchor_max = market_price * 1.5

    if price_min is None:
        current_price_min = anchor_min
    else:
        current_price_min = max(float(price_min), anchor_min)

    if price_max is None:
        current_price_max = anchor_max
    else:
        current_price_max = min(float(price_max), anchor_max)

    if current_price_max <= current_price_min:
        current_price_min = anchor_min
        current_price_max = anchor_max

    final_result = None

    for i in range(max_iterations):
        current_price_min = max(current_price_min, cost_per_seat)
        if current_price_max <= current_price_min:
            current_price_max = current_price_min + max(float(step), 1.0)

        print(
            f"[Adaptive Sweep] Iter {i+1}: "
            f"Range RM {current_price_min:.0f} - RM {current_price_max:.0f}"
        )

        price_history = np.arange(current_price_min, current_price_max + 1e-9, step)
        if price_history.size == 0:
            price_history = np.array([current_price_min], dtype=float)

        demand_history = []
        revenue_history = []
        profit_history = []
        user_views_history = []
        conversion_rate_history = []

        for p in price_history:
            row = base_row.copy()
            row["price"] = p
            row = _refresh_behavioral_features(row, p, market_price)
            user_views_history.append(float(row["user_views"]))
            conversion_rate_history.append(float(row["conversion_rate"]))
            X_input = pd.DataFrame([row], columns=mdl.FEATURES)
            demand_pred = trained_model.predict(X_input)[0]
            demand_capped = max(0, min(demand_pred, row["seats_remaining"]))
            rev = p * demand_capped
            profit = (p - cost_per_seat) * demand_capped
            demand_history.append(demand_capped)
            revenue_history.append(rev)
            profit_history.append(profit)

        best_idx = int(np.argmax(revenue_history))
        optimal_price = float(price_history[best_idx])

        final_result = {
            "optimal_price": optimal_price,
            "expected_demand": demand_history[best_idx],
            "expected_revenue": revenue_history[best_idx],
            "expected_profit": profit_history[best_idx],
            "price_history": price_history,
            "demand_history": demand_history,
            "revenue_history": revenue_history,
            "profit_history": profit_history,
            "user_views_history": user_views_history,
            "conversion_rate_history": conversion_rate_history,
            "market_price": float(market_price),
            "price_min_used": float(current_price_min),
            "price_max_used": float(current_price_max),
            "cost_per_seat": float(cost_per_seat),
            "iterations_used": i + 1,
        }

        hit_upper = optimal_price >= (current_price_max - step)
        hit_lower = optimal_price <= (current_price_min + step)

        if hit_upper:
            current_price_max *= float(expand_factor)
        elif hit_lower:
            current_price_min *= 0.7
            current_price_min = max(current_price_min, cost_per_seat)
        else:
            break

    if final_result is None:
        # Defensive fallback, should not be reached.
        final_result = {
            "optimal_price": float(current_price_min),
            "expected_demand": 0.0,
            "expected_revenue": 0.0,
            "expected_profit": 0.0,
            "price_history": np.array([current_price_min], dtype=float),
            "demand_history": [0.0],
            "revenue_history": [0.0],
            "profit_history": [0.0],
            "user_views_history": [0.0],
            "conversion_rate_history": [0.0],
            "market_price": float(market_price),
            "price_min_used": float(current_price_min),
            "price_max_used": float(current_price_max),
            "cost_per_seat": float(cost_per_seat),
            "iterations_used": 0,
        }

    return final_result
