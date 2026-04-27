import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_generator import simulate_behavioral_dataset
from model import train_demand_model, FEATURES
from optimizer import optimize_price
from data_ingestion import get_routes, get_routes_fingerprint, fetch_market_price


st.set_page_config(
    page_title="Airline Pricing Optimizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: "Space Grotesk", sans-serif;
}

[data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at top right, #ffe8cc 0%, #fff3e3 38%, #fffaf2 82%);
    color: #0b2239;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div {
    color: #0b2239;
}

[data-testid="stHeader"]{
    background: rgba(255, 255, 255, 0.75);
}

[data-testid="stForm"],
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-radius: 12px;
}

/* Inputs and dropdowns */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background: #ffffff !important;
    color: #0b2239 !important;
    border-color: #c8dff3 !important;
}

[data-baseweb="select"] * {
    color: #0b2239 !important;
}

/* Slider/readable labels */
[data-testid="stSlider"] * {
    color: #0b2239 !important;
}

/* Toggle (Holiday Period) readability */
[data-testid="stToggle"] [role="switch"] {
    background: #b8c4d0 !important;
    border: 1px solid #8fa2b3 !important;
}

[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background: #2f6b9a !important;
    border-color: #245779 !important;
}

[data-testid="stToggle"] [role="switch"] * {
    color: #ffffff !important;
}

[data-testid="stToggle"] label p {
    color: #0b2239 !important;
    font-weight: 600;
}

/* Buttons */
button[kind="primary"],
[data-testid="stFormSubmitButton"] button,
.stButton > button {
    background: #0b4f84 !important;
    color: #ffffff !important;
    border: 1px solid #0b4f84 !important;
}

button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    background: #083a62 !important;
    border-color: #083a62 !important;
}

.headline {
    padding: 0.7rem 1rem;
    border: 1px solid #d7e9f8;
    background: #f1f8ff;
    border-radius: 14px;
    color: #0b2239;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='headline'><h2 style='margin:0;'>Malaysia Flight Pricing Forecaster (RM)</h2>"
    "<p style='margin:0.25rem 0 0 0;'>Tune route conditions, run a price sweep, and read demand-revenue behavior instantly.</p></div>",
    unsafe_allow_html=True,
)

routes = get_routes()
route_lookup = {r["route_id"]: r for r in routes}
route_labels = {r["route_id"]: f"{r['origin']} -> {r['destination']} (Route {r['route_id']})" for r in routes}


@st.cache_resource
def load_pipeline(routes_fingerprint: str):
    df = simulate_behavioral_dataset(4000, seed=123)
    model, rmse = train_demand_model(df)
    return model, rmse, df.sample(1, random_state=42)[FEATURES]


def build_sweep_df(opt_result: dict, seats_remaining: float) -> pd.DataFrame:
    sweep_df = pd.DataFrame(
        {
            "price": opt_result["price_history"],
            "predicted_demand": opt_result["demand_history"],
            "revenue_rm": opt_result["revenue_history"],
            "user_views": opt_result["user_views_history"],
            "conversion_rate": opt_result["conversion_rate_history"],
        }
    )
    sweep_df["purchase_rate_pct"] = (
        sweep_df["predicted_demand"] / sweep_df["user_views"].replace(0, np.nan)
    ) * 100
    sweep_df["seat_takeup_pct"] = (sweep_df["predicted_demand"] / max(seats_remaining, 1.0)) * 100
    sweep_df["price_quartile"] = pd.qcut(
        sweep_df["price"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    return sweep_df


def summarize_quartiles(sweep_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sweep_df.groupby("price_quartile", observed=False)
        .agg(
            avg_price_rm=("price", "mean"),
            purchase_rate_pct=("purchase_rate_pct", "mean"),
            seat_takeup_pct=("seat_takeup_pct", "mean"),
            avg_revenue_rm=("revenue_rm", "mean"),
            revenue_count=("revenue_rm", "count"),
        )
        .reset_index()
    )


model, rmse, default_scenario = load_pipeline(get_routes_fingerprint())

st.write("")
ctrl_col, status_col = st.columns([1.15, 1.0], vertical_alignment="top")

with ctrl_col:
    with st.container(border=True):
        st.subheader("Controls")
        with st.form("controls"):
            route_id = st.selectbox(
                "Route",
                options=list(route_labels.keys()),
                format_func=lambda x: route_labels[x],
            )
            selected_route = route_lookup[route_id]

            c1, c2 = st.columns(2)
            with c1:
                days = st.slider("Days to Departure", 1, 60, 15)
                seats = st.slider(
                    "Seats Remaining",
                    10,
                    selected_route["capacity"],
                    min(150, selected_route["capacity"]),
                )
            with c2:
                weather = st.slider("Weather Score", 0.0, 1.0, 0.75)
                holiday = st.toggle("Holiday Period", value=False)

            price_range = st.slider("Price Sweep Range (RM)", 80, 1800, (120, 1200))
            run_forecast = st.form_submit_button("Run Forecast", use_container_width=True)

with status_col:
    with st.container(border=True):
        st.subheader("Scenario Snapshot")
        st.write(f"Route: **{selected_route['origin']} -> {selected_route['destination']}**")
        st.write(f"Seat capacity: **{selected_route['capacity']}**")
        st.write(f"Planning horizon: **{days} days**")
        st.write(f"Weather score: **{weather:.2f}**")
        st.write(f"Holiday period: **{'Yes' if holiday else 'No'}**")
        st.write(f"Price sweep: **RM {price_range[0]} - RM {price_range[1]}**")


if run_forecast:
    depart_date = (pd.Timestamp.now().normalize() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    market_price = fetch_market_price(
        selected_route["origin"], selected_route["destination"], depart_date
    )

    scenario = default_scenario.copy()
    scenario["route_id"] = route_id
    scenario["market_price"] = market_price
    scenario["seat_capacity"] = selected_route["capacity"]
    scenario["days_to_departure"] = days
    scenario["seats_remaining"] = seats
    scenario["weather_score"] = weather
    scenario["holiday_flag"] = int(holiday)

    opt = optimize_price(
        scenario,
        model,
        price_min=price_range[0],
        price_max=price_range[1],
        market_price=market_price,
    )
    seats_remaining = float(scenario["seats_remaining"].iloc[0])
    load_factor = (opt["expected_demand"] / max(seats_remaining, 1.0)) * 100
    rev_per_seat = opt["expected_revenue"] / max(seats_remaining, 1.0)
    demand_drop = 0.0
    if opt["demand_history"][0] > 0:
        demand_drop = (1 - (opt["demand_history"][-1] / opt["demand_history"][0])) * 100

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal Price", f"RM {opt['optimal_price']:.0f}")
    k2.metric("Expected Demand", f"{opt['expected_demand']:.0f}")
    k3.metric("Max Revenue", f"RM {opt['expected_revenue']:.2f}")
    k4.metric("Load Factor", f"{load_factor:.1f}%")

    st.caption(
        f"Model RMSE: {rmse:.2f} | Revenue per remaining seat: RM {rev_per_seat:.2f} | "
        f"Demand drop (min to max price): {demand_drop:.1f}% | "
        f"Market base fare: RM {opt['market_price']:.2f} | "
        f"Price band used: RM {opt['price_min_used']:.0f} - RM {opt['price_max_used']:.0f}"
    )

    sweep_df = build_sweep_df(opt, seats_remaining)
    quartile_summary = summarize_quartiles(sweep_df)
    best_quartile = quartile_summary.loc[
        quartile_summary["avg_revenue_rm"].idxmax(), "price_quartile"
    ]

    tab1, tab2, tab3 = st.tabs(["Curves", "Quartile Insights", "Forecast Table"])

    with tab1:
        v1, v2 = st.columns(2)
        with v1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(opt["price_history"], opt["revenue_history"], color="#0077b6", linewidth=2.3)
            ax1.axvline(opt["optimal_price"], color="#ff7f11", linestyle="--", linewidth=2)
            ax1.scatter([opt["optimal_price"]], [opt["expected_revenue"]], color="#ff7f11", s=80)
            ax1.set_xlabel("Price (RM)")
            ax1.set_ylabel("Revenue (RM)")
            ax1.set_title("Revenue vs Price")
            ax1.grid(True, alpha=0.25)
            st.pyplot(fig1)

        with v2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(opt["price_history"], opt["demand_history"], color="#2a9d8f", linewidth=2.3)
            ax2.set_xlabel("Price (RM)")
            ax2.set_ylabel("Predicted Demand")
            ax2.set_title("Demand vs Price")
            ax2.grid(True, alpha=0.25)
            st.pyplot(fig2)

    with tab2:
        st.write(
            f"Best revenue zone for this scenario: **{best_quartile}** "
            "(highest average quartile revenue)."
        )
        st.dataframe(
            quartile_summary.rename(
                columns={
                    "price_quartile": "Price Quartile",
                    "avg_price_rm": "Avg Price (RM)",
                    "purchase_rate_pct": "Rate to Purchase (%)",
                    "seat_takeup_pct": "Seat Take-up (%)",
                    "avg_revenue_rm": "Avg Revenue (RM)",
                    "revenue_count": "Revenue Count",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        show_df = sweep_df.copy()
        show_df = show_df.rename(
            columns={
                "price": "Price (RM)",
                "predicted_demand": "Predicted Demand",
                "revenue_rm": "Revenue (RM)",
                "user_views": "User Views",
                "conversion_rate": "Conversion Rate",
                "purchase_rate_pct": "Rate to Purchase (%)",
                "seat_takeup_pct": "Seat Take-up (%)",
                "price_quartile": "Price Quartile",
            }
        )
        st.dataframe(show_df, use_container_width=True, hide_index=True)
else:
    st.info("Set your scenario on top and click Run Forecast.")
