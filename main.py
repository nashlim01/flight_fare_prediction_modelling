"""
Airline Demand Forecasting & Pricing Optimizer (Real Data Integrated)
=====================================================================
This system combines real-world external signals (weather, holidays, route data, pricing benchmarks) 
with simulated behavioral demand to approximate airline revenue management.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_generator import simulate_behavioral_dataset
from model import train_demand_model, FEATURES
from optimizer import optimize_price
import matplotlib
matplotlib.use("Agg")  # Headless fallback, remove for local GUI

def plot_results(opt_result):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Price vs Demand
    axes[0].plot(opt_result["price_history"], opt_result["demand_history"], color="teal", linewidth=2)
    axes[0].set_xlabel("Ticket Price (RM)")
    axes[0].set_ylabel("Predicted Demand")
    axes[0].set_title("Price vs Demand Curve")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Price vs Revenue
    axes[1].plot(opt_result["price_history"], opt_result["revenue_history"], color="darkorange", linewidth=2)
    axes[1].plot(opt_result["optimal_price"], opt_result["expected_revenue"], "r*", markersize=15, label="Optimal")
    axes[1].set_xlabel("Ticket Price (RM)")
    axes[1].set_ylabel("Expected Revenue (RM)")
    axes[1].set_title("Price vs Revenue (Optimization)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Days to Departure vs Demand (Simulated Slice)
    days = np.arange(60, 1, -1)
    demands = []
    for d in days:
        row = opt_result["base_scenario"].copy()
        row["days_to_departure"] = d
        row_df = pd.DataFrame([row], columns=FEATURES)
        demands.append(trained_model.predict(row_df)[0])
    axes[2].plot(days, demands, color="crimson", linewidth=2)
    axes[2].set_xlabel("Days to Departure")
    axes[2].set_ylabel("Predicted Demand")
    axes[2].set_title("Proximity Effect: Days vs Demand")
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("optimization_results.png", dpi=300)
    print("📊 Plot saved to optimization_results.png")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    print("🚀 Generating dataset with real API integration & behavioral simulation...")
    df = simulate_behavioral_dataset(n_samples=6000)
    
    print("🤖 Training LightGBM Demand Model...")
    global trained_model  # For optimizer/plots scope
    trained_model, rmse = train_demand_model(df)
    
    print("🎯 Defining Scenario & Running Optimization...")
    # Pick a realistic mid-range scenario
    base_row = df[(df["route_id"] == 101) & (df["days_to_departure"] == 30)].iloc[0].copy()
    base_row = base_row[FEATURES]
    
    opt = optimize_price(base_row, trained_model, price_min=80, price_max=1800)
    base_row.name = "base_scenario"
    opt["base_scenario"] = base_row
    
    print("\n" + "="*50)
    print(f"💰 OPTIMAL PRICE:       RM {opt['optimal_price']:.2f}")
    print(f"👥 EXPECTED DEMAND:     {opt['expected_demand']:.0f} passengers")
    print(f"📈 EXPECTED REVENUE:    RM {opt['expected_revenue']:.2f}")
    print("="*50)
    
    plot_results(opt)
    print("✅ Pipeline Complete.")
