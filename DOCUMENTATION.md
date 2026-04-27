# Flight Fare Prediction Modelling — Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Process Flow](#3-process-flow)
4. [File-by-File Reference](#4-file-by-file-reference)
   - [data_ingestion.py](#41-data_ingestionpy)
   - [feature_engineering.py](#42-feature_engineeringpy)
   - [data_generator.py](#43-data_generatorpy)
   - [model.py](#44-modelpy)
   - [optimizer.py](#45-optimizerpy)
   - [main.py](#46-mainpy)
   - [dashboard.py](#47-dashboardpy)
   - [api.env](#48-apienv)
   - [api_cache/](#49-api_cache)
5. [Data Model & Features](#5-data-model--features)
6. [Core Algorithms & Formulae](#6-core-algorithms--formulae)
7. [External API Integrations](#7-external-api-integrations)
8. [Dependencies](#8-dependencies)
9. [Known Design Decisions & Caveats](#9-known-design-decisions--caveats)

---

## 1. Project Overview

This project is a **Malaysia-focused airline demand forecasting and dynamic pricing optimizer**. It simulates realistic booking behaviour across six routes departing from Kuala Lumpur (KUL), trains a gradient-boosted regression model on that data, and then sweeps a price grid to identify the ticket price that maximises expected revenue for a given flight scenario.

The project has two operating modes:

| Mode | Entry Point | Output |
|------|-------------|--------|
| CLI / Batch | `main.py` | Console metrics + `optimization_results.png` |
| Interactive Web Dashboard | `dashboard.py` (Streamlit) | Browser UI with live charts and tables |

Both modes share the same data generation, feature engineering, model training, and optimisation logic — they only differ in how results are presented.

---

## 2. System Architecture

```
External APIs
  ├── OpenWeatherMap  ──────────────────────────────┐
  └── Nager.Date (Public Holidays)  ────────────────┤
                                                    ▼
                                          data_ingestion.py
                                         (fetch + cache layer)
                                                    │
                                                    ▼
                                        feature_engineering.py
                                       (weather score + schema)
                                                    │
                                                    ▼
                                          data_generator.py
                                    (5000–6000 synthetic records)
                                                    │
                                                    ▼
                                             model.py
                                    (LightGBM / GBR training)
                                                    │
                                                    ▼
                                            optimizer.py
                                       (price sweep → revenue)
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                              ▼                                             ▼
                           main.py                                   dashboard.py
                  (CLI output + PNG chart)                  (Streamlit interactive UI)
```

---

## 3. Process Flow

### Step 1 — Data Ingestion (`data_ingestion.py`)

The pipeline begins by loading external real-world signals. For each flight in the simulation:

- Weather data for the **origin** and **destination** airport cities is fetched from OpenWeatherMap's current weather endpoint.
- Public holiday status for the departure date is checked via the Nager.Date API (defaulting to Malaysian holidays, `MY`).
- All API responses are cached locally to `./api_cache/` as JSON files, keyed by `{type}_{code}_{date}`. Subsequent runs for the same date/location are served from cache, avoiding redundant API calls and keeping the simulation reproducible.
- If an API call fails (network timeout, bad status code, missing API key), a **statistical fallback** is used: weather values are sampled from realistic distributions, and a hardcoded list of known Malaysian public holidays is checked instead.

Six fixed routes are defined here as well (all originating from KUL).

### Step 2 — Feature Engineering (`feature_engineering.py`)

Two roles:

1. **`compute_weather_score()`**: Converts raw weather dicts (temperature, precipitation, wind speed, sky condition) into a single normalised score between 0 and 1. This score represents how "flight-demand-friendly" the weather is. Extreme conditions (thunderstorm, heavy rain) trigger an additional 40% penalty.

2. **`engineer_features()`**: Takes raw route/date/holiday inputs and produces the canonical 10-feature + 1-target DataFrame schema used throughout the project. Primarily used for schema initialisation; the actual feature values are populated by `data_generator.py`.

### Step 3 — Synthetic Data Generation (`data_generator.py`)

This is the heart of the training data pipeline. Because no historical airline booking dataset is bundled, the project **simulates** behavioural demand using econometric-style relationships:

- **User views**: Modelled as an exponential decay as `days_to_departure` increases (people search more as the flight approaches), amplified by holiday periods and good weather.
- **Conversion rate**: Modelled as a function of price (exponential decay — higher price → lower conversion) and seat scarcity (urgency effect — fewer seats remaining → higher conversion).
- **Demand**: `views × conversion_rate`, capped at `seats_remaining`.

5,000–6,000 records are generated by cycling through the 6 routes, using real weather/holiday data from the API (cached), and applying controlled random noise for variability.

### Step 4 — Model Training (`model.py`)

The dataset is split 80/20 into train and test sets. A **LightGBM Regressor** is trained to predict `demand` from the 10 input features. If LightGBM is not installed, the pipeline falls back transparently to scikit-learn's `GradientBoostingRegressor` with equivalent hyperparameters. RMSE on the test set is printed and returned.

### Step 5 — Price Optimisation (`optimizer.py`)

Given a specific flight scenario (a row of feature values), the optimizer:

1. Sweeps ticket prices from `price_min` to `price_max` in steps of RM 5.
2. For each candidate price, **recomputes** the behavioural features (`user_views`, `conversion_rate`) that are causally affected by price — ensuring the model sees a consistent, realistic feature vector.
3. Feeds the updated feature row to the trained model to get a `demand` prediction.
4. Caps demand at `seats_remaining` (physical constraint).
5. Computes `revenue = price × capped_demand`.
6. Returns the price that maximises revenue, along with full sweep histories for visualisation.

### Step 6A — CLI Output (`main.py`)

Runs all steps above, prints the optimal price/demand/revenue summary to the console, and generates a 3-panel matplotlib figure (`optimization_results.png`):
- Panel 1: Price vs Demand curve
- Panel 2: Price vs Revenue curve with the optimal point marked
- Panel 3: Days-to-departure effect on predicted demand

### Step 6B — Interactive Dashboard (`dashboard.py`)

A Streamlit application that wraps the same pipeline with a browser UI. The model is trained once and cached (`@st.cache_resource`). Users can then:
- Select a route, adjust days to departure, seats remaining, weather score, and holiday flag via sliders/toggles.
- Set a price sweep range.
- Click **Run Forecast** to see:
  - Key metrics (optimal price, expected demand, max revenue, load factor).
  - Revenue vs Price and Demand vs Price charts.
  - A quartile breakdown of the price sweep.
  - A full tabular forecast sweep.

---

## 4. File-by-File Reference

### 4.1 `data_ingestion.py`

**Purpose:** External data layer — fetches, caches, and serves real-world API data.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_load_cache` | `(key: str) → dict \| None` | Reads a JSON cache file from `./api_cache/` by key. Returns `None` if not found. |
| `_save_cache` | `(key: str, data: dict)` | Writes a dict to `./api_cache/{key}.json`. |
| `fetch_weather` | `(iata_code: str, date: str) → dict` | Returns `{temp, precip_mm, wind_speed, condition}` for the city mapped from the IATA code. Uses cache first; falls back to statistical sampling on API failure. |
| `fetch_holiday` | `(date_str: str, country: str = "MY") → bool` | Returns `True` if the given date is a public holiday in the specified country. Caches the full year's holiday list. |
| `get_routes` | `() → list[dict]` | Returns the static list of 6 KUL-origin routes with route IDs and seat capacities. |

**Key constants:**
- `WEATHER_API_KEY`: Loaded from `OPENWEATHER_API_KEY` env var, with a hardcoded fallback key for testing.
- `IATA_TO_CITY`: Maps IATA airport codes to city names suitable for the OpenWeatherMap free tier query format.

**Route table:**

| Route ID | Origin | Destination | Capacity |
|----------|--------|-------------|----------|
| 101 | KUL | MYY (Miri) | 180 |
| 102 | KUL | KCH (Kuching) | 180 |
| 103 | KUL | BKK (Bangkok) | 220 |
| 104 | KUL | SGP (Singapore) | 220 |
| 105 | KUL | BKK (Bangkok) | 180 |
| 106 | KUL | TPE (Taipei) | 220 |

---

### 4.2 `feature_engineering.py`

**Purpose:** Converts raw API data into model-ready features and defines the canonical feature schema.

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_weather_score` | `(weather: dict, dest_weather: dict = None) → float` | Computes a composite 0–1 weather score from both origin and destination weather. Applies a disruption penalty (×0.6) for thunderstorm, snow, or heavy rain (>8mm). |
| `engineer_features` | `(df_raw, depart_dates, route_data, holiday_flag) → DataFrame` | Builds the feature schema DataFrame from route and date inputs. Initialises numeric columns to 0 (populated later by `data_generator.py`). |

**Weather score formula:**

```
temp_score    = exp(-((avg_temp - 25)² / 100))   # peaks at 25°C
precip_score  = max(0, 1 - avg_precip / 10)
wind_score    = max(0, 1 - avg_wind / 15)
cond_score    = avg(CONDITION_MAP[origin_cond], CONDITION_MAP[dest_cond])

score = 0.3 × temp_score + 0.3 × precip_score + 0.2 × wind_score + 0.2 × cond_score
```

**Condition map:**

| Condition | Score |
|-----------|-------|
| Clear | 1.0 |
| Clouds | 0.9 |
| Haze | 0.8 |
| Mist | 0.7 |
| Rain | 0.5 |
| Snow | 0.3 |
| Thunderstorm | 0.2 |

---

### 4.3 `data_generator.py`

**Purpose:** Generates the synthetic behavioural training dataset by simulating realistic booking patterns.

| Function | Signature | Description |
|----------|-----------|-------------|
| `simulate_behavioral_dataset` | `(n_samples: int = 5000, seed: int = 42) → DataFrame` | Generates `n_samples` rows of flight + booking behaviour data. Each row corresponds to one flight booking scenario. |

**Simulation logic per record:**

```python
# Search interest (views)
base_views    = 450 × exp(-days / 22)
holiday_mult  = 1.5 if holiday else 1.0
weather_mult  = 1.2 × weather_score + 0.8
views         = base_views × holiday_mult × weather_mult × noise[0.85, 1.15]

# Booking probability (conversion)
scarcity      = 1 + 0.6 × (1 - seats_remaining / capacity)
price_penalty = exp(-price / 220)
conv_rate     = clip(price_penalty × scarcity × noise[0.8, 1.1], 0.01, 0.85)

# Realised demand
demand = clip(views × conv_rate × noise[0.95, 1.05], 0, seats_remaining)
```

**Output columns:** `route_id`, `price`, `seat_capacity`, `seats_remaining`, `user_views`, `conversion_rate`, `weather_score`, `holiday_flag`, `days_to_departure`, `day_of_week`, `demand`

---

### 4.4 `model.py`

**Purpose:** Trains the demand prediction model.

| Function | Signature | Description |
|----------|-----------|-------------|
| `train_demand_model` | `(df: DataFrame) → (model, rmse)` | Splits data 80/20, trains LightGBM (or GBR fallback), returns the fitted model and test RMSE. |

**Model selection:**
- **Primary**: `LightGBM LGBMRegressor` — `n_estimators=300`, `learning_rate=0.05`, `max_depth=8`, `min_child_samples=10`
- **Fallback**: `sklearn GradientBoostingRegressor` — `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`

**Feature list (`FEATURES`):**

```python
["price", "days_to_departure", "route_id", "seat_capacity",
 "seats_remaining", "user_views", "conversion_rate", "weather_score",
 "holiday_flag", "day_of_week"]
```

**Target:** `demand` (integer passenger count)

---

### 4.5 `optimizer.py`

**Purpose:** Finds the revenue-maximising ticket price for a given scenario by sweeping the price space.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_refresh_behavioral_features` | `(row: Series, price: float) → Series` | Internal helper. Recomputes `user_views` and `conversion_rate` for a specific price point, maintaining causal consistency. |
| `optimize_price` | `(scenario_features, trained_model, price_min=100, price_max=1000, step=5) → dict` | Sweeps prices from `price_min` to `price_max` in `step` increments. Returns the optimal price and full sweep histories. |

**Return dict keys:**
- `optimal_price` — RM value that maximises revenue
- `expected_demand` — predicted passengers at optimal price
- `expected_revenue` — RM revenue at optimal price
- `price_history`, `demand_history`, `revenue_history` — full sweep arrays for plotting
- `user_views_history`, `conversion_rate_history` — behavioural sweep histories

---

### 4.6 `main.py`

**Purpose:** CLI entry point. Orchestrates the full pipeline end-to-end and produces the output chart.

**Execution flow:**
1. Calls `simulate_behavioral_dataset(n_samples=6000)`.
2. Calls `train_demand_model(df)`.
3. Selects a representative base scenario: Route 101, `days_to_departure == 30`.
4. Calls `optimize_price(base_row, trained_model, price_min=80, price_max=1800)`.
5. Prints optimal price, expected demand, and expected revenue.
6. Calls `plot_results()` to generate a 3-panel figure saved as `optimization_results.png`.

**Chart panels:**
- **Panel 1** — Price vs Demand (teal line)
- **Panel 2** — Price vs Revenue with the optimal point marked as a red star (orange line)
- **Panel 3** — Days-to-Departure vs Demand (inverted x-axis, showing how demand rises as departure approaches)

---

### 4.7 `dashboard.py`

**Purpose:** Interactive Streamlit web application wrapping the same pipeline with a full UI.

**Layout:**
- **Header**: Custom-styled headline with Space Grotesk font, warm cream background gradient.
- **Controls panel** (left): Route selector, days to departure slider, seats remaining slider, weather score slider, holiday toggle, price sweep range slider, "Run Forecast" button.
- **Scenario Snapshot panel** (right): Live summary of current control values.
- **Metrics row**: 4 KPI cards — Optimal Price, Expected Demand, Max Revenue, Load Factor.
- **Tabbed results**:
  - *Curves* — Revenue vs Price and Demand vs Price matplotlib charts.
  - *Quartile Insights* — Price range divided into Q1–Q4 with average metrics per quartile.
  - *Forecast Table* — Full sweep DataFrame with all computed columns.

**Caching:** The model training pipeline is wrapped in `@st.cache_resource`, so it runs only once per server session regardless of how many times the user clicks "Run Forecast".

**Key helper functions:**

| Function | Description |
|----------|-------------|
| `load_pipeline()` | Generates 4,000 training samples, trains the model, returns model + RMSE + a default scenario row. |
| `build_sweep_df()` | Converts the optimizer's output dict into a structured DataFrame with derived columns (`purchase_rate_pct`, `seat_takeup_pct`, `price_quartile`). |
| `summarize_quartiles()` | Aggregates sweep data by price quartile for the Quartile Insights tab. |

---

### 4.8 `api.env`

**Purpose:** Stores the OpenWeatherMap API key.

```
OPENWEATHER_API_KEY=your_key_here
```

> ⚠️ **Security note:** This file is committed to the repository and contains a real API key as a fallback. For production use, this file should be added to `.gitignore` and the key managed via environment variables or a secrets manager.

---

### 4.9 `api_cache/`

**Purpose:** Local JSON cache directory for API responses.

Cache files follow the naming convention:
- `weather_{IATA}_{YYYY-MM-DD}.json` — weather data per airport per date
- `holiday_{COUNTRY}_{YYYY}.json` — full year's public holiday list per country

This directory can be safely deleted to force fresh API calls on the next run.

---

## 5. Data Model & Features

| Feature | Type | Description |
|---------|------|-------------|
| `price` | float | Ticket price in RM (120–1800 during generation; sweep range configurable) |
| `days_to_departure` | int | Calendar days from query date to flight departure (1–60) |
| `route_id` | int | Integer identifier for the origin-destination pair (101–106) |
| `seat_capacity` | int | Total seats on the aircraft (180 or 220) |
| `seats_remaining` | int | Available unsold seats at time of query |
| `user_views` | int | Simulated number of users who viewed this flight's price |
| `conversion_rate` | float | Fraction of viewers predicted to book (0.01–0.85) |
| `weather_score` | float | Composite weather quality index, 0–1 (computed by `feature_engineering.py`) |
| `holiday_flag` | int | 1 if departure date is a Malaysian public holiday, 0 otherwise |
| `day_of_week` | int | ISO weekday of departure (0 = Monday, 6 = Sunday) |
| **`demand`** | int | **Target variable** — realised passenger bookings (capped at `seats_remaining`) |

---

## 6. Core Algorithms & Formulae

### Demand Simulation (Training Data)
```
views         = 450 × exp(-days/22) × holiday_mult × weather_mult
conversion    = exp(-price/220) × (1 + 0.6 × (1 - seats_rem/capacity))
demand        = clip(views × conversion, 0, seats_remaining)
```

### Weather Score
```
score = 0.3×temp_score + 0.3×precip_score + 0.2×wind_score + 0.2×cond_score
      × 0.6 (disruption penalty if thunderstorm/snow/heavy rain)
```

### Price Optimisation
```
for price in [price_min, price_min+5, ..., price_max]:
    recompute views(price), conversion(price)
    demand = model.predict(features_with_price)
    revenue = price × min(demand, seats_remaining)
return price at argmax(revenue)
```

### Dashboard Derived Metrics
```
load_factor       = expected_demand / seats_remaining × 100 (%)
rev_per_seat      = expected_revenue / seats_remaining
demand_drop       = (1 - demand_at_max_price / demand_at_min_price) × 100 (%)
purchase_rate_pct = predicted_demand / user_views × 100 (%)
seat_takeup_pct   = predicted_demand / seats_remaining × 100 (%)
```

---

## 7. External API Integrations

| API | Provider | Endpoint | Usage | Fallback |
|-----|----------|----------|-------|----------|
| Current Weather | OpenWeatherMap (free tier) | `/data/2.5/weather?q={city}` | Temperature, precipitation, wind, sky condition | Random statistical samples |
| Public Holidays | Nager.Date (free, no key) | `/api/v3/publicholidays/{year}/{country}` | Check if departure date is a public holiday | Hardcoded MY holiday list |

---

## 8. Dependencies

| Package | Role |
|---------|------|
| `numpy` | Numerical computations, random generation |
| `pandas` | DataFrame operations throughout |
| `scikit-learn` | Train/test split, RMSE metric, GBR fallback model |
| `lightgbm` | Primary demand prediction model |
| `matplotlib` | CLI output charts |
| `streamlit` | Interactive web dashboard |
| `requests` | HTTP calls to external APIs |
| `python-dotenv` | Loading API keys from `api.env` |

---

## 9. Known Design Decisions & Caveats

**Synthetic training data:** The model is trained on simulated data, not historical airline booking records. The behavioural relationships (exponential view decay, price-elasticity via `exp(-price/220)`, scarcity urgency) are economically plausible but not empirically calibrated to a specific airline or market. Predictions should be treated as relative indicators rather than absolute forecasts.

**Weather data is current, not forecast:** The project calls OpenWeatherMap's *current* weather endpoint. For future departure dates, this returns today's weather at that location — not a forecast. This is a known limitation; integrating a forecast API would improve realism.

**Static route table:** Routes and capacities are hardcoded in `data_ingestion.py`. To add new routes, both `get_routes()` and `IATA_TO_CITY` must be updated.

**API key in repository:** `api.env` contains a real OpenWeatherMap key committed to the public repo. This is an intentional convenience for immediate testing but is a security concern for production use.

**`venv/` in repository:** The Python virtual environment folder is committed to git. This inflates repo size significantly and is generally discouraged. Adding `venv/` to `.gitignore` is recommended.

**Global variable in `main.py`:** `trained_model` is declared as a global inside `__main__` to make it accessible to `plot_results()`. This is a minor code smell; passing the model as a parameter to `plot_results()` would be cleaner.
