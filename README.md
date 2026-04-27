# Flight Fare Prediction Modelling

Airline demand forecasting and price optimization project with:
- Synthetic behavioral demand simulation
- Real-world external signals (weather + holidays, with fallback)
- RM (Malaysian Ringgit) pricing outputs
- Interactive Streamlit dashboard
- Excel-based route input

## What This Project Does

- Simulates demand for flight routes using:
  - Price
  - Days to departure
  - Seat capacity / seats remaining
  - Weather score
  - Holiday flag
  - Behavioral features (`user_views`, `conversion_rate`)
- Trains a regression model (LightGBM if available, otherwise sklearn fallback)
- Sweeps price range and finds price with maximum expected revenue
- Visualizes forecast and quartile-level insights in the dashboard

## Main Files

- `dashboard.py` - Streamlit app (main UI)
- `main.py` - CLI run with plots and summary
- `data_ingestion.py` - Weather/holiday APIs, route loading from Excel
- `data_generator.py` - Behavioral dataset simulation
- `model.py` - Demand model training
- `optimizer.py` - Price sweep optimizer

## Installation

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## API Key Setup

This project uses OpenWeatherMap.

Set your API key via environment variable:

```bash
export OPENWEATHER_API_KEY="your_key_here"
```

Or place it in a `.env` file at project root:

```env
OPENWEATHER_API_KEY=your_key_here
```

Optional (for market base fare via Amadeus API):

```env
AMADEUS_CLIENT_ID=your_amadeus_client_id
AMADEUS_CLIENT_SECRET=your_amadeus_client_secret
```

## Route Input From Excel

You can provide your own route master via `routes.xlsx` in the project root.

Required columns (exact logical fields):

`route_id | origin | destination | capacity`

Example:

- `101 | KUL | MYY | 180`
- `102 | KUL | KCH | 180`
- `103 | KUL | BKK | 220`

Notes:

- `origin` and `destination` will be normalized to uppercase.
- `capacity` must be positive.
- If file is missing/invalid, app falls back to default routes.
- Dashboard cache invalidates when `routes.xlsx` changes (file timestamp fingerprint).

## Run The Dashboard

```bash
./venv/bin/streamlit run dashboard.py
```

Dashboard includes:
- Scenario controls (route, planning days, seats, weather, holiday)
- RM-based price sweep
- KPI strip (optimal price, expected demand, max revenue, load factor)
- Curves tab (revenue vs price, demand vs price)
- Quartile insights (Q1 to Q4 purchase rate + revenue metrics)
- Forecast table (full sweep detail)

## Run The CLI Pipeline

```bash
./venv/bin/python main.py
```

Outputs:
- Console summary (optimal price, demand, revenue in RM)
- `optimization_results.png`

## Caching Behavior

- API responses are cached in `api_cache/`.
- Route geocoding is cached too.
- Market base fare (`market_{origin}_{destination}_{date}`) is cached too.
- If you update routes or want fresh API results:
  - rerun Streamlit
  - optionally clear relevant files in `api_cache/`

## Known Fallback Behavior

- If weather/holiday API is unreachable, the project uses realistic synthetic fallback values.
- If LightGBM cannot load (common `libomp` issue on macOS), model falls back automatically to `GradientBoostingRegressor`.

## Troubleshooting

1. `API returned 404 ...`
- Geocoding fallback is implemented, but invalid or unknown route tokens can still fail.
- Use valid airport/city-like route tokens in `routes.xlsx`.

2. `Library not loaded: libomp.dylib` (LightGBM on macOS)
- Fallback model already handles this, so project still runs.

3. Streamlit still showing old behavior/layout
- Stop app and rerun.
- Hard refresh browser (`Cmd+Shift+R`).

4. `routes.xlsx` updates not reflected
- Ensure required columns exist and values are valid.
- Restart Streamlit run command.

## Requirements

See `requirements.txt`:
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- requests
- matplotlib
- streamlit
- python-dotenv
- openpyxl
