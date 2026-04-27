import os
import json
import requests
import numpy as np
import pandas as pd
import re
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# API Key integrated (loaded from .env, falls back to your key for immediate testing)
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "ff2e283169dc4bf2b587d0766be7c249")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET", "")

CACHE_DIR = Path("./api_cache")
CACHE_DIR.mkdir(exist_ok=True)
ROUTES_FILE = Path("./routes.xlsx")

# Reliable IATA → City mapping for OpenWeatherMap free tier
IATA_TO_CITY = {
    "KUL": "Kuala Lumpur",
    "MYY": "Miri",
    "KCH": "Kuching",
    "BKK": "Bangkok",
    "SIN": "Singapore",
    "SGP": "Singapore",
    "TPE": "Taipei"
}

# Helpful aliases for non-IATA location tokens that may appear in Excel routes.
LOCATION_ALIASES = {
    "JPN": "Tokyo",
    "MYS": "Kuala Lumpur",
    "THA": "Bangkok",
    "SGP": "Singapore",
    "TWN": "Taipei",
    "KOR": "Seoul",
    "SAW": "Istanbul",
    "NRT": "Tokyo",
    "HND": "Tokyo",
    "BTU": "Bintulu",
}

ROUTE_BASELINE = {
    ("KUL", "SIN"): 180.0,
    ("KUL", "BKK"): 350.0,
    ("KUL", "TPE"): 750.0,
    ("KUL", "KCH"): 220.0,
    ("KUL", "MYY"): 250.0,
}

def _load_cache(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def _save_cache(key: str, data: dict):
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _safe_cache_key(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip().lower())

def _resolve_location_query(location_code: str) -> str:
    code = str(location_code).strip().upper()
    return IATA_TO_CITY.get(code, LOCATION_ALIASES.get(code, code))

def _geocode_location(location_code: str) -> Optional[dict]:
    """
    Resolve a route token (IATA/city/country-like code) to lat/lon using OpenWeather geocoding.
    """
    code = str(location_code).strip().upper()
    cache_key = f"geocode_{_safe_cache_key(code)}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    base_query = _resolve_location_query(code)
    query_candidates = [base_query]
    if code != base_query:
        query_candidates.append(code)
    if len(code) == 3:
        query_candidates.append(f"{base_query} airport")

    for q in query_candidates:
        try:
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            resp = requests.get(
                geo_url,
                params={"q": q, "limit": 1, "appid": WEATHER_API_KEY},
                timeout=10
            )
            if resp.status_code != 200:
                continue
            items = resp.json()
            if not items:
                continue

            best = items[0]
            result = {
                "lat": best["lat"],
                "lon": best["lon"],
                "name": best.get("name", q),
                "country": best.get("country", "")
            }
            _save_cache(cache_key, result)
            return result
        except Exception:
            continue

    return None

def fetch_weather(iata_code: str, date: str) -> dict:
    """Fetch real-time weather using geocoded lat/lon with caching & graceful fallback."""
    cache_key = f"weather_{iata_code}_{date}"
    cached = _load_cache(cache_key)
    if cached: return cached

    loc = _geocode_location(iata_code)
    resolved_name = _resolve_location_query(iata_code)
    if loc:
        resolved_name = f"{loc.get('name', resolved_name)} {loc.get('country', '')}".strip()
        url = "http://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "lat": loc["lat"],
            "lon": loc["lon"],
            "units": "metric",
            "appid": WEATHER_API_KEY
        }
    else:
        url = "http://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "q": resolved_name,
            "units": "metric",
            "appid": WEATHER_API_KEY
        }
    
    try:
        resp = requests.get(url, params=weather_params, timeout=10)
        if resp.status_code == 200:
            d = resp.json()
            data = {
                "temp": d["main"]["temp"],
                "precip_mm": d.get("rain", {}).get("1h", d.get("rain", {}).get("3h", 0)),
                "wind_speed": d["wind"]["speed"],
                "condition": d["weather"][0]["main"]
            }
            _save_cache(cache_key, data)
            print(f"[✅] Real weather fetched for {resolved_name} ({iata_code})")
            return data
        else:
            print(f"[⚠️] API returned {resp.status_code} for {resolved_name}. Using fallback.")
    except Exception as e:
        print(f"[⚠️] Request failed for {resolved_name}: {e}")

    # Realistic statistical fallback (preserves reproducibility)
    fallback = {
        "temp": np.random.uniform(5, 35),
        "precip_mm": np.random.exponential(2),
        "wind_speed": np.random.uniform(2, 20),
        "condition": np.random.choice(["Clear", "Clouds", "Rain", "Mist"])
    }
    _save_cache(cache_key, fallback)
    return fallback

def fetch_holiday(date_str: str, country: str = "MY") -> bool:
    cache_key = f"holiday_{country}_{date_str[:4]}"
    cached = _load_cache(cache_key)
    if cached:
        return any(h["date"] == date_str for h in cached)

    try:
        resp = requests.get(f"https://date.nager.at/api/v3/publicholidays/{date_str[:4]}/{country}", timeout=5)
        if resp.status_code == 200:
            _save_cache(cache_key, resp.json())
            return any(h["date"] == date_str for h in resp.json())
    except Exception: pass
    
    print("[⚠️] Holiday API unreachable. Using fallback dates.")
    fallback = ["2026-01-29", "2026-02-17", "2026-05-01", "2026-08-31", "2026-09-16", "2026-12-25"]
    _save_cache(cache_key, [{"date": d} for d in fallback])
    return date_str in fallback

def _get_amadeus_access_token() -> Optional[str]:
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        return None

    token_cache_key = "amadeus_token"
    token_cached = _load_cache(token_cache_key)
    if token_cached and token_cached.get("access_token"):
        expires_at = float(token_cached.get("expires_at", 0))
        if time.time() < (expires_at - 60):
            return str(token_cached["access_token"])

    try:
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": AMADEUS_CLIENT_ID,
                "client_secret": AMADEUS_CLIENT_SECRET,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 1800))
        if not access_token:
            return None

        _save_cache(
            token_cache_key,
            {
                "access_token": access_token,
                "expires_at": time.time() + expires_in,
            },
        )
        return access_token
    except Exception:
        return None

def _fallback_market_price(origin: str, destination: str) -> float:
    base = ROUTE_BASELINE.get((origin, destination), 300.0)
    return float(base * np.random.uniform(0.9, 1.1))

def fetch_market_price(origin: str, destination: str, date: str) -> float:
    """
    Returns an estimated market base fare (RM) for a given route and date.
    Uses API first, then falls back to static calibrated values.
    """
    origin = str(origin).strip().upper()
    destination = str(destination).strip().upper()
    cache_key = f"market_{origin}_{destination}_{date}"
    cached = _load_cache(cache_key)
    if cached and "market_price" in cached:
        return float(cached["market_price"])

    token = _get_amadeus_access_token()
    if token:
        try:
            offers_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            headers = {"Authorization": f"Bearer {token}"}
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": date,
                "adults": 1,
                "currencyCode": "MYR",
                "max": 20,
            }
            resp = requests.get(offers_url, headers=headers, params=params, timeout=20)
            if resp.status_code == 200:
                payload = resp.json()
                response_data = payload.get("data", [])
                if response_data:
                    market_price = min(float(flight["price"]["total"]) for flight in response_data)
                    _save_cache(
                        cache_key,
                        {
                            "market_price": market_price,
                            "source": "amadeus",
                            "origin": origin,
                            "destination": destination,
                            "date": date,
                        },
                    )
                    return market_price
        except Exception:
            pass

    market_price = _fallback_market_price(origin, destination)
    _save_cache(
        cache_key,
        {
            "market_price": market_price,
            "source": "fallback",
            "origin": origin,
            "destination": destination,
            "date": date,
        },
    )
    return market_price

DEFAULT_ROUTES = [
        {"route_id": 101, "origin": "KUL", "destination": "MYY", "capacity": 180},
        {"route_id": 102, "origin": "KUL", "destination": "KCH", "capacity": 180},
        {"route_id": 103, "origin": "KUL", "destination": "BKK", "capacity": 220},
        {"route_id": 104, "origin": "KUL", "destination": "SIN", "capacity": 220},
        {"route_id": 105, "origin": "KUL", "destination": "BKK", "capacity": 180},
        {"route_id": 106, "origin": "KUL", "destination": "TPE", "capacity": 220}
]

def get_routes_fingerprint() -> str:
    """Fingerprint used to invalidate dashboard cache when routes.xlsx changes."""
    if ROUTES_FILE.exists():
        return str(int(ROUTES_FILE.stat().st_mtime))
    return "default_routes"

def get_routes() -> list:
    """
    Load routes from routes.xlsx if present.
    Required columns: route_id | origin | destination | capacity
    """
    if not ROUTES_FILE.exists():
        return DEFAULT_ROUTES

    required_cols = {"route_id", "origin", "destination", "capacity"}
    try:
        routes_df = pd.read_excel(ROUTES_FILE)
        routes_df.columns = [str(c).strip().lower() for c in routes_df.columns]
        missing = required_cols.difference(routes_df.columns)
        if missing:
            print(f"[⚠️] routes.xlsx missing columns: {sorted(missing)}. Using default routes.")
            return DEFAULT_ROUTES

        routes_df = routes_df[list(required_cols)].dropna()
        routes_df["route_id"] = routes_df["route_id"].astype(int)
        routes_df["origin"] = routes_df["origin"].astype(str).str.strip().str.upper()
        routes_df["destination"] = routes_df["destination"].astype(str).str.strip().str.upper()
        routes_df["capacity"] = routes_df["capacity"].astype(int)
        routes_df = routes_df[routes_df["capacity"] > 0]

        if routes_df.empty:
            print("[⚠️] routes.xlsx has no valid rows. Using default routes.")
            return DEFAULT_ROUTES

        return routes_df.sort_values("route_id").to_dict(orient="records")
    except Exception as e:
        print(f"[⚠️] Could not load routes.xlsx ({e}). Using default routes.")
        return DEFAULT_ROUTES
