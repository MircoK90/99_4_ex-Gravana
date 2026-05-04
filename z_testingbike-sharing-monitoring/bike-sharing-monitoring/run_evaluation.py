"""
run_evaluation.py

Loads the Bike Sharing UCI dataset, slices it by week of February 2011,
and POSTs each week to the bike-api /evaluate endpoint. Each POST updates
the regression-quality and drift gauges, which Prometheus picks up on the
next scrape (~15s).

Usage:
    python run_evaluation.py                  # send all February weeks
    python run_evaluation.py --fire-alert     # corrupt data so RMSE explodes
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from typing import Iterable, List, Tuple

import pandas as pd
import requests

API_URL = "http://localhost:8080"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)
FEATURE_COLS = [
    "season", "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "mnth", "hr", "weekday", "cnt",
]


def load_hour_csv() -> pd.DataFrame:
    """Download the UCI zip and return hour.csv as a DataFrame."""
    print(f"Downloading {UCI_URL} ...", flush=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as z:
        with z.open("hour.csv") as f:
            df = pd.read_csv(f, parse_dates=["dteday"])
    print(f"Loaded {len(df)} rows.", flush=True)
    return df


def slice_february_weeks(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """Return list of (label, dataframe) for each ISO-week of February 2011."""
    feb = df[(df["yr"] == 0) & (df["mnth"] == 2)].copy()
    feb["iso_week"] = feb["dteday"].dt.isocalendar().week
    out: List[Tuple[str, pd.DataFrame]] = []
    for week, sub in feb.groupby("iso_week", sort=True):
        if len(sub) < 5:
            # Skip stubs from week boundaries with too few rows
            continue
        out.append((f"feb_2011_w{int(week):02d}", sub))
    return out


def to_records(df: pd.DataFrame) -> List[dict]:
    """Subset to the API columns and convert to JSON-serializable records."""
    cleaned = df[FEATURE_COLS].copy()
    # Ensure ints are ints (some UCI columns may come back as floats after groupby)
    for col in ["season", "holiday", "workingday", "weathersit",
                "mnth", "hr", "weekday", "cnt"]:
        cleaned[col] = cleaned[col].astype(int)
    return cleaned.to_dict(orient="records")


def post_eval(label: str, records: Iterable[dict]) -> dict:
    payload = {"period_label": label, "records": list(records)}
    r = requests.post(f"{API_URL}/evaluate", json=payload, timeout=180)
    r.raise_for_status()
    body = r.json()
    print(f"  -> {label}: {json.dumps(body, indent=2)}", flush=True)
    return body


def fire_alert(df: pd.DataFrame) -> None:
    """Corrupt one week of February data so the model produces wildly wrong
    predictions. Used by `make fire-alert` as an alternative trigger path
    (the main fire-alert target uses the /trigger-drift endpoint instead --
    this is here for completeness / diversity of demo)."""
    sample = df[(df["yr"] == 0) & (df["mnth"] == 2)].head(168).copy()  # 1 week
    sample["temp"] = 0.99
    sample["atemp"] = 0.99
    sample["hum"] = 0.0
    sample["windspeed"] = 0.99
    sample["weathersit"] = 4
    # Inflate ground-truth so RMSE blows up regardless of model output
    sample["cnt"] = sample["cnt"] * 50
    print("Sending FIRE_ALERT_synthetic_drift batch (50x inflated counts)...")
    post_eval("FIRE_ALERT_synthetic_drift", to_records(sample))
    print("Alert state pushed. Watch Prometheus /alerts within ~30-60s.")


def wait_for_api(retries: int = 20, delay: float = 3.0) -> None:
    """Poll /health until the API is ready (model loaded)."""
    for i in range(retries):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                print("API is up and model is loaded.", flush=True)
                return
        except requests.RequestException:
            pass
        print(f"  [{i + 1}/{retries}] Waiting for API ({API_URL}) ...", flush=True)
        time.sleep(delay)
    print(f"ERROR: API at {API_URL} not responding after {retries} attempts.",
          file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-alert", action="store_true",
                        help="Send a corrupted batch instead of the normal weekly stream.")
    parser.add_argument("--api-url", default=API_URL,
                        help="Override the API base URL (default: %(default)s)")
    args = parser.parse_args()

    global API_URL  # noqa: PLW0603
    API_URL = args.api_url

    wait_for_api()
    df = load_hour_csv()

    if args.fire_alert:
        fire_alert(df)
        return

    weeks = slice_february_weeks(df)
    print(f"Sending {len(weeks)} weekly batches to {API_URL}/evaluate ...")
    for label, sub in weeks:
        post_eval(label, to_records(sub))
        time.sleep(1.0)  # gives Prometheus time to scrape between batches
    print("Done.")


if __name__ == "__main__":
    main()
