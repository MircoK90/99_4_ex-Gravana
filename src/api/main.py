"""
Bike Sharing prediction API.

- Trains a RandomForestRegressor on January 2011 (reference window).
- Exposes /predict for inference and /evaluate for batch monitoring.
- Instruments Prometheus metrics: request counter / latency histogram /
  regression-quality gauges (RMSE, MAE, R2) + custom drift gauges.
- Uses Evidently 0.4.16 (RegressionPreset + DataDriftPreset) to compute
  the regression quality on a current batch and the dataset drift
  between January (reference) and the current batch.
"""

from __future__ import annotations

import io
import logging
import os
import time
import urllib.request
import zipfile
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

# Evidently 0.4.x API
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("bike-api")

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "model.joblib"))
REFERENCE_DATA_PATH = os.environ.get(
    "REFERENCE_DATA_PATH", os.path.join(ARTIFACTS_DIR, "reference.parquet")
)
UCI_URL = os.environ.get(
    "UCI_BIKE_URL",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
)

# Feature lists matching the exam specification
NUMERICAL_FEATURES: List[str] = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CATEGORICAL_FEATURES: List[str] = ["season", "holiday", "workingday", "weathersit"]
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "cnt"
PREDICTION = "prediction"

COLUMN_MAPPING = ColumnMapping(
    target=TARGET,
    prediction=PREDICTION,
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
# We use a custom CollectorRegistry so our /metrics endpoint exposes ONLY
# the metrics we explicitly defined (no default process / gc collectors
# polluting the dashboards).
REGISTRY = CollectorRegistry()

REQ_COUNTER = Counter(
    "api_requests_total",
    "Total number of API requests received.",
    ["endpoint", "method", "status_code"],
    registry=REGISTRY,
)

REQ_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds.",
    ["endpoint", "method", "status_code"],
    # Buckets tuned for typical FastAPI latencies (1ms .. 5s)
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

MODEL_RMSE = Gauge(
    "model_rmse_score",
    "Root Mean Squared Error of the regression model on the latest evaluation batch.",
    registry=REGISTRY,
)
MODEL_MAE = Gauge(
    "model_mae_score",
    "Mean Absolute Error of the regression model on the latest evaluation batch.",
    registry=REGISTRY,
)
MODEL_R2 = Gauge(
    "model_r2_score",
    "R^2 score of the regression model on the latest evaluation batch.",
    registry=REGISTRY,
)

# -------- Custom metrics (justification below) -------------------------------
# `evidently_dataset_drift_share` (0..1): the fraction of monitored features
# Evidently flags as drifted on the current batch. Chosen because it is a
# single, normalized indicator that summarises whether the *overall* input
# distribution is shifting -- this is exactly the signal that should trigger
# retraining or human review in production. It is also alert-friendly:
# threshold > 0.5 ==> dataset drift.
DRIFT_SHARE = Gauge(
    "evidently_dataset_drift_share",
    "Share of features detected as drifted by Evidently on the latest batch (0..1).",
    registry=REGISTRY,
)
# Boolean flag (Evidently's own dataset_drift verdict) for binary alerting.
DRIFT_DETECTED = Gauge(
    "evidently_dataset_drift_detected",
    "1 if Evidently considers the dataset to have drifted on the latest batch, else 0.",
    registry=REGISTRY,
)

# Number of evaluations performed since startup. Useful to detect the
# presence/absence of monitoring traffic.
EVAL_COUNTER = Counter(
    "model_evaluations_total",
    "Total number of /evaluate calls processed.",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class BikeSharingInput(BaseModel):
    """A single record of bike-sharing features (no target)."""

    season: int = Field(..., ge=1, le=4)
    holiday: int = Field(..., ge=0, le=1)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float
    atemp: float
    hum: float
    windspeed: float
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    weekday: int = Field(..., ge=0, le=6)


class EvaluationRecord(BikeSharingInput):
    """Like BikeSharingInput but includes the actual target -- needed to
    compute regression quality metrics on the current batch."""

    cnt: int = Field(..., ge=0)



class EvaluationRequest(BaseModel):
    # Exam format
    records: Optional[List[EvaluationRecord]] = None
    period_label: Optional[str] = None

    # Original exercise format
    data: Optional[List[dict]] = None
    evaluation_period_name: Optional[str] = None

    def normalized_records(self) -> List[EvaluationRecord]:
        """Return a unified list of EvaluationRecord objects."""
        if self.records is not None:
            return self.records

        if self.data is not None:
            # Convert dicts → EvaluationRecord
            out = []
            for item in self.data:
                # remove dteday if present
                item = {k: v for k, v in item.items() if k != "dteday"}
                out.append(EvaluationRecord(**item))
            return out

        raise ValueError("No valid record list found.")
    
    def normalized_label(self) -> Optional[str]:
        return self.period_label or self.evaluation_period_name


class PredictionOutput(BaseModel):
    prediction: float


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int
    # period_label: Optional[str]
    # n_records: int
    # rmse: Optional[float]
    # mae: Optional[float]
    # r2: Optional[float]
    # drift_share: Optional[float]
    # dataset_drift: Optional[bool]


# ---------------------------------------------------------------------------
# Data fetching, processing & training
# ---------------------------------------------------------------------------
def _fetch_data() -> pd.DataFrame:
    """Download the UCI Bike Sharing hour.csv and return it as a DataFrame.

    Falls back to OpenML if the UCI archive is unreachable.
    """
    logger.info("Fetching Bike Sharing dataset from UCI ...")
    try:
        with urllib.request.urlopen(UCI_URL, timeout=30) as resp:
            buf = io.BytesIO(resp.read())
        with zipfile.ZipFile(buf) as z:
            with z.open("hour.csv") as f:
                df = pd.read_csv(f, parse_dates=["dteday"])
        logger.info("UCI dataset loaded (rows=%d).", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("UCI download failed (%s) -- falling back to OpenML.", exc)
        from sklearn.datasets import fetch_openml

        raw = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True).frame
        # Map OpenML columns onto the canonical UCI names used in the spec.
        rename = {
            "month": "mnth",
            "hour": "hr",
            "count": "cnt",
            "feel_temp": "atemp",
            "humidity": "hum",
            "weather": "weathersit",
            "year": "yr",
        }
        df = raw.rename(columns=rename).copy()
        # OpenML stores the year as 2011/2012; UCI stores it as 0/1. Normalise.
        if df["yr"].max() > 1:
            df["yr"] = df["yr"] - df["yr"].min()
        # Cast categoricals to int.
        for col in ["season", "holiday", "workingday", "weathersit", "weekday"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df


def _process_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning -- ensures all required columns exist and have int types."""
    df = raw.copy()
    for col in ["season", "holiday", "workingday", "weathersit", "weekday", "mnth", "hr", "yr"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Bike-sharing dataframe is missing columns: {missing}")
    return df


def _train_and_predict_reference_model(df: pd.DataFrame):
    """Train RF on January 2011, persist model + reference parquet, return both."""
    if "yr" in df.columns:
        ref = df[(df["yr"] == 0) & (df["mnth"] == 1)].copy()
    else:
        ref = df[df["mnth"] == 1].copy()

    if ref.empty:
        raise RuntimeError("Reference window (January 2011) is empty -- check data source.")

    X_ref = ref[ALL_FEATURES]
    y_ref = ref[TARGET]

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=None,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_ref, y_ref)
    ref[PREDICTION] = model.predict(X_ref)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    ref[ALL_FEATURES + [TARGET, PREDICTION]].to_parquet(REFERENCE_DATA_PATH)
    logger.info(
        "Reference model trained on %d rows. Saved to %s.", len(ref), MODEL_PATH
    )
    return model, ref


def _load_or_train():
    """Return (model, reference_df). Train on first run; afterwards just load."""
    if os.path.isfile(MODEL_PATH) and os.path.isfile(REFERENCE_DATA_PATH):
        logger.info("Loading cached model + reference data from %s.", ARTIFACTS_DIR)
        model = joblib.load(MODEL_PATH)
        reference = pd.read_parquet(REFERENCE_DATA_PATH)
        return model, reference
    raw = _fetch_data()
    df = _process_data(raw)
    return _train_and_predict_reference_model(df)


# ---------------------------------------------------------------------------
# Evidently report -> metric extraction
# ---------------------------------------------------------------------------
def _extract_metrics(report_dict: dict) -> dict:
    """Walk an Evidently 0.4.x as_dict() and pull RMSE/MAE/R2/drift values.

    The result is a flat dict; missing values stay None so that callers can
    decide whether to skip the corresponding gauge update.
    """
    out = {
        "rmse": None,
        "mae": None,
        "r2": None,
        "drift_share": None,
        "dataset_drift": None,
    }
    for entry in report_dict.get("metrics", []):
        name = entry.get("metric", "")
        result = entry.get("result", {}) or {}

        if name == "RegressionQualityMetric":
            current = result.get("current", {}) or {}
            out["rmse"] = current.get("rmse")
            out["mae"] = current.get("mean_abs_error")
            # Some evidently versions name it differently -- try both.
            out["r2"] = current.get("r2_score", current.get("r2"))
            out["mape"] = current.get("mean_abs_perc_error")   # new

        elif name == "DatasetDriftMetric":
            out["drift_share"] = result.get("drift_share")
            out["dataset_drift"] = result.get("dataset_drift")
    return out


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------
state = {"model": None, "reference": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure model + reference are available.
    model, reference = _load_or_train()
    state["model"] = model
    state["reference"] = reference
    logger.info("API ready. Reference rows=%d.", len(reference))
    yield
    # Shutdown: nothing to clean up.
    logger.info("API shutting down.")


app = FastAPI(
    title="Bike Sharing Monitoring API",
    version="1.0.0",
    description="Regression API instrumented with Prometheus + Evidently.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: count + time every request
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Record api_requests_total + api_request_duration_seconds for each request."""
    # Skip the /metrics endpoint to avoid recursive counting noise.
    endpoint = request.url.path
    method = request.method
    if endpoint == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        labels = {"endpoint": endpoint, "method": method, "status_code": str(status_code)}
        REQ_COUNTER.labels(**labels).inc()
        REQ_LATENCY.labels(**labels).observe(elapsed)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/")
def root():
    return {
        "service": "bike-api",
        "endpoints": ["/predict", "/evaluate", "/metrics", "/health", "/trigger-drift"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: BikeSharingInput):
    """Return the model's predicted bike-rental count for a single record."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    row = pd.DataFrame([payload.model_dump()])[ALL_FEATURES]
    yhat = float(state["model"].predict(row)[0])
    return PredictionOutput(prediction=yhat)


@app.post("/evaluate", response_model=EvaluationReportOutput)
def evaluate(payload: EvaluationRequest):
    """Run an Evidently report on a current batch vs. the January reference,
    extract regression quality + drift metrics, and update Prometheus gauges.
    """
    if state["model"] is None or state["reference"] is None:
        raise HTTPException(status_code=503, detail="Model / reference not ready.")

    if not payload.records:
        raise HTTPException(status_code=400, detail="`records` must be non-empty.")

    # Build the current dataframe and add the model's predictions
    records = payload.normalized_records()
    label = payload.normalized_label()

    current = pd.DataFrame([r.model_dump() for r in records])
    current[PREDICTION] = state["model"].predict(current[ALL_FEATURES])

    reference = state["reference"]

    report = Report(metrics=[RegressionPreset(), DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=COLUMN_MAPPING,
    )
    extracted = _extract_metrics(report.as_dict())

    # Update Prometheus gauges (only when Evidently produced a value).
    if extracted["rmse"] is not None:
        MODEL_RMSE.set(float(extracted["rmse"]))
    if extracted["mae"] is not None:
        MODEL_MAE.set(float(extracted["mae"]))
    if extracted["r2"] is not None:
        MODEL_R2.set(float(extracted["r2"]))
    if extracted["drift_share"] is not None:
        DRIFT_SHARE.set(float(extracted["drift_share"]))
    if extracted["dataset_drift"] is not None:
        DRIFT_DETECTED.set(1 if extracted["dataset_drift"] else 0)

    EVAL_COUNTER.inc()
    logger.info(
        "Evaluation '%s' on %d rows: RMSE=%.3f MAE=%.3f R2=%.3f drift_share=%.3f drift=%s",
        label,
        len(current),
        extracted["rmse"] or float("nan"),
        extracted["mae"] or float("nan"),
        extracted["r2"] or float("nan"),
        extracted["drift_share"] or float("nan"),
        extracted["dataset_drift"],
    )

    return EvaluationReportOutput(

        message=f"Evaluation for '{payload.period_label}' completed.",
        rmse=extracted["rmse"],
        mape=extracted.get("mape"),       # add MAPE extraction below
        mae=extracted["mae"],
        r2score=extracted["r2"],
        drift_detected=1 if extracted["dataset_drift"] else 0,
        evaluated_items=len(current),
        # period_label=label,
        # n_records=len(current),
        # rmse=extracted["rmse"],
        # mae=extracted["mae"],
        # r2=extracted["r2"],
        # drift_share=extracted["drift_share"],
        # dataset_drift=extracted["dataset_drift"],
    )


@app.post("/trigger-drift")
def trigger_drift():
    """Force the regression and drift gauges into 'alarming' values so that
    BOTH Prometheus alert rules (HighModelRMSE, DatasetDriftDetected) and
    the Grafana ML alert fire. Used by `make fire-alert`.

    Justification: this exercises the alerting path end-to-end without
    needing real drifted data -- we just push values that exceed the
    configured thresholds and rely on Prometheus' evaluation_interval to
    pick them up within ~30s.
    """
    MODEL_RMSE.set(999.0)
    MODEL_MAE.set(750.0)
    MODEL_R2.set(-1.0)
    DRIFT_SHARE.set(1.0)
    DRIFT_DETECTED.set(1)
    logger.warning("trigger-drift called -- metrics pushed into alert region.")
    return {
        "status": "ok",
        "message": "Synthetic alert state set on metrics. "
                   "Watch Prometheus /alerts and Grafana within ~30s.",
        "metrics": {
            "model_rmse_score": 999.0,
            "model_mae_score": 750.0,
            "model_r2_score": -1.0,
            "evidently_dataset_drift_share": 1.0,
            "evidently_dataset_drift_detected": 1,
        },
    }


@app.get("/metrics")
def metrics():
    """Expose metrics in the Prometheus text format."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
