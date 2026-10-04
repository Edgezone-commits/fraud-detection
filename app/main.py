"""Fraud detection API.

Run from the project root (after `pip install -e .` and `python scripts/train.py`):

    uvicorn app.main:app --reload

Interactive docs are generated automatically at http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
import logging

import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import ExplainOut, HealthOut, PredictOut, Reason, TransactionIn
from fraud_detection.config import MODELS_DIR
from fraud_detection.explain import make_explainer, top_reasons
from fraud_detection.train import predict_fraud_proba

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
EXPLANATION_NOTE = (
    "SHAP contributions are in log-odds of the raw (uncalibrated) XGBoost model. "
    "fraud_probability uses the calibrated score, so the two are related but not identical."
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the SHAP explainer ONCE at startup, not on every request."""
    app.state.bundle = None
    app.state.explainer = None
    if not MODEL_PATH.exists():
        logger.warning("No model at %s. Run scripts/train.py first. API starts degraded.", MODEL_PATH)
    else:
        bundle = joblib.load(MODEL_PATH)
        # The schema and the saved model must agree on the 30 features and their names.
        if set(TransactionIn.model_fields) != set(bundle["feature_names"]):
            raise RuntimeError("TransactionIn fields do not match the saved model's feature_names.")
        app.state.bundle = bundle
        app.state.explainer = make_explainer(bundle["model"])
        logger.info("Model loaded from %s (threshold=%.5f)", MODEL_PATH, bundle["threshold"])
    yield


app = FastAPI(
    title="Credit card fraud detection",
    version="0.1.0",
    description="Fraud probability, a threshold-based decision and SHAP reasons per transaction.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clear 422 for bad input.

    FastAPI's default handler echoes the rejected value back as JSON, and NaN or
    Infinity cannot be written as JSON, which would turn bad input into a 500.
    Non-finite numbers are reported as text instead.
    """
    errors = []
    for err in exc.errors():
        err = dict(err)
        value = err.get("input")
        if isinstance(value, float) and not math.isfinite(value):
            err["input"] = str(value)
        errors.append(err)
    return JSONResponse(status_code=422, content={"detail": errors})


def _loaded_bundle(request: Request):
    """Return the loaded model bundle, or a clear 503 if the model is missing."""
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python scripts/train.py` to create models/fraud_model.joblib, then restart the API.",
        )
    return bundle


def _as_frame(tx: TransactionIn, feature_names):
    """One-row DataFrame with columns in the exact order the model was trained on."""
    return pd.DataFrame([tx.model_dump()])[feature_names]


def _score(bundle, frame):
    """Return (probability, decision, threshold) for one prepared row."""
    proba = float(predict_fraud_proba(bundle["model"], bundle["calibrator"], frame)[0])
    threshold = float(bundle["threshold"])
    decision = "fraud" if proba >= threshold else "legitimate"
    return proba, decision, threshold


@app.get("/health", response_model=HealthOut)
def health(request: Request):
    """Is the service up, and is the model loaded?"""
    bundle = request.app.state.bundle
    if bundle is None:
        return HealthOut(status="degraded", model_loaded=False, threshold=None, calibration=None)
    calibration = "calibrated" if bundle["calibrator"] is not None else "raw"
    return HealthOut(status="ok", model_loaded=True, threshold=float(bundle["threshold"]), calibration=calibration)


@app.post("/predict", response_model=PredictOut)
def predict(tx: TransactionIn, request: Request):
    """Fraud probability and decision for one transaction."""
    bundle = _loaded_bundle(request)
    frame = _as_frame(tx, bundle["feature_names"])
    proba, decision, threshold = _score(bundle, frame)
    return PredictOut(fraud_probability=proba, decision=decision, threshold=threshold)


@app.post("/explain", response_model=ExplainOut)
def explain(tx: TransactionIn, request: Request):
    """Decision plus the 5 features that moved this transaction's score the most."""
    bundle = _loaded_bundle(request)
    frame = _as_frame(tx, bundle["feature_names"])
    proba, decision, threshold = _score(bundle, frame)
    reasons = top_reasons(request.app.state.explainer, bundle["feature_names"], frame, k=5)
    return ExplainOut(
        fraud_probability=proba,
        decision=decision,
        threshold=threshold,
        top_reasons=[
            Reason(
                feature=r["feature"],
                value=r["value"],
                contribution=r["contribution"],
                direction="toward_fraud" if r["contribution"] > 0 else "toward_legitimate",
            )
            for r in reasons
        ],
        explanation_note=EXPLANATION_NOTE,
    )
