import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
PREDICTION_LOG = os.getenv("PREDICTION_LOG", "predictions.jsonl")

model = None
metadata = None
threshold = 0.5
feature_cols = None

# Prometheus metrics
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["model"])
PREDICTION_LATENCY = Histogram("prediction_seconds", "Prediction latency seconds")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")


def load_model_package(path: str):
    global model, metadata, threshold, feature_cols
    try:
        pkg = joblib.load(path)
        if isinstance(pkg, dict) and "model" in pkg and "metadata" in pkg:
            model = pkg["model"]
            metadata = pkg["metadata"]
            threshold = metadata.get("classification_threshold", 0.5)
            feature_cols = metadata.get("features", None)
            logger.info("Model loaded from %s (v1 format)", path)
        else:
            model = pkg
            metadata = {"model_name": "legacy", "classification_threshold": 0.5}
            threshold = 0.5
            feature_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "BMI", "Age",
            ]
            logger.info("Model loaded from %s (legacy format)", path)
        return True
    except FileNotFoundError:
        logger.error("Model file not found at %s", path)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_package(MODEL_PATH)
    yield


app = FastAPI(
    title="Diabetes Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request ID middleware ----
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---- Schemas ----
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., gt=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(
        ..., gt=0, description="Diastolic blood pressure (mm Hg)"
    )
    SkinThickness: float = Field(
        ..., ge=0, description="Triceps skin fold thickness (mm)"
    )
    Insulin: float = Field(
        ..., ge=0, description="2-Hour serum insulin (mu U/ml)"
    )
    BMI: float = Field(..., gt=0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0, description="Diabetes pedigree function"
    )
    Age: int = Field(..., ge=0, description="Age in years")

    @field_validator("Pregnancies", "Age")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Must be non-negative")
        return v

    @field_validator("Glucose", "BloodPressure", "BMI")
    @classmethod
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class BatchInput(BaseModel):
    inputs: list[DiabetesInput]


class PredictionResponse(BaseModel):
    diabetic: bool
    probability: float
    request_id: str = ""


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ExplainResponse(BaseModel):
    diabetic: bool
    probability: float
    top_features: list[dict]
    shap_values: list[float] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str = ""
    threshold: float = 0.5


# ---- Feature engineering (must match train.py) ----
def engineer_features_dict(data: dict) -> dict:
    features = dict(data)
    bmi = features["BMI"]
    age = features["Age"]
    glucose = features["Glucose"]
    insulin = features["Insulin"]

    # BMI categories
    if bmi < 18.5:
        features["BMI_Category"] = 0
    elif bmi < 25:
        features["BMI_Category"] = 1
    elif bmi < 30:
        features["BMI_Category"] = 2
    else:
        features["BMI_Category"] = 3

    # Age groups
    if age < 30:
        features["Age_Group"] = 0
    elif age < 45:
        features["Age_Group"] = 1
    elif age < 60:
        features["Age_Group"] = 2
    else:
        features["Age_Group"] = 3

    # Glucose risk bins
    if glucose < 100:
        features["Glucose_Risk"] = 0
    elif glucose < 126:
        features["Glucose_Risk"] = 1
    elif glucose < 200:
        features["Glucose_Risk"] = 2
    else:
        features["Glucose_Risk"] = 3

    # Interactions
    features["Insulin_Glucose"] = insulin * glucose / 1000.0
    features["BMI_Age"] = bmi * age / 100.0
    features["Insulin_Resistance"] = glucose * insulin / 405.0

    return features


def build_feature_vector(data: DiabetesInput) -> np.ndarray:
    raw = data.model_dump()
    engineered = engineer_features_dict(raw)
    if feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not configured")
    row = [engineered.get(col, 0.0) for col in feature_cols]
    return pd.DataFrame([row], columns=feature_cols)


def log_prediction(input_data: dict, result: dict, latency_ms: float):
    try:
        with open(PREDICTION_LOG, "a") as f:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": input_data,
                "result": result,
                "latency_ms": round(latency_ms, 2),
            }
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to log prediction: %s", e)


# ---- Endpoints ----
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=metadata.get("model_name", "") if metadata else "",
        threshold=threshold,
    )


@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API v2.0", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: DiabetesInput, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = request.headers.get("X-Request-ID", "")
    start = time.time()

    try:
        features = build_feature_vector(data)
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= threshold)
        latency = (time.time() - start) * 1000

        PREDICTION_COUNT.labels(
            model=(metadata or {}).get("model_name", "unknown")
        ).inc()
        PREDICTION_LATENCY.observe(time.time() - start)

        result = {
            "diabetic": bool(prediction),
            "probability": round(float(proba[1]), 4),
            "request_id": request_id,
        }

        log_prediction(data.model_dump(), result, latency)
        logger.info(
            "Prediction: %s (prob: %.4f, th: %.2f) | req: %s | %.1fms",
            prediction, proba[1], threshold, request_id, latency,
        )
        return PredictionResponse(**result)  # type: ignore[arg-type]

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(data: BatchInput, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = request.headers.get("X-Request-ID", "")
    results = []

    for inp in data.inputs:
        features = build_feature_vector(inp)
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= threshold)
        results.append(PredictionResponse(
            diabetic=bool(prediction),
            probability=round(float(proba[1]), 4),
            request_id=request_id,
        ))

    PREDICTION_COUNT.labels(
        model=(metadata or {}).get("model_name", "unknown")
    ).inc(len(results))
    return BatchPredictionResponse(predictions=results)


@app.post("/explain", response_model=ExplainResponse)
async def explain(data: DiabetesInput, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_feature_vector(data)
    proba = model.predict_proba(features)[0]
    prediction = int(proba[1] >= threshold)

    top_features = []
    shap_values_list = None

    try:
        import shap as shap_lib

        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("model")
        elif hasattr(model, "estimators_"):
            clf = model.estimators_[0]
        else:
            clf = model

        if clf is not None:
            if hasattr(clf, "feature_importances_"):
                explainer = shap_lib.TreeExplainer(clf)
            elif hasattr(clf, "coef_"):
                explainer = shap_lib.LinearExplainer(clf, features)
            else:
                explainer = None

            if explainer is not None:
                raw_shap = explainer.shap_values(features)

                if isinstance(raw_shap, list) and len(raw_shap) > 1:
                    vals = raw_shap[1].flatten()
                else:
                    vals = np.array(raw_shap).flatten()

                shap_values_list = [round(float(v), 4) for v in vals]

                if feature_cols:
                    paired = list(zip(feature_cols, vals))
                    paired.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_features = [
                        {"feature": name, "importance": round(float(v), 4)}
                        for name, v in paired[:5]
                    ]
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        # Fallback: global feature importances or coefficients
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("model")
            if hasattr(est, "feature_importances_"):
                fi = est.feature_importances_
            elif hasattr(est, "coef_"):
                fi = np.abs(est.coef_[0])
            else:
                fi = None
            if fi is not None and feature_cols:
                paired = list(zip(feature_cols, fi))
                paired.sort(key=lambda x: x[1], reverse=True)
                top_features = [
                    {"feature": name, "importance": round(float(v), 4)}
                    for name, v in paired[:5]
                ]

    return ExplainResponse(
        diabetic=bool(prediction),
        probability=round(float(proba[1]), 4),
        top_features=top_features,
        shap_values=shap_values_list,
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model_info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": metadata.get("model_name"),
        "threshold": metadata.get("classification_threshold"),
        "features": metadata.get("features"),
        "data_hash": metadata.get("data_hash"),
        "test_metrics": metadata.get("test_metrics"),
        "all_results": metadata.get("all_results"),
    }
