import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found at %s", MODEL_PATH)
        model = None
    yield


app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

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


class PredictionResponse(BaseModel):
    diabetic: bool
    probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
    )


@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = np.array(
        [[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]]
    )
    proba = model.predict_proba(features)[0]
    prediction = int(proba[1] >= 0.5)
    logger.info(
        "Prediction: %s (probability: %.4f) for input: %s",
        prediction, proba[1], data.model_dump(),
    )
    return PredictionResponse(diabetic=bool(prediction), probability=round(proba[1], 4))
