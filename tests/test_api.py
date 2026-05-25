import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def load_test_model():
    dummy = RandomForestClassifier(n_estimators=2, random_state=42)
    X_dummy = np.random.rand(20, 14)
    y_dummy = np.random.randint(0, 2, 20)
    dummy.fit(X_dummy, y_dummy)
    import main as m
    m.model = dummy
    m.metadata = {"model_name": "test", "classification_threshold": 0.5}
    m.feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "BMI_Category", "Age_Group", "Glucose_Risk",
        "Insulin_Glucose", "BMI_Age", "Insulin_Resistance",
    ]
    m.threshold = 0.5
    yield


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] is not None


def test_predict_valid():
    payload = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "diabetic" in data
    assert "probability" in data
    assert "request_id" in data
    assert isinstance(data["diabetic"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert "X-Request-ID" in resp.headers


def test_batch_predict():
    def make_input(p, g, bp, st, ins, bmi, dpf, age):
        return {
        "Pregnancies": p, "Glucose": g, "BloodPressure": bp,
        "SkinThickness": st, "Insulin": ins, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }
    payload = {
        "inputs": [
            make_input(1, 100, 65, 20, 50, 25.0, 0.3, 30),
            make_input(5, 180, 85, 35, 200, 35.0, 0.8, 55),
        ]
    }
    resp = client.post("/batch_predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    for p in data["predictions"]:
        assert "diabetic" in p
        assert "probability" in p


def test_explain():
    payload = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "diabetic" in data
    assert "probability" in data
    assert "top_features" in data
    assert len(data["top_features"]) > 0


def test_model_info():
    resp = client.get("/model_info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert "threshold" in data


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200


@pytest.mark.parametrize("field,value", [
    ("Pregnancies", -1),
    ("Age", -5),
    ("Glucose", 0),
    ("BloodPressure", -10),
    ("BMI", 0),
    ("Glucose", -1),
    ("BloodPressure", 0),
])
def test_predict_invalid_input(field, value):
    payload = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }
    payload[field] = value
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_model_not_loaded():
    import main as m
    saved = m.model
    m.model = None
    payload = {
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5, "Age": 45,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 503
    m.model = saved
