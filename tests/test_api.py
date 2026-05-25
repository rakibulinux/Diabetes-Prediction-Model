import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def load_test_model():
    dummy = RandomForestClassifier(n_estimators=2, random_state=42)
    X_dummy = np.random.rand(20, 5)
    y_dummy = np.random.randint(0, 2, 20)
    dummy.fit(X_dummy, y_dummy)
    import main as m
    m.model = dummy
    yield


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] is not None


def test_predict_valid():
    payload = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "BMI": 28.5,
        "Age": 45,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "diabetic" in data
    assert "probability" in data
    assert isinstance(data["diabetic"], bool)
    assert 0.0 <= data["probability"] <= 1.0


@pytest.mark.parametrize("field,value", [
    ("Pregnancies", -1),
    ("Age", -5),
    ("Glucose", 0),
    ("BloodPressure", -10),
    ("BMI", 0),
])
def test_predict_invalid_input(field, value):
    payload = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "BMI": 28.5,
        "Age": 45,
    }
    payload[field] = value
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_model_not_loaded():
    import main as m
    saved = m.model
    m.model = None
    resp = client.post("/predict", json={
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70, "BMI": 28.5, "Age": 45,
    })
    assert resp.status_code == 503
    m.model = saved
