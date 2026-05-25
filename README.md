# Diabetes Prediction API

Predict whether a person is diabetic based on health metrics using a Random Forest classifier, served via FastAPI, Dockerized, and deployable to Kubernetes.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
```

### Train the model

```bash
make train
# or: python train.py
```

### Run the API locally

```bash
make test
uvicorn main:app --reload
```

### Docker

```bash
make docker-build
make docker-run
```

### Kubernetes

```bash
make k8s-deploy
# or: kubectl apply -f k8s-deploy.yml
```

## API Endpoints

| Method | Path       | Description                    |
|--------|------------|--------------------------------|
| GET    | `/`        | Root message                   |
| GET    | `/health`  | Health check                   |
| POST   | `/predict` | Predict diabetes (see schema)  |

### Sample Predict Request

```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

### Sample Predict Response

```json
{
  "diabetic": true,
  "probability": 0.8134
}
```

## Project Structure

```
├── main.py              # FastAPI application
├── train.py             # Model training with MLflow tracking
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Local dev with hot-reload
├── k8s-deploy.yml       # Kubernetes manifests (Deployment, Service, HPA, ConfigMap)
├── Makefile             # Standardized commands
├── pyproject.toml       # Project config (ruff, pytest)
├── requirements/
│   ├── base.txt         # Runtime dependencies (pinned)
│   ├── train.txt        # Training dependencies
│   └── dev.txt          # Development dependencies
└── tests/
    └── test_api.py      # API tests
```
