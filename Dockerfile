# syntax=docker/dockerfile:1
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/train.txt

COPY . .
RUN python train.py


FROM python:3.10-slim AS runtime

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/base.txt

COPY main.py .

COPY --from=builder /app/diabetes_model.pkl .

EXPOSE 8000

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
