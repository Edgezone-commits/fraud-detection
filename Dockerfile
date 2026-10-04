# Serving image for the fraud detection API.
# Python 3.12 matches the local venv. Build it AFTER running scripts/train.py,
# because the image copies the trained model from models/.
#
#   docker build -t fraud-api .
#   docker run --rm -p 8000:8000 fraud-api

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first, so Docker can reuse this layer when only code changes.
# requirements.txt has only runtime packages: no torch, no notebooks, no test tools.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code and the trained model (models/fraud_model.joblib).
COPY src ./src
COPY app ./app
COPY models ./models

# Do not run the API as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Container health check. The API returns 200 on /health even when the model is missing,
# so the check reads the "model_loaded" field, not just the HTTP status.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json,urllib.request,sys; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); sys.exit(0 if d['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
