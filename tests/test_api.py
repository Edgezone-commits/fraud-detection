"""API tests: bad input is rejected, the model-missing path is clear, and a valid request works."""

import json
import math

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.schemas import TransactionIn
from helpers import FEATURES, requires_model

ZERO_ROW = {name: 0.0 for name in FEATURES}


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def post_raw(client, text):
    """Send a request body exactly as written, for inputs JSON clients cannot build."""
    return client.post("/predict", content=text, headers={"content-type": "application/json"})


def test_schema_fields_match_training_features_in_order():
    assert list(TransactionIn.model_fields) == FEATURES


def test_valid_zero_transaction_passes_validation_even_without_a_model(client, monkeypatch):
    # With no model the answer is 503, not 422: the input itself was accepted.
    monkeypatch.setattr(api, "MODEL_PATH", api.MODEL_PATH.with_name("no_such_model.joblib"))
    with TestClient(api.app) as c:
        assert c.post("/predict", json=ZERO_ROW).status_code == 503


def test_missing_field_is_rejected(client):
    row = {k: v for k, v in ZERO_ROW.items() if k != "V28"}
    response = client.post("/predict", json=row)
    assert response.status_code == 422
    assert any(d["loc"][-1] == "V28" for d in response.json()["detail"])


def test_unknown_field_is_rejected(client):
    row = dict(ZERO_ROW, v1=1.0)  # lower-case typo of V1
    response = client.post("/predict", json=row)
    assert response.status_code == 422


def test_negative_amount_is_rejected(client):
    response = client.post("/predict", json=dict(ZERO_ROW, Amount=-5))
    assert response.status_code == 422


def test_wrong_type_is_rejected(client):
    response = client.post("/predict", json=dict(ZERO_ROW, V3="abc"))
    assert response.status_code == 422


def test_out_of_range_pca_value_is_rejected(client):
    response = client.post("/predict", json=dict(ZERO_ROW, V1=5000))
    assert response.status_code == 422


def test_nan_is_a_422_not_a_500(client):
    body = json.dumps(ZERO_ROW)[:-1] + ', "V1": NaN}'
    response = post_raw(client, body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == "nan"


def test_infinity_is_a_422(client):
    body = json.dumps(ZERO_ROW)[:-1] + ', "V2": Infinity}'
    assert post_raw(client, body).status_code == 422


def test_health_reports_the_real_state(client):
    body = client.get("/health").json()
    if api.MODEL_PATH.exists():
        assert body["status"] == "ok" and body["model_loaded"] is True
        assert body["calibration"] in {"calibrated", "raw"}
        assert 0 < body["threshold"] < 1
    else:
        assert body == {"status": "degraded", "model_loaded": False, "threshold": None, "calibration": None}


@requires_model
def test_predict_returns_a_valid_decision(client):
    response = client.post("/predict", json=ZERO_ROW)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"fraud", "legitimate"}
    expected = "fraud" if body["fraud_probability"] >= body["threshold"] else "legitimate"
    assert body["decision"] == expected


@requires_model
def test_explain_returns_five_reasons(client):
    response = client.post("/explain", json=ZERO_ROW)
    assert response.status_code == 200
    reasons = response.json()["top_reasons"]
    assert len(reasons) == 5
    assert all(math.isfinite(r["contribution"]) for r in reasons)
    assert all(r["direction"] in {"toward_fraud", "toward_legitimate"} for r in reasons)


def test_model_missing_gives_a_clear_503(monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", api.MODEL_PATH.with_name("no_such_model.joblib"))
    with TestClient(api.app) as c:
        response = c.post("/predict", json=ZERO_ROW)
        assert response.status_code == 503
        assert "scripts/train.py" in response.json()["detail"]
