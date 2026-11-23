import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def customer_record():
    return {
        "Age": 42,
        "Gender": "Female",
        "Tenure": 24,
        "Usage Frequency": 12,
        "Support Calls": 3,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Annual",
        "Total Spend": 1800.50,
        "Last Interaction": 10,
        "Spend_Per_Tenure": 75.020833,
        "Support_Calls_Per_Usage": 0.25,
    }


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["api_name"]
    assert body["status"] == "running"


def test_health_loads_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["number_of_features"] == 12
    assert len(body["expected_feature_names"]) == 12
    assert body["prediction_classes"] == {"0": "Stayed", "1": "Churned"}


def test_single_prediction(client):
    response = client.post("/predict", json=customer_record())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("Stayed", "Churned")
    assert 0 <= body["churn_probability"] <= 1
    assert 0 <= body["retention_probability"] <= 1


def test_batch_prediction(client):
    response = client.post(
        "/predict/batch",
        json={"records": [customer_record(), customer_record()]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    for result in body["predictions"]:
        assert result["prediction"] in (0, 1)
        assert 0 <= result["churn_probability"] <= 1
        assert 0 <= result["retention_probability"] <= 1


def test_missing_required_feature(client):
    record = customer_record()
    del record["Age"]
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Age" in str(response.json())


def test_invalid_data_type(client):
    record = customer_record()
    record["Age"] = "forty"
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Age" in str(response.json())


def test_unexpected_feature(client):
    record = customer_record()
    record["Unexpected"] = "value"
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Unexpected" in str(response.json())


def test_invalid_categorical_value(client):
    record = customer_record()
    record["Gender"] = "Other"
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Gender" in str(response.json())


def test_empty_batch(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422
