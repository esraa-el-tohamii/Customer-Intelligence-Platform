import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert body["feature_schema"]["Age"] == "number"
    assert body["feature_schema"]["Gender"] == "string"
    assert body["prediction_classes"] == {"0": "Stayed", "1": "Churned"}
    assert body["model_name"] == "Gradient Boosting(Tuned)"
    assert body["evaluation_metrics"]["Accuracy"] == pytest.approx(0.9363531996595476)


def test_single_prediction(client):
    response = client.post("/predict", json=customer_record())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("Stayed", "Churned")
    assert 0 <= body["churn_probability"] <= 1
    assert 0 <= body["retention_probability"] <= 1


def test_prediction_matches_exported_pipeline_probabilities(client):
    record = customer_record()
    model = joblib.load(PROJECT_ROOT / "deployment" / "customer_churn_model.joblib")
    frame = pd.DataFrame([record], columns=list(record))
    classes = list(model.classes_)
    probabilities = model.predict_proba(frame)[0]

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == int(model.predict(frame)[0])
    assert result["churn_probability"] == pytest.approx(probabilities[classes.index(1)], abs=1e-6)
    assert result["retention_probability"] == pytest.approx(probabilities[classes.index(0)], abs=1e-6)


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


def test_feature_importance_uses_gradient_boosting_values_in_feature_order(client):
    response = client.get("/feature-importance")

    assert response.status_code == 200
    body = response.json()
    expected_names = json.loads(
        (PROJECT_ROOT / "deployment" / "shap_feature_names.json").read_text(encoding="utf-8")
    )
    expected_importance = pd.read_csv(
        PROJECT_ROOT / "deployment" / "gradient_boosting_shap_importance.csv"
    ).set_index("Feature")
    assert body["model_name"] == "Gradient Boosting(Tuned)"
    assert body["method"] == "mean_absolute_shap_value"
    assert body["feature_names"] == expected_names
    assert [item["feature"] for item in body["feature_importance"]] == expected_names
    assert all(set(item) == {"feature", "mean_abs_shap_value"} for item in body["feature_importance"])
    for item in body["feature_importance"]:
        assert item["mean_abs_shap_value"] == pytest.approx(
            expected_importance.loc[item["feature"], "Gradient Boosting"]
        )


def test_individual_explanation_is_for_submitted_customer(client):
    response = client.post("/explain", json=customer_record())

    assert response.status_code == 200
    body = response.json()
    expected_names = json.loads(
        (PROJECT_ROOT / "deployment" / "shap_feature_names.json").read_text(encoding="utf-8")
    )
    assert body["model_name"] == "Gradient Boosting(Tuned)"
    assert [item["feature"] for item in body["features"]] == expected_names
    assert [item["value"] for item in body["features"]] == [
        customer_record()[name] for name in expected_names
    ]
    assert all(
        item["impact"] in {"increases_churn", "decreases_churn", "no_effect"}
        for item in body["features"]
    )
    assert sum(item["shap_value"] for item in body["features"]) + body[
        "baseline_churn_probability"
    ] == pytest.approx(body["churn_probability"], abs=1e-4)


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


def test_non_finite_numeric_input(client):
    record = customer_record()
    record["Age"] = float("inf")
    response = client.post(
        "/predict",
        content=json.dumps(record),
        headers={"content-type": "application/json"},
    )
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


def test_batch_rejects_non_object_record(client):
    response = client.post("/predict/batch", json={"records": [customer_record(), []]})
    assert response.status_code == 422
