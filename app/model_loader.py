import json
from pathlib import Path
from typing import Any

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"
MODEL_PATH = DEPLOYMENT_DIR / "customer_churn_model.joblib"
SCHEMA_PATH = DEPLOYMENT_DIR / "feature_schema.json"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"


class DeploymentLoadError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentLoadError(f"Could not read {path.name}") from exc


def load_deployment() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    schema = _read_json(SCHEMA_PATH)
    metadata = _read_json(METADATA_PATH)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise DeploymentLoadError("Could not load the trained model") from exc

    features = metadata.get("features")
    if not isinstance(features, list) or list(schema.keys()) != features:
        raise DeploymentLoadError("Model metadata and feature schema do not match")

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise DeploymentLoadError("Loaded model does not support prediction")

    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError as exc:
        raise DeploymentLoadError("Loaded pipeline has no feature_names_in_") from exc
    if feature_names != features:
        raise DeploymentLoadError("Pipeline feature names do not match deployment metadata")

    return model, schema, metadata
