import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"
MODEL_PATH = DEPLOYMENT_DIR / "customer_churn_model.joblib"
SCHEMA_PATH = DEPLOYMENT_DIR / "feature_schema.json"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
RESULTS_PATH = DEPLOYMENT_DIR / "final_results.json"
CATEGORY_MAPPINGS_PATH = DEPLOYMENT_DIR / "category_mappings.json"
SHAP_FEATURE_NAMES_PATH = DEPLOYMENT_DIR / "shap_feature_names.json"
SHAP_IMPORTANCE_PATH = DEPLOYMENT_DIR / "gradient_boosting_shap_importance.csv"
SHAP_BACKGROUND_PATH = DEPLOYMENT_DIR / "shap_background_encoded.pkl"


class DeploymentLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeploymentArtifacts:
    model: Any
    schema: dict[str, str]
    metadata: dict[str, Any]
    category_mappings: dict[str, list[str]]
    evaluation_metrics: dict[str, float]
    shap_feature_names: list[str]
    shap_feature_importance: list[dict[str, Any]]
    shap_background_path: Path


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentLoadError(f"Could not read deployment artifact {path.name}") from exc


def _load_schema(document: Any, metadata: dict[str, Any]) -> dict[str, str]:
    if not isinstance(document, dict) or not isinstance(document.get("features"), dict):
        raise DeploymentLoadError("Feature schema has an invalid structure")

    definitions = document["features"]
    feature_names = metadata.get("features")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise DeploymentLoadError("Model metadata has an invalid feature list")
    if len(set(feature_names)) != len(feature_names) or list(definitions) != feature_names:
        raise DeploymentLoadError("Model metadata and feature schema do not match")
    if document.get("number_of_features") != len(feature_names):
        raise DeploymentLoadError("Feature schema feature count does not match")
    if metadata.get("number_of_features") != len(feature_names):
        raise DeploymentLoadError("Model metadata feature count does not match")

    schema: dict[str, str] = {}
    for feature_name in feature_names:
        definition = definitions.get(feature_name)
        if (
            not isinstance(definition, dict)
            or definition.get("type") not in {"number", "string"}
            or definition.get("required") is not True
        ):
            raise DeploymentLoadError(f"Feature schema is invalid for {feature_name}")
        schema[feature_name] = definition["type"]

    numerical_features = metadata.get("numerical_features")
    categorical_features = metadata.get("categorical_features")
    if (
        not isinstance(numerical_features, list)
        or not isinstance(categorical_features, list)
        or not all(isinstance(name, str) for name in numerical_features + categorical_features)
        or set(numerical_features) & set(categorical_features)
        or len(set(numerical_features)) != len(numerical_features)
        or len(set(categorical_features)) != len(categorical_features)
        or set(numerical_features) | set(categorical_features) != set(feature_names)
    ):
        raise DeploymentLoadError("Model metadata feature types do not match the schema")
    if any(schema[name] != "number" for name in numerical_features) or any(
        schema[name] != "string" for name in categorical_features
    ):
        raise DeploymentLoadError("Model metadata feature types do not match the schema")
    return schema


def _pipeline_categories(model: Any) -> dict[str, list[str]]:
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    transformers = getattr(preprocessor, "transformers_", [])
    for name, transformer, columns in transformers:
        if name != "cat":
            continue
        encoder = getattr(transformer, "named_steps", {}).get("encoder")
        categories = getattr(encoder, "categories_", None)
        if categories is None:
            break
        return {
            column: [str(value) for value in values]
            for column, values in zip(columns, categories)
        }
    raise DeploymentLoadError("Model pipeline has no fitted categorical encoder")


def _load_category_mappings(document: Any, metadata: dict[str, Any], model: Any) -> dict[str, list[str]]:
    if not isinstance(document, dict):
        raise DeploymentLoadError("Category mappings have an invalid structure")

    categorical_features = metadata["categorical_features"]
    if set(document) != set(categorical_features):
        raise DeploymentLoadError("Category mappings do not match model metadata")

    mappings: dict[str, list[str]] = {}
    pipeline_categories = _pipeline_categories(model)
    for feature_name in categorical_features:
        categories = document.get(feature_name)
        if (
            not isinstance(categories, list)
            or not categories
            or not all(isinstance(category, str) for category in categories)
            or len(set(categories)) != len(categories)
            or set(categories) != set(pipeline_categories.get(feature_name, []))
        ):
            raise DeploymentLoadError(f"Category mappings are invalid for {feature_name}")
        mappings[feature_name] = categories
    return mappings


def _load_evaluation_metrics(document: Any, metadata: dict[str, Any]) -> dict[str, float]:
    metadata_metrics = metadata.get("evaluation_metrics")
    if not isinstance(document, dict) or not isinstance(metadata_metrics, dict):
        raise DeploymentLoadError("Evaluation metrics have an invalid structure")
    if set(document) != set(metadata_metrics):
        raise DeploymentLoadError("Evaluation metrics do not match model metadata")

    metrics: dict[str, float] = {}
    for name, value in document.items():
        metadata_value = metadata_metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or isinstance(metadata_value, bool)
            or not isinstance(metadata_value, (int, float))
            or not math.isclose(value, metadata_value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise DeploymentLoadError(f"Evaluation metric {name} is invalid or inconsistent")
        metrics[name] = float(value)
    return metrics


def _load_shap_importance(feature_names: list[str]) -> list[dict[str, Any]]:
    try:
        with SHAP_IMPORTANCE_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"Feature", "Gradient Boosting"}.issubset(reader.fieldnames):
                raise DeploymentLoadError("Gradient Boosting SHAP importance has an invalid structure")
            rows = list(reader)
    except OSError as exc:
        raise DeploymentLoadError("Could not read Gradient Boosting SHAP importance") from exc

    importance_by_feature: dict[str, float] = {}
    for row in rows:
        feature_name = row.get("Feature")
        try:
            value = float(row["Gradient Boosting"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentLoadError("Gradient Boosting SHAP importance contains invalid values") from exc
        if feature_name in importance_by_feature or not math.isfinite(value) or value < 0:
            raise DeploymentLoadError("Gradient Boosting SHAP importance contains invalid values")
        importance_by_feature[feature_name] = value

    if set(importance_by_feature) != set(feature_names):
        raise DeploymentLoadError("Gradient Boosting SHAP importance features do not match the model")
    return [
        {"feature": feature_name, "mean_abs_shap_value": importance_by_feature[feature_name]}
        for feature_name in feature_names
    ]


def load_deployment() -> DeploymentArtifacts:
    metadata = _read_json(METADATA_PATH)
    if not isinstance(metadata, dict):
        raise DeploymentLoadError("Model metadata has an invalid structure")
    schema = _load_schema(_read_json(SCHEMA_PATH), metadata)

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise DeploymentLoadError("Could not load the trained model") from exc

    feature_names = metadata["features"]
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise DeploymentLoadError("Loaded model does not support prediction")
    if list(getattr(model, "feature_names_in_", [])) != feature_names:
        raise DeploymentLoadError("Pipeline feature names do not match deployment metadata")
    if not isinstance(getattr(model, "named_steps", {}).get("classifier"), HistGradientBoostingClassifier):
        raise DeploymentLoadError("Loaded model is not the exported Gradient Boosting pipeline")

    positive_class = metadata.get("positive_class")
    classes = list(getattr(model, "classes_", []))
    if positive_class != 1 or 0 not in classes or positive_class not in classes:
        raise DeploymentLoadError("Model prediction classes do not match deployment metadata")
    prediction_classes = metadata.get("prediction_classes")
    if not isinstance(prediction_classes, dict) or set(prediction_classes) != {"0", "1"}:
        raise DeploymentLoadError("Prediction class metadata is invalid")

    category_mappings = _load_category_mappings(_read_json(CATEGORY_MAPPINGS_PATH), metadata, model)
    evaluation_metrics = _load_evaluation_metrics(_read_json(RESULTS_PATH), metadata)
    shap_feature_names = _read_json(SHAP_FEATURE_NAMES_PATH)
    if shap_feature_names != feature_names:
        raise DeploymentLoadError("Gradient Boosting SHAP feature names do not match the model")
    shap_feature_importance = _load_shap_importance(shap_feature_names)

    return DeploymentArtifacts(
        model=model,
        schema=schema,
        metadata=metadata,
        category_mappings=category_mappings,
        evaluation_metrics=evaluation_metrics,
        shap_feature_names=shap_feature_names,
        shap_feature_importance=shap_feature_importance,
        shap_background_path=SHAP_BACKGROUND_PATH,
    )
