import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app.model_loader import DeploymentArtifacts, DeploymentLoadError, load_deployment
from app.schemas import create_customer_model
from app.services.prediction_service import PredictionService
from app.services.shap_service import ShapArtifactError, ShapExplanationService


logger = logging.getLogger(__name__)


class AppState:
    model: Any = None
    schema: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    evaluation_metrics: dict[str, float] = {}
    shap_feature_names: list[str] = []
    shap_feature_importance: list[dict[str, Any]] = []
    service: PredictionService | None = None
    explanation_service: ShapExplanationService | None = None
    request_model: type[BaseModel] | None = None
    load_error: str | None = None


state = AppState()


def _activate_deployment(deployment: DeploymentArtifacts) -> None:
    state.model = deployment.model
    state.schema = deployment.schema
    state.metadata = deployment.metadata
    state.evaluation_metrics = deployment.evaluation_metrics
    state.shap_feature_names = deployment.shap_feature_names
    state.shap_feature_importance = deployment.shap_feature_importance
    state.service = PredictionService(
        deployment.model,
        deployment.schema,
        deployment.metadata,
        deployment.category_mappings,
    )
    state.explanation_service = ShapExplanationService(
        deployment.model,
        deployment.shap_feature_names,
        deployment.category_mappings,
        deployment.shap_background_path,
        deployment.metadata["positive_class"],
        deployment.metadata["model_name"],
    )
    state.request_model = create_customer_model(deployment.schema, deployment.category_mappings)
    state.load_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _activate_deployment(load_deployment())
    except DeploymentLoadError as exc:
        logger.exception("Could not load deployment artifacts")
        state.load_error = str(exc)
        state.service = None
        state.explanation_service = None
        state.request_model = None
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn using an exported scikit-learn pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"] if part != "body")
        errors.append(
            {
                "field": location or "body",
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


def _validate_customer(payload: dict[str, Any], context: str = "Invalid input") -> dict[str, Any]:
    if state.request_model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    try:
        validated = state.request_model.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        raise HTTPException(
            status_code=422,
            detail={"message": context, "errors": errors},
        ) from exc
    return validated.model_dump()


@app.get("/")
def root():
    return {
        "api_name": "Customer Churn Prediction API",
        "status": "running",
        "description": "Predict whether a customer is likely to churn using the deployed ML pipeline.",
    }


@app.get("/health")
def health():
    if state.service is not None:
        return {"status": "healthy", "model_loaded": True}
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "model_loaded": False,
            "detail": state.load_error or "Model is not loaded",
        },
    )


@app.get("/model-info")
def model_info():
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable")
    metadata = state.metadata
    return {
        "model_name": metadata.get("model_name"),
        "project_name": metadata.get("project_name"),
        "number_of_features": metadata.get("number_of_features"),
        "expected_feature_names": metadata.get("features"),
        "feature_schema": state.schema,
        "target_column": metadata.get("target_column"),
        "prediction_classes": metadata.get("prediction_classes"),
        "positive_class": metadata.get("positive_class"),
        "threshold": metadata.get("threshold"),
        "evaluation_metrics": state.evaluation_metrics,
    }


@app.get("/feature-importance")
def feature_importance():
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return {
        "model_name": state.metadata["model_name"],
        "method": "mean_absolute_shap_value",
        "feature_names": state.shap_feature_names,
        "feature_importance": state.shap_feature_importance,
    }


@app.post("/predict")
def predict(payload: dict[str, Any]):
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    validated = _validate_customer(payload)
    return state.service.predict_one(validated)


@app.post("/predict/batch")
def predict_batch(payload: dict[str, Any]):
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    records = payload.get("records")
    if "records" not in payload:
        raise HTTPException(status_code=422, detail="Request body must contain a 'records' field")
    if not isinstance(records, list):
        raise HTTPException(status_code=422, detail="'records' must be a JSON array")
    if not records:
        raise HTTPException(status_code=422, detail="Batch request must contain at least one customer record")

    validated_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise HTTPException(status_code=422, detail=f"Record {index} must be a JSON object")
        validated_records.append(_validate_customer(record, f"Invalid input in record {index}"))

    predictions = state.service.predict_many(validated_records)
    return {"count": len(predictions), "predictions": predictions}


@app.post("/explain")
def explain(payload: dict[str, Any]):
    if state.service is None or state.explanation_service is None:
        raise HTTPException(status_code=503, detail="Model explanations are unavailable")
    validated = _validate_customer(payload)
    prediction = state.service.predict_one(validated)
    try:
        return state.explanation_service.explain(validated, prediction)
    except ShapArtifactError as exc:
        logger.error("Gradient Boosting SHAP artifacts are unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="SHAP explanation is unavailable") from exc
    except RuntimeError as exc:
        logger.exception("Could not generate a Gradient Boosting SHAP explanation")
        raise HTTPException(status_code=500, detail="SHAP explanation failed") from exc
