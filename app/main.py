from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.model_loader import DeploymentLoadError, load_deployment
from app.schemas import create_customer_model
from app.services.prediction_service import PredictionService


class AppState:
    model: Any = None
    schema: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    service: PredictionService | None = None
    load_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, schema, metadata = load_deployment()
        state.model = model
        state.schema = schema
        state.metadata = metadata
        state.load_error = None
        state.service = PredictionService(model, schema, metadata)
    except DeploymentLoadError as exc:
        state.load_error = str(exc)
        state.service = None
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
    if state.load_error:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable")
    metadata = state.metadata
    return {
        "model_name": metadata.get("model_name"),
        "project_name": metadata.get("project_name"),
        "number_of_features": metadata.get("number_of_features"),
        "expected_feature_names": metadata.get("features"),
        "target_column": metadata.get("target_column"),
        "prediction_classes": metadata.get("prediction_classes"),
        "evaluation_metrics": metadata.get("evaluation_metrics"),
    }


@app.post("/predict")
def predict(payload: dict):
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    CustomerRequest = create_customer_model(state.schema, state.service.allowed_categories)
    try:
        validated = CustomerRequest.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        raise HTTPException(status_code=422, detail={"message": "Invalid input", "errors": errors})
    return state.service.predict_one(validated.model_dump())


@app.post("/predict/batch")
def predict_batch(payload: dict):
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    records = payload.get("records")
    if "records" not in payload:
        raise HTTPException(status_code=422, detail="Request body must contain a 'records' field")
    if not isinstance(records, list):
        raise HTTPException(status_code=422, detail="'records' must be a JSON array")
    if not records:
        raise HTTPException(status_code=422, detail="Batch request must contain at least one customer record")

    CustomerRequest = create_customer_model(state.schema, state.service.allowed_categories)
    validated_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise HTTPException(status_code=422, detail=f"Record {index} must be a JSON object")
        try:
            validated_records.append(CustomerRequest.model_validate(record).model_dump())
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
                detail={"message": f"Invalid input in record {index}", "errors": errors},
            )

    predictions = state.service.predict_many(validated_records)
    return {"count": len(predictions), "predictions": predictions}
