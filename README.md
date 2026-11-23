# Customer Churn Prediction API

A FastAPI backend for the exported scikit-learn customer churn pipeline. The backend does not retrain or modify the model.

## Architecture

- `app/main.py`: FastAPI application, startup lifespan, endpoints, and HTTP error responses.
- `app/model_loader.py`: Loads the deployment JSON files and joblib pipeline once at startup.
- `app/schemas.py`: Creates Pydantic request models from `feature_schema.json`.
- `app/services/prediction_service.py`: Reorders features, calls the pipeline, and formats predictions.
- `deployment/`: Preserved model artifacts.

## Setup

Python 3.12 is recommended because `customer_churn_model.joblib` was created with scikit-learn 1.6.1.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- API: `http://localhost:8000/`
- Health: `http://localhost:8000/health`

## Endpoints

### GET /
Returns API name, status, and description.

### GET /health
Reports whether the model loaded successfully.

### GET /model-info
Returns non-sensitive model metadata.

### POST /predict
Accepts one customer object containing all 12 required features.

```json
{
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
  "Support_Calls_Per_Usage": 0.25
}
```

Example response:

```json
{
  "prediction": 1,
  "prediction_label": "Churned",
  "churn_probability": 0.871234,
  "retention_probability": 0.128766,
  "message": "This customer is likely to churn."
}
```

### POST /predict/batch

```json
{
  "records": [
    {
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
      "Support_Calls_Per_Usage": 0.25
    }
  ]
}
```

The batch response contains `count` and a `predictions` array. Missing fields, unexpected fields, invalid types, invalid categorical values, and empty batches return HTTP 422.

## Test

```bash
pytest -q
```

## Railway deployment

1. Push this project to GitHub or connect an existing repository.
2. Create a Railway service and set the root directory to the project directory.
3. Railway installs dependencies from `requirements.txt`.
4. Set the start command:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The `Procfile` also contains this command. Railway supplies `$PORT`; do not set a fixed local port. Commit all files under `deployment/` so the model and JSON metadata are available at runtime.
