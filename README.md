# Customer Churn Prediction API

A FastAPI backend for the exported scikit-learn customer churn pipeline. The backend does not retrain or modify the model.

## Architecture

- `app/main.py`: FastAPI application, startup lifespan, endpoints, and HTTP error responses.
- `app/model_loader.py`: Validates deployment metadata and loads the joblib pipeline once at startup.
- `app/schemas.py`: Creates Pydantic request models from `feature_schema.json`.
- `app/services/prediction_service.py`: Reorders features, calls the pipeline, and formats predictions.
- `app/services/shap_service.py`: Lazily creates a Gradient Boosting permutation explainer from its encoded background data.
- `deployment/`: Preserved model artifacts.

The saved pipeline already contains its imputation, scaling, and one-hot encoding steps. The API validates raw customer features and passes them to that pipeline without recreating preprocessing.

## Setup

Python 3.12 is recommended because `customer_churn_model.joblib` was created with scikit-learn 1.6.1.

```bash
py -3.12 -m venv .venv
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

### GET /feature-importance

Returns mean absolute SHAP importance for the Gradient Boosting model, aligned with the original feature order. It reads only the Gradient Boosting column from `gradient_boosting_shap_importance.csv`.

### POST /explain

Accepts the same customer object as `/predict`. It computes a fresh local explanation for that request using `shap_background_encoded.pkl` and the deployed pipeline. Positive SHAP values increase churn probability; negative values reduce it. The saved 1,000-row `gradient_boosting_shap_values.npy` sample is not used as an explanation for API customers.

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

The `Procfile` already contains this command. Railway supplies `$PORT`; do not set a fixed local port.

The production service needs these files under `deployment/`: `customer_churn_model.joblib`, `feature_schema.json`, `model_metadata.json`, `final_results.json`, `category_mappings.json`, `shap_feature_names.json`, `gradient_boosting_shap_importance.csv`, and `shap_background_encoded.pkl`.

The following are not loaded by the production API and do not need to be uploaded to Railway: `Datasets/`, `customer-churn.ipynb`, `Explain/`, `gradient_boosting_shap_values.npy`, `gradient_boosting_summary.png`, `gradient_boosting_importance.png`, `gradient_boosting_permutation_importance.csv`, `gradient_boosting_rank_comparison.csv`, `shap_sample.pkl`, `shap_sample_encoded.pkl`, and `shap_background.pkl`. Do not upload training data or notebooks containing customer records to a public repository; verify any notebook outputs before sharing.
