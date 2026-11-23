from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException


class PredictionService:
    def __init__(self, model: Any, schema: dict[str, str], metadata: dict[str, Any]):
        self.model = model
        self.schema = schema
        self.metadata = metadata
        self.feature_names = list(metadata["features"])
        self.prediction_classes = metadata["prediction_classes"]
        self.allowed_categories = self._get_allowed_categories()

    def _get_allowed_categories(self) -> dict[str, list[str]]:
        preprocessor = getattr(self.model, "named_steps", {}).get("preprocessor")
        if preprocessor is None:
            return {}

        categories: dict[str, list[str]] = {}
        for transformer in getattr(preprocessor, "transformers_", []):
            name, _, columns = transformer
            if name == "cat":
                encoder = None
                if hasattr(transformer[1], "named_steps"):
                    encoder = transformer[1].named_steps.get("encoder")
                if encoder is not None and hasattr(encoder, "categories_"):
                    for column, values in zip(columns, encoder.categories_):
                        categories[column] = [str(value) for value in values]
        return categories

    def validate_categorical_values(self, records: list[dict[str, Any]]) -> None:
        for record_index, record in enumerate(records):
            for feature_name, allowed in self.allowed_categories.items():
                value = record.get(feature_name)
                if value is not None and str(value) not in allowed:
                    allowed_text = ", ".join(repr(category) for category in allowed)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Record {record_index}: '{feature_name}' must be one of: {allowed_text}",
                    )

    def _dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(records)
        return frame.reindex(columns=self.feature_names)

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([record])[0]

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            raise HTTPException(status_code=422, detail="Batch request must contain at least one customer record")

        self.validate_categorical_values(records)
        frame = self._dataframe(records)

        try:
            predictions = np.asarray(self.model.predict(frame)).reshape(-1)
            probabilities = np.asarray(self.model.predict_proba(frame))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

        classes = list(getattr(self.model, "classes_", []))
        if len(classes) == 0:
            raise HTTPException(status_code=500, detail="Loaded model did not expose prediction classes")

        try:
            positive_index = classes.index(1)
        except ValueError:
            positive_index = int(np.argmax(classes))

        results: list[dict[str, Any]] = []
        for prediction, row_probabilities in zip(predictions, probabilities):
            class_value = int(prediction)
            positive_probability = float(row_probabilities[positive_index])
            retention_probability = float(row_probabilities[classes.index(0)]) if 0 in classes else float(1.0 - positive_probability)
            churn_probability = positive_probability

            label_lookup = {str(key): str(value) for key, value in self.prediction_classes.items()}
            prediction_label = label_lookup.get(str(class_value), str(class_value))
            if class_value == 1:
                message = "This customer is likely to churn."
            else:
                message = "This customer is likely to stay."

            results.append(
                {
                    "prediction": class_value,
                    "prediction_label": prediction_label,
                    "churn_probability": round(churn_probability, 6),
                    "retention_probability": round(retention_probability, 6),
                    "message": message,
                }
            )
        return results
