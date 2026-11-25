import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException


logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        model: Any,
        schema: dict[str, str],
        metadata: dict[str, Any],
        category_mappings: dict[str, list[str]],
    ):
        self.model = model
        self.schema = schema
        self.metadata = metadata
        self.feature_names = list(metadata["features"])
        self.prediction_classes = metadata["prediction_classes"]
        self.positive_class = metadata["positive_class"]
        self.classes = list(model.classes_)
        self.positive_index = self.classes.index(self.positive_class)
        self.stay_index = self.classes.index(0)
        self.allowed_categories = category_mappings

    def validate_categorical_values(self, records: list[dict[str, Any]]) -> None:
        for record_index, record in enumerate(records):
            for feature_name, allowed in self.allowed_categories.items():
                value = record.get(feature_name)
                if value not in allowed:
                    allowed_text = ", ".join(repr(category) for category in allowed)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Record {record_index}: '{feature_name}' must be one of: {allowed_text}",
                    )

    def _dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(records, columns=self.feature_names)

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
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

        if (
            predictions.size != len(records)
            or probabilities.ndim != 2
            or probabilities.shape != (len(records), len(self.classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            logger.error("Loaded model returned invalid prediction results")
            raise HTTPException(status_code=500, detail="Prediction failed")

        results: list[dict[str, Any]] = []
        label_lookup = {str(key): str(value) for key, value in self.prediction_classes.items()}
        for prediction, row_probabilities in zip(predictions, probabilities):
            class_value = int(prediction)
            churn_probability = float(row_probabilities[self.positive_index])
            retention_probability = float(row_probabilities[self.stay_index])
            prediction_label = label_lookup.get(str(class_value), str(class_value))
            message = (
                "This customer is likely to churn."
                if class_value == self.positive_class
                else "This customer is likely to stay."
            )

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
