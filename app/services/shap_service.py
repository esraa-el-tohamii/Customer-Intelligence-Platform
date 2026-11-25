import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ShapArtifactError(RuntimeError):
    pass


class ShapExplanationService:
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        category_mappings: dict[str, list[str]],
        background_path: Path,
        positive_class: int,
        model_name: str,
    ):
        self.model = model
        self.feature_names = feature_names
        self.category_mappings = category_mappings
        self.background_path = background_path
        self.positive_index = list(model.classes_).index(positive_class)
        self.model_name = model_name
        self._explainer: Any = None
        self._lock = threading.Lock()

    def _encode_record(self, record: dict[str, Any]) -> pd.DataFrame:
        encoded: list[float] = []
        for feature_name in self.feature_names:
            value = record[feature_name]
            if feature_name in self.category_mappings:
                value = self.category_mappings[feature_name].index(value)
            encoded.append(float(value))
        return pd.DataFrame([encoded], columns=self.feature_names)

    def _decode_rows(self, encoded: pd.DataFrame) -> pd.DataFrame:
        decoded = encoded.copy()
        for feature_name, categories in self.category_mappings.items():
            indices = np.rint(decoded[feature_name].to_numpy(dtype=float)).astype(int)
            decoded[feature_name] = [
                categories[index] if 0 <= index < len(categories) else categories[0]
                for index in indices
            ]
        return decoded.loc[:, self.feature_names]

    def _predict_churn_probability(self, encoded: pd.DataFrame) -> np.ndarray:
        decoded = self._decode_rows(encoded)
        probabilities = np.asarray(self.model.predict_proba(decoded))
        return probabilities[:, self.positive_index]

    def _get_explainer(self) -> Any:
        if self._explainer is not None:
            return self._explainer

        try:
            background = pd.read_pickle(self.background_path)
            if (
                not isinstance(background, pd.DataFrame)
                or background.empty
                or list(background.columns) != self.feature_names
            ):
                raise ShapArtifactError("Gradient Boosting SHAP background has an invalid structure")
            background = background.astype(float)
            if not np.isfinite(background.to_numpy()).all():
                raise ShapArtifactError("Gradient Boosting SHAP background contains invalid values")

            import shap

            self._explainer = shap.Explainer(
                self._predict_churn_probability,
                background,
                algorithm="permutation",
                feature_names=self.feature_names,
                seed=42,
            )
        except ShapArtifactError:
            raise
        except Exception as exc:
            logger.exception("Could not initialize the Gradient Boosting SHAP explainer")
            raise ShapArtifactError("Gradient Boosting SHAP explainer is unavailable") from exc
        return self._explainer

    def explain(self, record: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        encoded_record = self._encode_record(record)
        with self._lock:
            try:
                explainer = self._get_explainer()
                explanation = explainer(
                    encoded_record,
                    max_evals=2 * len(self.feature_names) + 1,
                )
                values = np.asarray(explanation.values).reshape(-1)
                base_values = np.asarray(explanation.base_values).reshape(-1)
            except ShapArtifactError:
                raise
            except Exception as exc:
                logger.exception("Gradient Boosting SHAP explanation failed")
                raise RuntimeError("Gradient Boosting SHAP explanation failed") from exc

        if values.size != len(self.feature_names) or base_values.size == 0:
            raise RuntimeError("Gradient Boosting SHAP explanation returned an invalid shape")

        features = []
        for index, feature_name in enumerate(self.feature_names):
            contribution = float(values[index])
            if contribution > 0:
                impact = "increases_churn"
            elif contribution < 0:
                impact = "decreases_churn"
            else:
                impact = "no_effect"
            features.append(
                {
                    "feature": feature_name,
                    "value": record[feature_name],
                    "shap_value": round(contribution, 6),
                    "impact": impact,
                }
            )

        return {
            "model_name": self.model_name,
            "prediction": prediction["prediction"],
            "prediction_label": prediction["prediction_label"],
            "churn_probability": prediction["churn_probability"],
            "baseline_churn_probability": round(float(base_values[0]), 6),
            "features": features,
            "interpretation": (
                "SHAP values are contributions to churn probability; positive values increase "
                "the probability and negative values decrease it."
            ),
        }
