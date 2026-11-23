from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator


NUMBER_TYPES = (StrictInt, StrictFloat)


def create_customer_model(
    feature_schema: dict[str, str],
    allowed_categories: dict[str, list[str]] | None = None,
) -> type[BaseModel]:
    annotations: dict[str, Any] = {}
    allowed_categories = allowed_categories or {}
    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "model_config": ConfigDict(extra="forbid"),
    }

    for feature_name, data_type in feature_schema.items():
        if data_type == "number":
            annotations[feature_name] = StrictInt | StrictFloat
            namespace[feature_name] = ...
        elif data_type == "string":
            annotations[feature_name] = str
            namespace[feature_name] = ...
        else:
            raise ValueError(f"Unsupported feature type in schema: {data_type}")

    if allowed_categories:
        for feature_name, categories in allowed_categories.items():
            def category_validator(value: str, c: list[str] = categories) -> str:
                return _validate_category(value, c)

            namespace[f"validate_{feature_name.replace(' ', '_')}"] = field_validator(
                feature_name, mode="after"
            )(category_validator)

    model = type("CustomerChurnRequest", (BaseModel,), namespace)
    return model


def _validate_category(value: str, categories: list[str]) -> str:
    if value not in categories:
        allowed = ", ".join(repr(category) for category in categories)
        raise ValueError(f"must be one of: {allowed}")
    return value
