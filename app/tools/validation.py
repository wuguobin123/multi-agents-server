from __future__ import annotations

from typing import Any


_TYPE_MAPPING: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    if not schema:
        return []

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type:
        python_type = _TYPE_MAPPING.get(expected_type)
        if python_type is not None and not isinstance(payload, python_type):
            errors.append(f"payload must be {expected_type}")
            return errors

    required = schema.get("required", [])
    for field_name in required:
        if field_name not in payload:
            errors.append(f"missing required field: {field_name}")

    properties = schema.get("properties", {})
    for field_name, field_schema in properties.items():
        if field_name not in payload:
            continue
        field_type = field_schema.get("type")
        python_type = _TYPE_MAPPING.get(field_type)
        if python_type is not None and not isinstance(payload[field_name], python_type):
            errors.append(f"{field_name} must be {field_type}")

    return errors
