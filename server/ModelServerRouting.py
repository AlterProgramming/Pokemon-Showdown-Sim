from __future__ import annotations

from typing import Any, Sequence


def is_entity_model_entry(model_entry: dict[str, Any]) -> bool:
    family_id = str(model_entry.get("family_id") or "")
    state_schema_version = str(model_entry.get("state_schema_version") or "")
    return family_id.startswith("entity_") or state_schema_version.startswith("entity_")


def request_prefers_entity_payload(request_data: dict[str, Any]) -> bool:
    return "battle_state" in request_data


def select_default_entity_model_id(
    model_ids: Sequence[str],
    model_artifacts: dict[str, dict[str, Any]],
) -> str | None:
    for model_id in model_ids:
        if is_entity_model_entry(model_artifacts.get(model_id, {})):
            return model_id
    return None


def choose_model_id_for_request(
    request_data: dict[str, Any],
    *,
    default_model_id: str,
    default_entity_model_id: str | None,
    supported_model_ids: Sequence[str],
) -> str:
    requested_model_id = request_data.get("model_id")
    if requested_model_id:
        model_id = str(requested_model_id)
        if model_id not in supported_model_ids:
            raise KeyError(
                f"Unsupported model_id '{requested_model_id}'. "
                f"Supported values: {', '.join(sorted(supported_model_ids))}"
            )
        return model_id

    if request_prefers_entity_payload(request_data):
        if default_entity_model_id is None:
            raise KeyError("battle_state payloads require an entity model, but none were loaded.")
        return default_entity_model_id

    if default_model_id not in supported_model_ids:
        raise KeyError("No default model is loaded.")
    return default_model_id
