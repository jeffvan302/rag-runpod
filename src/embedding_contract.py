from __future__ import annotations

from typing import Any


MODEL_NATIVE_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
}

MODEL_OUTPUT_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-4B": (1024, 2560),
    "Qwen/Qwen3-Embedding-0.6B": (1024,),
}

QUERY_PREFIX = (
    "Instruct: Retrieve the most relevant document chunks for the user's question\n"
    "Query: "
)


def normalize_output_dimensions(model_id: str, value: Any) -> int:
    supported = MODEL_OUTPUT_DIMENSIONS.get(model_id)
    if not supported:
        raise ValueError(f"Unsupported embedding model: {model_id}")

    if value is None or value == "":
        return MODEL_NATIVE_DIMENSIONS[model_id]

    try:
        dimensions = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding dimensions must be an integer.") from error

    if dimensions not in supported:
        choices = ", ".join(str(item) for item in supported)
        raise ValueError(
            f"{model_id} supports configured output dimensions: {choices}."
        )

    return dimensions


def format_embedding_text(value: Any, purpose: str) -> str:
    clean = str(value or "").strip() or " "
    if purpose == "document":
        return clean
    if purpose != "query":
        raise ValueError("embedding purpose must be query or document.")
    return clean if clean.startswith(QUERY_PREFIX) else f"{QUERY_PREFIX}{clean}"


def require_model_revision(value: Any, model_state: dict[str, Any]) -> None:
    requested = str(value or "").strip()
    if requested and requested != model_state["snapshot_revision"]:
        raise ValueError(
            f"Model revision mismatch for {model_state['id']}: "
            f"requested {requested}, loaded {model_state['snapshot_revision']}."
        )
