from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
import requests
import runpod

from chunking import normalize_chunking_settings, split_markdown_for_chunks, utf16_offsets
from embedding_contract import (
    MODEL_NATIVE_DIMENSIONS,
    MODEL_OUTPUT_DIMENSIONS,
    format_embedding_text,
    normalize_output_dimensions,
    require_model_revision,
)


PROTOCOL_VERSION = int(os.getenv("BUBBLE_RAG_PROTOCOL_VERSION", "1"))
WORKER_VERSION = "0.3.1"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
MODEL_ALIASES = {
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen/qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen/qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}
MAX_MARKDOWN_BYTES = int(os.getenv("BUBBLE_RAG_MAX_MARKDOWN_BYTES", str(50 * 1024 * 1024)))
MAX_DIRECT_INPUTS = int(os.getenv("BUBBLE_RAG_MAX_DIRECT_INPUTS", "64"))
MAX_DIRECT_INPUT_BYTES = int(os.getenv("BUBBLE_RAG_MAX_DIRECT_INPUT_BYTES", str(1024 * 1024)))
DEFAULT_BATCH_SIZE = int(os.getenv("BUBBLE_RAG_EMBED_BATCH_SIZE", "8"))
MANIFEST_PATH = Path(
    os.getenv(
        "BUBBLE_RAG_MODEL_MANIFEST",
        "/runpod-volume/bubble-rag/models/manifest.json",
    )
)
HF_CACHE_ROOT = os.getenv("HF_HOME", "/runpod-volume/huggingface")

_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = as_record(job.get("input"))
    operation = str(job_input.get("operation") or "").strip()

    try:
        assert_protocol(job_input)

        if operation == "health":
            return health()

        if operation == "preload_models":
            return preload_models(job_input)

        if operation == "embed_texts":
            return embed_texts(job_input)

        if operation == "chunk_and_embed":
            return chunk_and_embed(job, job_input)

        raise ValueError(f"Unsupported operation: {operation or '(missing)'}")
    except Exception as error:
        if operation == "chunk_and_embed":
            post_failure_callback(job, job_input, error)
        raise


def health() -> dict[str, Any]:
    return {
        "ok": True,
        "operation": "health",
        "protocolVersion": PROTOCOL_VERSION,
        "workerVersion": WORKER_VERSION,
        "device": device_summary(),
        "allowedModels": allowed_models(),
        "defaultModel": default_model(),
        "manifest": read_manifest_summary(),
    }


def preload_models(job_input: dict[str, Any]) -> dict[str, Any]:
    models = normalize_models(job_input.get("models") or allowed_models())
    selected_default = normalize_model_id(job_input.get("defaultModel") or default_model())

    if selected_default not in models:
        raise ValueError("defaultModel must be included in models.")

    manifest_models = []
    for model_id in models:
        require_allowed_model(model_id)
        state = get_model_state(model_id)
        manifest_models.append(
            {
                "id": model_id,
                "dimensions": state["dimensions"],
                "supportedDimensions": list(MODEL_OUTPUT_DIMENSIONS[model_id]),
                "snapshotPath": state["snapshot_path"],
                "snapshotRevision": state["snapshot_revision"],
                "device": state["device"],
            }
        )

    manifest = {
        "ok": True,
        "operation": "preload_models",
        "protocolVersion": PROTOCOL_VERSION,
        "models": manifest_models,
        "defaultModel": selected_default,
        "updatedAt": iso_now(),
    }
    write_manifest(manifest)
    return manifest


def embed_texts(job_input: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    model_id = normalize_model_id(job_input.get("model") or default_model())
    require_allowed_model(model_id)
    dimensions = normalize_output_dimensions(model_id, job_input.get("dimensions"))
    purpose = str(job_input.get("purpose") or "query").strip().lower()
    raw_input = job_input.get("input")

    if raw_input is None:
        raise ValueError("input is required.")

    raw_texts = raw_input if isinstance(raw_input, list) else [raw_input]

    if not raw_texts or len(raw_texts) > MAX_DIRECT_INPUTS:
        raise ValueError(f"input must contain between 1 and {MAX_DIRECT_INPUTS} texts.")

    texts = [format_embedding_text(value, purpose) for value in raw_texts]
    input_bytes = sum(len(text.encode("utf-8")) for text in texts)
    if input_bytes > MAX_DIRECT_INPUT_BYTES:
        raise ValueError("Direct embedding input exceeds the worker size limit.")

    state = get_model_state(model_id)
    require_model_revision(job_input.get("revision"), state)
    batch_size = min(positive_int(job_input.get("batchSize"), DEFAULT_BATCH_SIZE), 64)
    vectors = encode_texts(state, texts, batch_size, dimensions)

    return {
        "ok": True,
        "operation": "embed_texts",
        "protocolVersion": PROTOCOL_VERSION,
        "workerVersion": WORKER_VERSION,
        "model": model_id,
        "modelRevision": state["snapshot_revision"],
        "dimensions": dimensions,
        "purpose": purpose,
        "embeddings": vectors.tolist(),
        "elapsedMs": elapsed_ms(started),
    }


def chunk_and_embed(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    job_id = required_string(job_input, "jobId")
    job_token = required_string(job_input, "jobToken")
    download_url = required_string(job_input, "downloadUrl")
    results_url = required_string(job_input, "resultsUrl")
    attempt_id = str(job_input.get("attemptId") or uuid4())
    runpod_job_id = str(job.get("id") or job.get("jobId") or "")

    embedding = as_record(job_input.get("embedding"))
    model_id = normalize_model_id(embedding.get("model") or default_model())
    require_allowed_model(model_id)
    output_dimensions = normalize_output_dimensions(model_id, embedding.get("dimensions"))

    download_started = time.perf_counter()
    markdown = download_markdown(download_url, job_token, as_record(job_input.get("document")))
    download_ms = elapsed_ms(download_started)

    chunk_started = time.perf_counter()
    chunk_size_tokens, overlap_tokens = normalize_chunking_settings(
        as_record(job_input.get("chunking"))
    )
    chunks = split_markdown_for_chunks(markdown, chunk_size_tokens, overlap_tokens)
    code_unit_offsets = utf16_offsets(markdown)
    chunking_ms = elapsed_ms(chunk_started)

    model_state = get_model_state(model_id)
    require_model_revision(embedding.get("revision"), model_state)
    batch_size = positive_int(embedding.get("batchSize"), DEFAULT_BATCH_SIZE)
    total_batches = max(1, math.ceil(len(chunks) / batch_size))
    batch_hashes: list[str] = []
    embedded_chunks = 0
    callback_ms = 0
    embedding_started = time.perf_counter()

    for batch_index, first in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[first : first + batch_size]
        vectors = encode_texts(
            model_state,
            [chunk.text for chunk in batch],
            batch_size,
            output_dimensions,
        )
        payload_chunks = []

        for offset, chunk in enumerate(batch):
            vector = vectors[offset]
            start_offset = code_unit_offsets[chunk.start_offset]
            end_offset = code_unit_offsets[chunk.end_offset]
            payload_chunks.append(
                {
                    "chunkIndex": first + offset,
                    "startOffset": start_offset,
                    "endOffset": end_offset,
                    "tokenCount": chunk.token_count,
                    "overlapTokens": chunk.overlap_tokens,
                    "contentSha256": sha256_text(chunk.text),
                    "vectorEncoding": "float32-le-base64",
                    "vector": encode_float32_vector(vector),
                }
            )

        batch_hash = sha256_json(payload_chunks)
        batch_hashes.append(batch_hash)
        payload = {
            "protocolVersion": PROTOCOL_VERSION,
            "jobId": job_id,
            "runpodJobId": runpod_job_id,
            "attemptId": attempt_id,
            "kind": "batch",
            "batchIndex": batch_index,
            "firstChunk": first,
            "lastChunk": first + len(batch) - 1,
            "totalChunks": len(chunks),
            "batchSha256": batch_hash,
            "chunks": payload_chunks,
        }
        callback_started = time.perf_counter()
        post_result(results_url, job_token, f"{job_id}:{attempt_id}:{batch_index}:{batch_hash}", payload)
        callback_ms += elapsed_ms(callback_started)
        embedded_chunks += len(batch)

    embedding_ms = elapsed_ms(embedding_started)
    result_sha = sha256_json(
        {
            "jobId": job_id,
            "attemptId": attempt_id,
            "totalChunks": len(chunks),
            "totalBatches": total_batches,
            "batchSha256": batch_hashes,
        }
    )
    complete_payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "jobId": job_id,
        "runpodJobId": runpod_job_id,
        "attemptId": attempt_id,
        "kind": "complete",
        "totalChunks": len(chunks),
        "totalBatches": total_batches,
        "documentSha256": as_record(job_input.get("document")).get("sha256"),
        "resultSha256": result_sha,
        "timings": {
            "downloadMs": download_ms,
            "chunkingMs": chunking_ms,
            "embeddingMs": embedding_ms,
            "callbackMs": callback_ms,
        },
    }
    callback_started = time.perf_counter()
    post_result(results_url, job_token, f"{job_id}:{attempt_id}:complete:{result_sha}", complete_payload)
    callback_ms += elapsed_ms(callback_started)
    complete_payload["timings"]["callbackMs"] = callback_ms

    return {
        "ok": True,
        "operation": "chunk_and_embed",
        "protocolVersion": PROTOCOL_VERSION,
        "jobId": job_id,
        "attemptId": attempt_id,
        "model": model_id,
        "dimensions": output_dimensions,
        "chunks": embedded_chunks,
        "totalBatches": total_batches,
        "elapsedMs": elapsed_ms(started),
    }


def get_model_state(model_id: str) -> dict[str, Any]:
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch

    snapshot_path = snapshot_download(repo_id=model_id, cache_dir=HF_CACHE_ROOT)
    device = preferred_device()
    model_kwargs = {}

    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        snapshot_path,
        device=device,
        model_kwargs=model_kwargs or None,
    )
    probe = model.encode(
        ["BubbleAPI RunPod embedding preload check."],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dimensions = int(probe.shape[-1])
    expected = MODEL_NATIVE_DIMENSIONS.get(model_id)

    if expected is not None and dimensions != expected:
        raise ValueError(f"{model_id} returned {dimensions} dimensions, expected {expected}.")

    state = {
        "model": model,
        "id": model_id,
        "dimensions": dimensions,
        "snapshot_path": snapshot_path,
        "snapshot_revision": Path(snapshot_path).name,
        "device": device,
    }
    _MODEL_CACHE.clear()
    _MODEL_CACHE[model_id] = state
    return state


def encode_texts(
    model_state: dict[str, Any],
    texts: list[str],
    batch_size: int,
    output_dimensions: int,
) -> np.ndarray:
    model = model_state["model"]
    vectors = model.encode(
        [text.strip() or " " for text in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
        truncate_dim=output_dimensions,
    )
    vectors = np.asarray(vectors, dtype="<f4")

    if vectors.ndim != 2 or vectors.shape[1] != output_dimensions:
        raise ValueError("Embedding model returned an unexpected vector shape.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned non-finite values.")

    return vectors


def download_markdown(url: str, token: str, document: dict[str, Any]) -> str:
    response = requests.get(
        url,
        headers={"authorization": f"Bearer {token}", "accept": "text/markdown"},
        timeout=120,
    )
    response.raise_for_status()
    content = response.content

    if len(content) > MAX_MARKDOWN_BYTES:
        raise ValueError("Markdown document exceeds the RunPod worker size limit.")

    expected_size = document.get("byteSize")
    if isinstance(expected_size, int) and expected_size >= 0 and len(content) != expected_size:
        raise ValueError("Downloaded Markdown byte size does not match the job request.")

    expected_sha = str(document.get("sha256") or "").strip().lower()
    actual_sha = hashlib.sha256(content).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("Downloaded Markdown checksum does not match the job request.")

    return content.decode("utf-8-sig")


def post_failure_callback(job: dict[str, Any], job_input: dict[str, Any], error: Exception) -> None:
    results_url = str(job_input.get("resultsUrl") or "")
    job_token = str(job_input.get("jobToken") or "")
    job_id = str(job_input.get("jobId") or "")

    if not results_url or not job_token or not job_id:
        return

    attempt_id = str(job_input.get("attemptId") or "")
    payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "jobId": job_id,
        "runpodJobId": str(job.get("id") or job.get("jobId") or ""),
        "attemptId": attempt_id,
        "kind": "failure",
        "errorCode": "runpod_worker_failed",
        "message": safe_error_message(error),
    }
    failure_hash = sha256_json(payload)

    try:
        post_result(results_url, job_token, f"{job_id}:{attempt_id}:failure:{failure_hash}", payload)
    except Exception:
        return


def post_result(url: str, token: str, idempotency_key: str, payload: dict[str, Any]) -> None:
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "idempotency-key": idempotency_key,
    }
    last_error = "unknown callback error"

    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if 200 <= response.status_code < 300:
                return

            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                break
        except requests.RequestException as error:
            last_error = str(error)

        time.sleep(min(8, 2**attempt))

    raise RuntimeError(f"Railway callback failed: {last_error}")


def assert_protocol(job_input: dict[str, Any]) -> None:
    if int(job_input.get("protocolVersion") or 0) != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocolVersion. Expected {PROTOCOL_VERSION}.")


def allowed_models() -> list[str]:
    raw = os.getenv("BUBBLE_RAG_ALLOWED_MODELS", "")
    if raw:
        try:
            return normalize_models(json.loads(raw))
        except Exception:
            pass
    return [DEFAULT_MODEL]


def default_model() -> str:
    return normalize_model_id(os.getenv("BUBBLE_RAG_DEFAULT_MODEL") or DEFAULT_MODEL)


def normalize_models(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_models = [entry.strip() for entry in value.split(",")]
    elif isinstance(value, list):
        raw_models = value
    else:
        raise ValueError("models must be a string or array.")

    models = []
    for raw in raw_models:
        model_id = normalize_model_id(raw)
        if model_id not in models:
            models.append(model_id)

    if not models:
        raise ValueError("models must contain at least one model.")

    return models


def normalize_model_id(value: Any) -> str:
    cleaned = str(value or "").strip()
    alias = MODEL_ALIASES.get(cleaned.lower())
    model_id = alias or cleaned

    if "/" not in model_id:
        raise ValueError(f"Invalid Hugging Face model ID: {cleaned}")

    return model_id


def require_allowed_model(model_id: str) -> None:
    if model_id not in allowed_models():
        raise ValueError(f"Model is not in this worker's allowlist: {model_id}")


def preferred_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def device_summary() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            return {
                "type": "cuda",
                "name": torch.cuda.get_device_name(index),
                "count": torch.cuda.device_count(),
            }
        return {"type": "cpu", "count": 0}
    except Exception as error:
        return {"type": "unknown", "error": safe_error_message(error)}


def write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest_summary() -> dict[str, Any] | None:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {
            "ok": manifest.get("ok") is True,
            "models": manifest.get("models") or [],
            "defaultModel": manifest.get("defaultModel"),
            "updatedAt": manifest.get("updatedAt"),
        }
    except FileNotFoundError:
        return None
    except Exception as error:
        return {"ok": False, "error": safe_error_message(error)}


def encode_float32_vector(vector: np.ndarray) -> str:
    little_endian = np.asarray(vector, dtype="<f4")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def required_string(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required.")
    return value


def positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def safe_error_message(error: Exception) -> str:
    return str(error).strip()[:1000] or error.__class__.__name__


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
