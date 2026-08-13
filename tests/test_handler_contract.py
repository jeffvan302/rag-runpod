import pytest

from embedding_contract import require_model_revision


def test_matching_model_revision_is_accepted():
    require_model_revision(
        "abc123",
        {"id": "Qwen/Qwen3-Embedding-4B", "snapshot_revision": "abc123"},
    )


def test_mismatched_model_revision_is_rejected():
    with pytest.raises(ValueError, match="Model revision mismatch"):
        require_model_revision(
            "older",
            {"id": "Qwen/Qwen3-Embedding-4B", "snapshot_revision": "newer"},
        )
