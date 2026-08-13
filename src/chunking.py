from __future__ import annotations

from dataclasses import dataclass
import math
import re


DEFAULT_CHUNK_SIZE_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
MIN_CHUNK_SIZE_TOKENS = 100
MAX_CHUNK_SIZE_TOKENS = 4000
MAX_CHUNK_OVERLAP_TOKENS = 1000


@dataclass(frozen=True)
class MarkdownChunk:
    text: str
    start_offset: int
    end_offset: int
    token_count: int
    overlap_tokens: int


def normalize_chunking_settings(value: dict | None) -> tuple[int, int]:
    value = value or {}
    chunk_size = clamp_integer(
        value.get("chunkSizeTokens"),
        MIN_CHUNK_SIZE_TOKENS,
        MAX_CHUNK_SIZE_TOKENS,
        DEFAULT_CHUNK_SIZE_TOKENS,
    )
    overlap = min(
        chunk_size - 1,
        clamp_integer(
            value.get("overlapTokens"),
            0,
            MAX_CHUNK_OVERLAP_TOKENS,
            DEFAULT_CHUNK_OVERLAP_TOKENS,
        ),
    )
    return chunk_size, overlap


def split_markdown_for_chunks(
    value: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[MarkdownChunk]:
    max_length = token_chars(chunk_size_tokens)
    overlap_length = token_chars(overlap_tokens)
    chunks: list[MarkdownChunk] = []
    start_offset = trim_start_offset(value, 0, len(value))
    previous_end_offset: int | None = None

    while start_offset < len(value):
        hard_end_offset = min(len(value), start_offset + max_length)
        preferred_end_offset = (
            find_preferred_chunk_end(value, start_offset, hard_end_offset, max_length)
            if hard_end_offset < len(value)
            else hard_end_offset
        )
        end_offset = trim_end_offset(
            value,
            start_offset,
            max(preferred_end_offset, start_offset + 1),
        )
        safe_end_offset = end_offset if end_offset > start_offset else hard_end_offset
        text = value[start_offset:safe_end_offset].strip()

        if text:
            overlap_text = (
                ""
                if previous_end_offset is None
                else value[start_offset : min(previous_end_offset, safe_end_offset)]
            )
            chunks.append(
                MarkdownChunk(
                    text=text,
                    start_offset=start_offset,
                    end_offset=safe_end_offset,
                    token_count=estimate_token_count(text),
                    overlap_tokens=0
                    if previous_end_offset is None
                    else estimate_token_count(overlap_text),
                )
            )

        if safe_end_offset >= len(value):
            break

        previous_end_offset = safe_end_offset
        next_start = (
            find_preferred_overlap_start(value, safe_end_offset, overlap_length)
            if overlap_tokens > 0
            else safe_end_offset
        )
        trimmed_next_start = trim_start_offset(value, next_start, len(value))

        if trimmed_next_start <= start_offset:
            start_offset = trim_start_offset(value, safe_end_offset, len(value))
        else:
            start_offset = trimmed_next_start

    if not chunks:
        text = value.strip() or "(empty document)"
        return [
            MarkdownChunk(
                text=text,
                start_offset=0,
                end_offset=max(1, len(value)),
                token_count=estimate_token_count(text),
                overlap_tokens=0,
            )
        ]

    return chunks


def find_preferred_chunk_end(
    value: str,
    start_offset: int,
    hard_end_offset: int,
    max_length: int,
) -> int:
    min_end_offset = min(hard_end_offset, start_offset + math.floor(max_length * 0.96))
    candidates = [
        last_boundary_before(value, "\n## Page ", min_end_offset, hard_end_offset, 0),
        last_boundary_before(value, "\n\n", min_end_offset, hard_end_offset, 2),
        last_boundary_before(value, "\n", min_end_offset, hard_end_offset, 1),
        last_sentence_boundary_before(value, min_end_offset, hard_end_offset),
    ]
    filtered = [candidate for candidate in candidates if candidate is not None]
    return max(filtered) if filtered else hard_end_offset


def find_preferred_overlap_start(value: str, end_offset: int, overlap_length: int) -> int:
    if overlap_length <= 0:
        return end_offset

    target_offset = max(0, end_offset - overlap_length)
    min_offset = max(0, target_offset - 120)
    max_offset = min(end_offset - 1, target_offset + 120)
    candidates = [
        first_sentence_boundary_after(value, min_offset, max_offset),
        first_boundary_after(value, "\n\n", min_offset, max_offset, 2),
        first_boundary_after(value, "\n", min_offset, max_offset, 1),
        first_word_boundary_after(value, target_offset, max_offset),
        last_word_boundary_before(value, min_offset, target_offset),
    ]
    filtered = [candidate for candidate in candidates if candidate is not None]

    if not filtered:
        return target_offset

    return sorted(filtered, key=lambda candidate: abs(candidate - target_offset))[0]


def first_word_boundary_after(value: str, min_offset: int, max_offset: int) -> int | None:
    for index in range(min_offset, max_offset + 1):
        if index < len(value) and value[index].isspace():
            return index + 1
    return None


def last_word_boundary_before(value: str, min_offset: int, max_offset: int) -> int | None:
    for index in range(max_offset, min_offset - 1, -1):
        if index < len(value) and value[index].isspace():
            return index + 1
    return None


def last_boundary_before(
    value: str,
    boundary: str,
    min_offset: int,
    max_offset: int,
    boundary_length: int,
) -> int | None:
    index = value.rfind(boundary, 0, max_offset + len(boundary))

    if index < min_offset:
        return None

    return index + boundary_length


def first_boundary_after(
    value: str,
    boundary: str,
    min_offset: int,
    max_offset: int,
    boundary_length: int,
) -> int | None:
    index = value.find(boundary, min_offset)

    if index < 0 or index > max_offset:
        return None

    return index + boundary_length


def last_sentence_boundary_before(
    value: str,
    min_offset: int,
    max_offset: int,
) -> int | None:
    for index in range(max_offset, min_offset - 1, -1):
        if is_sentence_boundary(value, index):
            return index + 1
    return None


def first_sentence_boundary_after(
    value: str,
    min_offset: int,
    max_offset: int,
) -> int | None:
    for index in range(min_offset, max_offset + 1):
        if is_sentence_boundary(value, index):
            return index + 1
    return None


def is_sentence_boundary(value: str, index: int) -> bool:
    if index < 0 or index >= len(value):
        return False
    next_character = value[index + 1] if index + 1 < len(value) else ""
    return value[index] in ".!?" and bool(next_character and next_character.isspace())


def trim_start_offset(value: str, start_offset: int, end_offset: int) -> int:
    offset = max(0, start_offset)

    while offset < end_offset and offset < len(value) and value[offset].isspace():
        offset += 1

    return offset


def trim_end_offset(value: str, start_offset: int, end_offset: int) -> int:
    offset = min(len(value), end_offset)

    while offset > start_offset and value[offset - 1].isspace():
        offset -= 1

    return offset


def estimate_token_count(value: str) -> int:
    return max(1, math.ceil(utf16_length(value) / 4))


def utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def token_chars(tokens: int) -> int:
    return max(1, tokens * 4)


def split_sentences(value: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", value)
    return [
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", normalized)
        if sentence.strip()
    ]


def utf16_offsets(value: str) -> list[int]:
    offsets = [0]
    total = 0

    for character in value:
        total += 2 if ord(character) > 0xFFFF else 1
        offsets.append(total)

    return offsets


def clamp_integer(value: object, minimum: int, maximum: int, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback

    if parsed < minimum or parsed > maximum:
        return fallback

    return parsed
