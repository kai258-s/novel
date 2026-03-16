import json
import re
import time
from typing import Any

from loguru import logger


def call_chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 120,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """Call chat completion API with retry for transient failures."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content.")
            return content.strip()
        except Exception as exc:
            last_error = exc
            logger.warning("LLM call failed (attempt {}/{}): {}", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts.") from last_error


def parse_json_from_text(text: str) -> Any:
    """Best-effort JSON extraction from raw LLM response text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = []
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model output.")


def safe_filename(name: str, fallback: str = "untitled") -> str:
    value = (name or "").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or fallback

