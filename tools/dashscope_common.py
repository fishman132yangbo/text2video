from __future__ import annotations

import json
from typing import Any


class NonRetryableProviderError(RuntimeError):
    """Provider rejected the request; retrying the same payload will not help."""


def compact_response(payload: Any, max_chars: int = 1000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)
    if len(text) > max_chars:
        return f"{text[:max_chars]}..."
    return text


def provider_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if message:
                return str(message)
        output = payload.get("output")
        if isinstance(output, dict):
            message = output.get("message") or output.get("task_status")
            if message:
                return str(message)
        message = payload.get("message")
        if message:
            return str(message)
    return compact_response(payload)


def raise_for_provider_status(prefix: str, status: int, payload: Any) -> None:
    if status < 400:
        return
    message = provider_error_message(payload)
    error_cls = NonRetryableProviderError if status in {400, 401, 403, 404, 429} else RuntimeError
    raise error_cls(f"{prefix} failed with HTTP {status}: {message}; response={compact_response(payload)}")
