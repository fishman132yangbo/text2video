"""Chat model config normalization for the Web video console."""

from typing import Any


def resolve_chat_model_config(init_args: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of chat model init args.

    The Web console uses explicit provider settings from YAML config files.
    Keeping this helper preserves the pipeline call sites without carrying
    provider-specific preset code that is no longer part of the main flow.
    """
    return dict(init_args)
