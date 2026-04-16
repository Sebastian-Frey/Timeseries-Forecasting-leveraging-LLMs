"""Normalised API-key access for Colab and local runs.

Module named `api_keys` (not `secrets`) to avoid shadowing the stdlib
`secrets` module, which some notebooks may import.
"""
from __future__ import annotations

import os

_NIXTLA_ALIASES = (
    "NIXTLA_API_KEY",
    "NIXTLA_FELIX_KEY",
    "NIXTLA_CHARLOTTE_KEY",
    "NIXTLA_PAUL_KEY",
    "NIXTLA_MARNO_KEY",
)


def get_nixtla_key() -> str:
    key = os.environ.get("NIXTLA_API_KEY", "").strip()
    if key:
        return key
    try:
        from google.colab import userdata
    except ImportError:
        raise RuntimeError(
            "NIXTLA_API_KEY is not set. Copy .env.example to .env and fill it."
        )
    for name in _NIXTLA_ALIASES:
        try:
            val = userdata.get(name)
        except Exception:
            val = None
        if val:
            return val
    raise RuntimeError("No Nixtla key found in env or Colab userdata.")


def get_hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _colab_userdata_first(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    )


def _colab_userdata_first(names: tuple[str, ...]) -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    for name in names:
        try:
            val = userdata.get(name)
        except Exception:
            val = None
        if val:
            return val
    return None
