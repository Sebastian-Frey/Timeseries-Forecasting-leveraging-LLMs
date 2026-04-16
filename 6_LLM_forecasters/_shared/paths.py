"""Data/path bootstrap for Colab and local runs.

Usage from a notebook two levels deep (e.g. `prophet/01_*.ipynb`):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('..', '_shared').resolve()))
    from paths import ensure_env, DATA_ROOT, KEYWORDS_DIR_5, DTW_DFS_DIR
    ensure_env()
"""
from __future__ import annotations

import os
from pathlib import Path

_COLAB_DEFAULT = Path("/content/drive/MyDrive/colab_data/cleaned_cpu")


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_env() -> None:
    """Mount Drive if in Colab, or load `.env` when running locally.

    Idempotent — safe to call more than once per session.
    """
    if _in_colab():
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break


def _resolve_root() -> Path:
    root = os.environ.get("PAPER_DATA_ROOT")
    if root:
        return Path(root)
    if _in_colab():
        return _COLAB_DEFAULT
    raise RuntimeError(
        "PAPER_DATA_ROOT is not set. Copy .env.example to .env and fill it."
    )


def _try_root() -> Path | None:
    try:
        return _resolve_root()
    except RuntimeError:
        return None


_root = _try_root()

DATA_ROOT: Path | None = _root
KEYWORDS_DIR_5: Path | None = (_root / "keywords_dfs_full_5") if _root else None
KEYWORDS_DIR_20: Path | None = (_root / "keywords_dfs_full_20") if _root else None
DTW_DFS_DIR: Path | None = (_root / "dtw_neighbour_dfs") if _root else None


def prophet_results_dir(suffix: str) -> Path:
    return _resolve_root() / suffix


def timegpt_results_dir(suffix: str) -> Path:
    return _resolve_root() / suffix


def moirai_results_dir(suffix: str) -> Path:
    return _resolve_root() / f"moirai_results_{suffix}"
