"""Optional tqdm-backed progress reporting with a plain-text fallback."""
from __future__ import annotations

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # Keep core training usable after a partial dependency install.
    _tqdm = None


class _PlainProgress:
    def __init__(self, iterable, **_):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **_):
        return None


def progress(iterable, **kwargs):
    return _tqdm(iterable, **kwargs) if _tqdm is not None else _PlainProgress(iterable, **kwargs)


def progress_write(message: str) -> None:
    if _tqdm is not None:
        _tqdm.write(message)
    else:
        print(message, flush=True)
