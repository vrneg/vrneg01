"""Shared naming and validation helpers for anchor-relative event windows.

Dataset creation represents an interval with two context-distance values:
``window_ms_left`` is subtracted from the anchor and ``window_ms_right`` is
added to it.  Allowing either value to be negative makes fully pre-anchor and
fully post-anchor sliding windows possible without changing that established
API.  Dataset names encode a negative integer as ``mN`` because a literal
minus sign would produce ambiguous double-hyphen names such as ``wR--500``.
"""

from __future__ import annotations


SIGNED_MILLISECONDS_PATTERN = r"m?\d+"


def encode_signed_milliseconds(value: int) -> str:
    """Return the canonical dataset-name token for an integer millisecond value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("window millisecond values must be integers")
    return f"m{-value}" if value < 0 else str(value)


def decode_signed_milliseconds(value: str) -> int:
    """Decode a dataset-name token, where ``mN`` denotes ``-N``."""

    if not isinstance(value, str) or not value:
        raise ValueError("window millisecond token must be a non-empty string")
    if value.startswith("m"):
        magnitude = value[1:]
        if not magnitude.isdigit():
            raise ValueError(f"invalid signed millisecond token {value!r}")
        return -int(magnitude)
    if not value.isdigit():
        raise ValueError(f"invalid signed millisecond token {value!r}")
    return int(value)


def context_window_bounds_ms(
    window_ms_left: int,
    window_ms_right: int,
) -> tuple[int, int]:
    """Convert context distances to the actual anchor-relative interval.

    The returned bounds are ``(-window_ms_left, window_ms_right)``.  They must
    define a non-empty increasing interval.  Existing non-negative distances
    retain their original meaning.
    """

    for name, value in (
        ("window_ms_left", window_ms_left),
        ("window_ms_right", window_ms_right),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    window_start_ms = -window_ms_left
    window_end_ms = window_ms_right
    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window bounds must define a non-empty increasing interval: "
            f"start={window_start_ms} ms, end={window_end_ms} ms "
            f"(window_ms_left + window_ms_right must be greater than zero)"
        )
    return window_start_ms, window_end_ms


__all__ = [
    "SIGNED_MILLISECONDS_PATTERN",
    "context_window_bounds_ms",
    "decode_signed_milliseconds",
    "encode_signed_milliseconds",
]
