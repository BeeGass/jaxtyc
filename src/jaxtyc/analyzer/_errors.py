"""Error message utilities for the analyzer layer."""

from __future__ import annotations

_MAX_ERROR_LEN: int = 500
_ELLIPSIS = " ..."


def truncate_error(error: Exception | str, max_len: int = _MAX_ERROR_LEN) -> str:
    """Truncate a verbose exception message to a reasonable length.

    Takes the first line of the message. If it exceeds *max_len*, truncates
    and appends an ellipsis marker.

    Args:
        error: Exception instance or string to truncate.
        max_len: Maximum character length before truncation.

    Returns:
        Truncated error string.
    """
    msg = str(error)
    first_line = msg.split("\n")[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[:max_len] + _ELLIPSIS
