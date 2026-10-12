"""Parse JSON with orjson, and give the stdlib's answer when orjson refuses.

A large catalog spends most of its graph load parsing documents, and orjson
parses them several times faster than :func:`json.loads`. The two parsers
disagree on some input, though. orjson rejects ``NaN``, ``Infinity``, and a
lone surrogate escape, all of which the stdlib accepts. When orjson rejects a
document, :func:`loads` asks the stdlib, so a document parses exactly when it
did before and a parse failure carries the stdlib's message, which findings
quote.

One difference stays. orjson reads an integer beyond the 64-bit range as a
float, without an error. No STAC field holds such a value.
"""

from __future__ import annotations

import json
from typing import Any

import orjson


def loads(raw: bytes | str) -> Any:
    """Parse ``raw``; bytes are UTF-8, as every document rashid reads must be.

    Raises :class:`ValueError` as :func:`json.loads` does, including
    :class:`UnicodeDecodeError` for bytes that are not UTF-8.
    """
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text)
