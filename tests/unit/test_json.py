"""Unit tests for rashid._json: orjson, with the stdlib's answer where they differ.

orjson rejects some documents the stdlib accepts. Each case here must parse
exactly as ``json.loads`` parses it, and a document neither accepts must fail
with the stdlib's message, because ``CatalogGraph.load`` quotes that message
in the parse-error finding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rashid import _json
from rashid.catalog import CatalogGraph


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b'{"a": -Infinity}',
        b'{"a": "\\ud800"}',
    ],
    ids=["nan", "infinity", "negative-infinity", "lone-surrogate"],
)
def test_input_orjson_rejects_parses_as_the_stdlib_parses_it(raw: bytes) -> None:
    expected = json.loads(raw.decode("utf-8"))
    actual = _json.loads(raw)
    if isinstance(expected, dict) and isinstance(expected["a"], float):
        assert math.isnan(actual["a"]) == math.isnan(expected["a"])
        if not math.isnan(expected["a"]):
            assert actual == expected
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "raw",
    [b"", b'{"a": 1},', b'{"a": }', b"\xef\xbb\xbf{}"],
    ids=["empty", "extra-data", "missing-value", "utf8-bom"],
)
def test_invalid_json_fails_with_the_stdlib_message(raw: bytes) -> None:
    with pytest.raises(json.JSONDecodeError) as expected:
        json.loads(raw.decode("utf-8"))
    with pytest.raises(ValueError) as actual:
        _json.loads(raw)
    assert str(actual.value) == str(expected.value)


def test_bytes_that_are_not_utf8_fail_as_a_decode_error() -> None:
    with pytest.raises(UnicodeDecodeError, match="'utf-8' codec"):
        _json.loads(b'{"a": "\xff"}')


def test_str_input_is_accepted() -> None:
    assert _json.loads('{"a": [1, 2.5, null, true]}') == {"a": [1, 2.5, None, True]}


@pytest.mark.parametrize(
    "raw",
    [b"", b'{"type": "Catalog",}', b"\xef\xbb\xbf{}", b'{"a": "\xff"}'],
    ids=["empty", "trailing-comma", "utf8-bom", "not-utf8"],
)
def test_a_catalog_that_fails_to_parse_keeps_its_parse_error(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "catalog.json").write_bytes(raw)
    try:
        json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        expected = str(exc)

    graph = CatalogGraph.load(tmp_path)

    (node,) = graph.nodes.values()
    assert node.parse_error == expected


def test_a_catalog_with_nan_still_loads(tmp_path: Path) -> None:
    (tmp_path / "catalog.json").write_bytes(
        b'{"type": "Catalog", "id": "c", "description": "d", "links": [], "x": NaN}'
    )

    graph = CatalogGraph.load(tmp_path)

    (node,) = graph.nodes.values()
    assert node.parse_error is None and node.kind == "catalog"
    assert math.isnan(node.data["x"])
