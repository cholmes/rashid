"""Unit tests for rashid._store: obstore reads against a local HTTPS server."""

from __future__ import annotations

import socket
from importlib.metadata import version

import pytest

from rashid import _store
from tests._https import HttpsServer, Route

_AGENT = f"rashid/{version('rashid')} (+https://github.com/portolan-sdi/rashid)"


def test_get_returns_the_body_and_sends_the_named_agent(https_server: HttpsServer) -> None:
    https_server.routes["/a/doc.json"] = Route(body=b'{"a": 1}')

    assert _store.get(https_server.url("a/doc.json")) == b'{"a": 1}'
    (seen,) = https_server.seen
    assert (seen.method, seen.path) == ("GET", "/a/doc.json")
    assert seen.headers["user-agent"] == _AGENT


@pytest.mark.parametrize("status", [403, 404, 405, 410, 500])
def test_get_reports_the_status(https_server: HttpsServer, status: int) -> None:
    https_server.routes["/x.json"] = Route(status=status)

    with pytest.raises(_store.StoreError) as raised:
        _store.get(https_server.url("x.json"))
    assert raised.value.status == status
    assert f"returned {status}" in str(raised.value)


def test_a_failure_is_an_os_error(https_server: HttpsServer) -> None:
    with pytest.raises(OSError):
        _store.get(https_server.url("missing.json"))


def test_get_without_a_server_has_no_status() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(_store.StoreError) as raised:
        _store.get(f"https://127.0.0.1:{port}/catalog.json")
    assert raised.value.status is None


def test_get_refuses_a_body_past_the_cap(https_server: HttpsServer) -> None:
    https_server.routes["/big.json"] = Route(body=b"x" * 100)

    with pytest.raises(_store.StoreError, match="exceeds 99 bytes"):
        _store.get(https_server.url("big.json"), max_bytes=99)
    assert _store.get(https_server.url("big.json"), max_bytes=100) == b"x" * 100


def test_get_follows_a_redirect_to_https(https_server: HttpsServer) -> None:
    https_server.routes["/old.json"] = Route(status=301, headers={"Location": "/new.json"})
    https_server.routes["/new.json"] = Route(body=b"{}")

    assert _store.get(https_server.url("old.json")) == b"{}"


def test_get_refuses_a_redirect_to_http(https_server: HttpsServer) -> None:
    https_server.routes["/old.json"] = Route(
        status=302, headers={"Location": "http://127.0.0.1:9/new.json"}
    )

    with pytest.raises(_store.StoreError) as raised:
        _store.get(https_server.url("old.json"))
    assert raised.value.status is None
    assert "redirect" in str(raised.value)


def test_a_response_without_content_length_fails(https_server: HttpsServer) -> None:
    https_server.routes["/chunked.json"] = Route(body=b"{}", length=False)

    with pytest.raises(_store.StoreError, match="Content-Length"):
        _store.get(https_server.url("chunked.json"))


def test_the_path_is_encoded_once(https_server: HttpsServer) -> None:
    https_server.routes["/a%20b/c%25d.json"] = Route(body=b"{}")

    assert _store.get(https_server.url("a%20b/c%25d.json")) == b"{}"
    assert https_server.seen[0].path == "/a%20b/c%25d.json"


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("http://example.org/x.json", ValueError),
        ("s3://bucket/x.json", ValueError),
        ("https://example.org/x.json?v=1", _store.StoreError),
        ("https://example.org/", _store.StoreError),
        ("https://example.org/a//b.json", _store.StoreError),
    ],
)
def test_urls_obstore_cannot_request_are_refused(url: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        _store.get(url)


def test_head_size_reads_content_length(https_server: HttpsServer) -> None:
    https_server.routes["/x.bin"] = Route(body=b"abcdef")

    assert _store.head_size(https_server.url("x.bin")) == 6
    assert [s.method for s in https_server.seen] == ["HEAD"]
    assert https_server.seen[0].headers["user-agent"] == _AGENT


def test_head_size_reports_the_status(https_server: HttpsServer) -> None:
    with pytest.raises(_store.StoreError) as raised:
        _store.head_size(https_server.url("missing.bin"))
    assert raised.value.status == 404


def test_get_range_reads_a_slice(https_server: HttpsServer) -> None:
    https_server.routes["/x.bin"] = Route(body=bytes(range(64)))

    assert _store.get_range(https_server.url("x.bin"), 8, 12) == bytes(range(8, 12))
    assert https_server.seen[0].headers["range"] == "bytes=8-11"


def test_get_range_slices_when_the_server_ignores_range(https_server: HttpsServer) -> None:
    https_server.routes["/x.bin"] = Route(body=bytes(range(64)), honor_range=False)

    assert _store.get_range(https_server.url("x.bin"), 8, 12) == bytes(range(8, 12))


def test_stream_yields_the_whole_body_lazily(https_server: HttpsServer) -> None:
    payload = bytes(range(256)) * 1000
    https_server.routes["/x.bin"] = Route(body=payload)

    chunks = _store.stream(https_server.url("x.bin"), 1 << 16)
    assert https_server.seen == []  # nothing sent before iteration
    assert b"".join(chunks) == payload


def test_stream_failure_is_an_os_error(https_server: HttpsServer) -> None:
    with pytest.raises(OSError, match="returned 404"):
        b"".join(_store.stream(https_server.url("missing.bin"), 1 << 16))


def test_one_store_per_host(https_server: HttpsServer) -> None:
    https_server.routes["/a.json"] = Route(body=b"{}")
    https_server.routes["/b/c.json"] = Route(body=b"{}")

    _store.get(https_server.url("a.json"))
    _store.get(https_server.url("b/c.json"))
    assert _store._store.cache_info().currsize == 1


def test_only_the_live_prober_imports_urllib_request() -> None:
    """Every read goes through obstore except the live pass's per-host probes.

    Those probes read CORS and range headers, and send OPTIONS, which obstore
    does not support. Any other module that reaches for urllib.request bypasses
    the connection pool and the shared error handling here.
    """
    import ast
    from pathlib import Path

    import rashid

    importers = []
    for path in sorted(Path(rashid.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.startswith("urllib.request") for name in names):
                importers.append(path.name)
    assert importers == ["live.py"]
