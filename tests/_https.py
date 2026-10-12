"""A local HTTPS server for the tests that drive obstore end to end.

obstore sends its requests from Rust, so a monkeypatched ``urlopen`` cannot
stand in for the network. This server answers over real TLS with a
certificate from ``trustme``, and ``rashid._store`` is told to accept it. Each
test sets the routes it needs and reads back the requests the server saw.
"""

from __future__ import annotations

import ssl
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import trustme

from rashid import _store


@dataclass
class Route:
    """One canned answer. ``length=False`` sends no ``Content-Length``.

    ``delay`` is a pause, in seconds, before each byte of the body after the
    first, so a test can serve a slow body or one that stalls. The first
    ``fail_first`` requests answer 503 instead, as a throttling host does.
    """

    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    honor_range: bool = True
    length: bool = True
    delay: float = 0.0
    fail_first: int = 0


@dataclass
class Seen:
    method: str
    path: str
    headers: dict[str, str]


class HttpsServer:
    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.seen: list[Seen] = []
        self.base = ""

    def url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"


def _handler(server: HttpsServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return None

        def _answer(self, send_body: bool) -> None:
            server.seen.append(
                Seen(self.command, self.path, {k.lower(): v for k, v in self.headers.items()})
            )
            route = server.routes.get(self.path, Route(status=404))
            body, status = route.body, route.status
            if route.fail_first:
                route.fail_first -= 1
                body, status = b"", 503
            headers = dict(route.headers)
            byte_range = self.headers.get("Range")
            if byte_range and route.honor_range and status == 200:
                first, last = byte_range.removeprefix("bytes=").split("-")
                start, end = int(first), min(int(last), len(body) - 1)
                headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
                body, status = body[start : end + 1], 206
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            if route.length:
                self.send_header("Content-Length", str(len(body)))
            else:
                self.send_header("Connection", "close")
            self.end_headers()
            if not send_body:
                return
            if not route.delay:
                self.wfile.write(body)
                return
            for index in range(len(body)):
                if index:
                    time.sleep(route.delay)
                self.wfile.write(body[index : index + 1])
                self.wfile.flush()

        def do_GET(self) -> None:
            self._answer(True)

        def do_HEAD(self) -> None:
            self._answer(False)

    return Handler


@pytest.fixture
def https_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[HttpsServer]:
    """A running local HTTPS server that ``rashid._store`` trusts."""
    authority = trustme.CA()
    certificate = authority.issue_cert("127.0.0.1", "localhost")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    certificate.configure_cert(context)
    state = HttpsServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    state.base = f"https://127.0.0.1:{httpd.server_address[1]}"
    # A short poll interval: shutdown() waits up to one interval, once per test.
    thread = threading.Thread(target=httpd.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    monkeypatch.setitem(_store.CLIENT_OPTIONS, "allow_invalid_certificates", True)
    _store.reset()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        _store.reset()
