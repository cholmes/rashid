"""Read https objects through obstore, and through urllib where obstore cannot.

The crawl, the data reader, the schema fetch, and the live pass's per-asset
HEADs all read objects by URL. obstore sends those requests from Rust. It keeps
one connection pool per host, so a crawl of thousands of documents reuses its
TLS connections, and it releases the GIL while a request is in flight, so a
thread pool runs the requests in parallel.

obstore reports a failure as an exception, and an HTTP status only in the
exception's text. :class:`StoreError` carries the status that text names, so a
caller can tell a 404 from a dead host. Every failure is an :class:`OSError`,
which is what the data pass treats as an unread asset.

obstore cannot send every request a catalog can name, nor read every answer a
server can give:

- It builds the request URL from an object path, so a URL with a query string,
  an encoded ``/`` (``%2F``), or an empty, ``.`` or ``..`` segment has no
  object path that requests the same URL.
- It refuses a response with no ``Content-Length``, which a chunked HTTP/1.1
  answer or an HTTP/2 answer may omit.

Those requests go out over urllib instead, with the same agent, the same
https-only redirect policy, the same retries of a 5xx or a failed connection,
and the same errors, so the transport is invisible to callers. One difference
stays: obstore also retries a body that breaks off part-way, and urllib does
not.

A :class:`StoreError` with ``refused`` set means the server answered but the
answer was not one rashid accepts: a redirect off https, or a HEAD with no
``Content-Length``. urllib cannot do better with either, so neither is sent
again here; the live prober, which reads such an answer anyway, sends it again
itself.

Timeouts: ``read_timeout`` bounds each wait for bytes, as urllib's socket
timeout does, so a stalled server fails in 30 s while a large asset that keeps
arriving downloads in full. obstore also requires an overall ``timeout``; it is
set far past any download a check makes, so only a server that trickles bytes
forever reaches it.
"""

from __future__ import annotations

import re
import ssl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from functools import cache
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    OpenerDirector,
    Request,
    build_opener,
)

import obstore
from obstore.exceptions import InvalidPathError, NotSupportedError
from obstore.store import HTTPStore

from rashid._http import user_agent

# Longest wait for the next bytes of a response, in either transport.
_IDLE = timedelta(seconds=30)

# obstore's errors name the status as "status code: 404 Not Found".
_STATUS = re.compile(r"status code: (\d{3})\b")

# obstore's text for a response with no Content-Length, and for a redirect
# it would not follow.
_NO_LENGTH = "MissingContentLength"
_REDIRECT = "error following redirect"

_ENCODED_SLASH = re.compile(r"%2f", re.IGNORECASE)

#: Client options for every store. Tests add ``allow_invalid_certificates``
#: here to reach a local server with a self-signed certificate; the urllib
#: transport reads the same key.
CLIENT_OPTIONS: dict[str, Any] = {"read_timeout": _IDLE, "timeout": timedelta(hours=24)}

# Retry a 5xx or a dropped connection up to 3 times, for up to 30 s after the
# first attempt. A host that throttles a parallel crawl answers 503 now and then.
# The urllib transport reads the same values.
_RETRY: dict[str, Any] = {"max_retries": 3, "retry_timeout": _IDLE}

# The urllib transport's first wait before a retry; each retry doubles it.
_BACKOFF = 0.1  # seconds


class StoreError(OSError):
    """A request that failed.

    ``status`` is the HTTP status, when one came back. ``refused`` is set when
    the server answered but rashid did not accept the answer: a redirect off
    https, or a HEAD with no ``Content-Length``.
    """

    def __init__(self, message: str, status: int | None = None, *, refused: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.refused = refused


@cache
def _store(origin: str) -> HTTPStore:
    """One store, and so one connection pool, per ``https://host``."""
    options = {"user_agent": user_agent(), **CLIENT_OPTIONS}
    return HTTPStore.from_url(origin, client_options=options, retry_config=_RETRY)  # type: ignore[arg-type]


def reset() -> None:
    """Drop every cached client, so the next request reads ``CLIENT_OPTIONS`` again."""
    _store.cache_clear()
    _opener.cache_clear()


def _locate(url: str) -> tuple[HTTPStore, str] | None:
    """The store for ``url``'s host and its object path; None if obstore cannot request it.

    ``urlsplit``, not ``urlparse``: the latter moves a ``;…`` suffix of the last
    segment out of the path, and the object requested would lose it.
    """
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"refusing to fetch non-https URL: {url!r}")
    if parsed.query or _ENCODED_SLASH.search(parsed.path):
        return None
    path = unquote(parsed.path).lstrip("/")
    if not path or any(part in ("", ".", "..") for part in path.split("/")):
        return None
    return _store(f"https://{parsed.netloc.lower()}"), path


def _error(method: str, url: str, exc: Exception) -> StoreError:
    """``exc`` as a :class:`StoreError`, with the status its text names."""
    text = str(exc)
    match = _STATUS.search(text)
    if match is not None:
        return StoreError(f"{method} {url} returned {match[1]}", int(match[1]))
    if isinstance(exc, FileNotFoundError):
        return StoreError(f"{method} {url} returned 404", 404)
    reason = text.splitlines()[0] if text else type(exc).__name__
    refused = _NO_LENGTH in text or _REDIRECT in text
    return StoreError(f"{method} {url}: {reason}", refused=refused)


def _urllib_can_read(exc: Exception) -> bool:
    """Whether urllib can send the request or read the answer obstore did not.

    obstore cannot parse some paths, and it refuses a body with no
    ``Content-Length``; urllib handles both. A redirect off https is refused by
    both transports, so asking urllib only repeats the request.
    """
    return isinstance(exc, InvalidPathError) or _NO_LENGTH in str(exc)


# --- the urllib transport ----------------------------------------------------


class _HttpsOnlyRedirects(HTTPRedirectHandler):
    """Follow a redirect only to another https URL, as obstore does."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        if urlsplit(newurl).scheme.lower() != "https":
            raise StoreError(f"redirect to a non-https URL refused: {newurl}", refused=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@cache
def _opener() -> OpenerDirector:
    context = ssl.create_default_context()
    if CLIENT_OPTIONS.get("allow_invalid_certificates"):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return build_opener(_HttpsOnlyRedirects(), HTTPSHandler(context=context))


def _send(url: str, method: str, headers: dict[str, str] | None) -> Any:
    """Open ``url`` over urllib, retrying a 5xx or a failed connection as obstore does."""
    request = Request(url, method=method, headers={"User-Agent": user_agent(), **(headers or {})})
    idle = CLIENT_OPTIONS["read_timeout"].total_seconds()
    deadline = time.monotonic() + _RETRY["retry_timeout"].total_seconds()
    attempt = 0
    cause: Exception
    while True:
        try:
            return _opener().open(request, timeout=idle)
        except HTTPError as exc:
            exc.close()
            error = StoreError(f"{method} {url} returned {exc.code}", exc.code)
            retry, cause = exc.code >= 500, exc
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - urllib raises many types; the message is what matters
            error = StoreError(f"{method} {url}: {exc}")
            retry, cause = True, exc
        delay = _BACKOFF * 2**attempt
        if not retry or attempt >= _RETRY["max_retries"] or time.monotonic() + delay > deadline:
            raise error from cause
        time.sleep(delay)
        attempt += 1


@contextmanager
def _urlopen(url: str, method: str, headers: dict[str, str] | None = None) -> Iterator[Any]:
    """An open urllib response; every failure, including a read's, is a :class:`StoreError`."""
    response = _send(url, method, headers)
    try:
        with response:
            yield response
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - urllib raises many types; the message is what matters
        raise StoreError(f"{method} {url}: {exc}") from exc


def _urllib_get(url: str, max_bytes: int | None) -> bytes:
    with _urlopen(url, "GET") as response:
        body: bytes = response.read() if max_bytes is None else response.read(max_bytes + 1)
    if max_bytes is not None and len(body) > max_bytes:
        raise StoreError(f"GET {url}: body exceeds {max_bytes} bytes")
    return body


# --- reads -------------------------------------------------------------------


def get(url: str, *, max_bytes: int | None = None) -> bytes:
    """The whole body of ``url``. Raises :class:`StoreError` past ``max_bytes``."""
    located = _locate(url)
    if located is None:
        return _urllib_get(url, max_bytes)
    try:
        result = obstore.get(*located)
        if max_bytes is not None and result.meta["size"] > max_bytes:
            raise StoreError(f"GET {url}: body exceeds {max_bytes} bytes")
        return result.bytes().to_bytes()
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
        if not _urllib_can_read(exc):
            raise _error("GET", url, exc) from exc
    return _urllib_get(url, max_bytes)


def head_size(url: str) -> int:
    """The ``Content-Length`` a HEAD of ``url`` answers.

    A HEAD with no ``Content-Length`` has no size for urllib to read either,
    so only a path obstore cannot parse goes out over urllib.
    """
    located = _locate(url)
    if located is not None:
        try:
            return int(obstore.head(*located)["size"])
        except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
            if not isinstance(exc, InvalidPathError):
                raise _error("HEAD", url, exc) from exc
    with _urlopen(url, "HEAD") as response:
        length = response.headers.get("Content-Length")
    if length is None or not length.isdigit():
        raise StoreError(f"HEAD {url} returned no Content-Length", refused=True)
    return int(length)


def get_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``start`` up to, not including, ``end``.

    A server that ignores ``Range`` answers 200 with the whole object, and
    obstore refuses that answer. The object is then read whole and sliced, as
    a client without range support has to.
    """
    located = _locate(url)
    if located is not None:
        try:
            return obstore.get_range(*located, start=start, end=end).to_bytes()
        except NotSupportedError:
            return get(url)[start:end]
        except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
            if not _urllib_can_read(exc):
                raise _error("GET", url, exc) from exc
    with _urlopen(url, "GET", {"Range": f"bytes={start}-{end - 1}"}) as response:
        data: bytes = response.read()
        status = response.status
    return data[start:end] if status == 200 else data


def stream(url: str, chunk_size: int) -> Iterator[bytes]:
    """The body of ``url`` in chunks. No request leaves until iteration starts."""
    located = _locate(url)
    if located is not None:
        try:
            result = obstore.get(*located)
        except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
            if not _urllib_can_read(exc):
                raise _error("GET", url, exc) from exc
        else:
            try:
                for chunk in result.stream(min_chunk_size=chunk_size):
                    yield bytes(chunk)
            except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
                raise _error("GET", url, exc) from exc
            return
    with _urlopen(url, "GET") as response:
        while chunk := response.read(chunk_size):
            yield chunk
