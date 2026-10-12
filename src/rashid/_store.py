"""Read https objects through obstore.

The crawl, the data reader, the schema fetch, and the live pass's per-asset
HEADs all read objects by URL. obstore sends those requests from Rust. It keeps
one connection pool per host, so a crawl of thousands of documents reuses its
TLS connections, and it releases the GIL while a request is in flight, so a
thread pool runs the requests in parallel.

obstore reports a failure as an exception, and an HTTP status only in the
exception's text. :class:`StoreError` carries the status that text names, so a
caller can tell a 404 from a dead host. Every failure is an :class:`OSError`,
which is what the data pass treats as an unread asset.

obstore builds the request URL from an object path. It encodes ``%`` and
``?`` itself, so :func:`_locate` decodes the URL's path first and refuses a URL
with a query string, which no object path can carry. obstore also refuses a
response with no ``Content-Length`` and a redirect to a non-https URL. Both
become a :class:`StoreError` with no status.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from functools import cache
from typing import Any
from urllib.parse import unquote, urlparse

import obstore
from obstore.exceptions import NotSupportedError
from obstore.store import HTTPStore

from rashid._http import user_agent

_TIMEOUT = timedelta(seconds=30)

# obstore's errors name the status as "status code: 404 Not Found".
_STATUS = re.compile(r"status code: (\d{3})\b")

#: Client options for every store. Tests add ``allow_invalid_certificates``
#: here to reach a local server with a self-signed certificate.
CLIENT_OPTIONS: dict[str, Any] = {"timeout": _TIMEOUT}

# Retry a 5xx or a dropped connection up to 3 times, within the request
# timeout. A host that throttles a parallel crawl answers 503 now and then.
_RETRY: dict[str, Any] = {"max_retries": 3, "retry_timeout": _TIMEOUT}


class StoreError(OSError):
    """A request that failed. ``status`` is the HTTP status, when one came back."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@cache
def _store(origin: str) -> HTTPStore:
    """One store, and so one connection pool, per ``https://host``."""
    options = {"user_agent": user_agent(), **CLIENT_OPTIONS}
    return HTTPStore.from_url(origin, client_options=options, retry_config=_RETRY)  # type: ignore[arg-type]


def reset() -> None:
    """Drop every cached store, so the next request reads ``CLIENT_OPTIONS`` again."""
    _store.cache_clear()


def _locate(url: str) -> tuple[HTTPStore, str]:
    """The store for ``url``'s host and the object path obstore requests."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"refusing to fetch non-https URL: {url!r}")
    if parsed.query:
        raise StoreError(f"cannot request a URL with a query string: {url}")
    path = unquote(parsed.path).lstrip("/")
    if not path or any(part == "" for part in path.split("/")):
        raise StoreError(f"URL does not name an object: {url}")
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
    return StoreError(f"{method} {url}: {reason}")


def get(url: str, *, max_bytes: int | None = None) -> bytes:
    """The whole body of ``url``. Raises :class:`StoreError` past ``max_bytes``."""
    store, path = _locate(url)
    try:
        result = obstore.get(store, path)
        if max_bytes is not None and result.meta["size"] > max_bytes:
            raise StoreError(f"GET {url}: body exceeds {max_bytes} bytes")
        return result.bytes().to_bytes()
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
        raise _error("GET", url, exc) from exc


def head_size(url: str) -> int:
    """The ``Content-Length`` a HEAD of ``url`` answers."""
    store, path = _locate(url)
    try:
        return int(obstore.head(store, path)["size"])
    except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
        raise _error("HEAD", url, exc) from exc


def get_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``start`` up to, not including, ``end``.

    A server that ignores ``Range`` answers 200 with the whole object, and
    obstore refuses that answer. The object is then read whole and sliced, as
    a client without range support has to.
    """
    store, path = _locate(url)
    try:
        return obstore.get_range(store, path, start=start, end=end).to_bytes()
    except NotSupportedError:
        return get(url)[start:end]
    except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
        raise _error("GET", url, exc) from exc


def stream(url: str, chunk_size: int) -> Iterator[bytes]:
    """The body of ``url`` in chunks. No request leaves until iteration starts."""
    store, path = _locate(url)
    try:
        result = obstore.get(store, path)
        for chunk in result.stream(min_chunk_size=chunk_size):
            yield bytes(chunk)
    except Exception as exc:  # noqa: BLE001 - obstore raises several types; the text has the status
        raise _error("GET", url, exc) from exc
