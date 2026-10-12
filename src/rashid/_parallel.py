"""Run a pass's per-object validator across processes.

The structural, extension, and schema passes call one validator per object,
and no call reads another object. :func:`precomputed` runs those calls in a
process pool before the pass's own loop starts, then hands the loop a
validator that answers from the results. The loop is unchanged, so the
findings, their order, and the pass's error handling are the same as with
one process.

A worker builds its validator once, from a module-level factory and its
arguments, because a compiled validator cannot be pickled. A validator
injected by a caller is a closure, so a pass runs it in one process.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# The validator a worker process built in _init_worker.
_worker_validator: Callable[..., Any] | None = None


def _init_worker(factory: Callable[..., Callable[..., Any]], kwargs: dict[str, Any]) -> None:
    global _worker_validator
    _worker_validator = factory(**kwargs)


def _call_in_worker(args: tuple[Any, ...]) -> tuple[bool, Any]:
    """One validator call: ``(True, result)`` or ``(False, exception)``.

    The exception comes back as a value so the pass can raise it at the same
    point in its loop as with one process. An exception that cannot be pickled
    becomes a ``RuntimeError`` with the same message, which is all a pass
    reports.
    """
    assert _worker_validator is not None  # noqa: S101  # nosec B101 - set by _init_worker
    try:
        return True, _worker_validator(*args)
    except Exception as exc:  # noqa: BLE001 - the pass decides what each failure means
        try:
            pickle.dumps(exc)
        except Exception:  # noqa: BLE001 - any pickling failure
            exc = RuntimeError(str(exc))
        return False, exc


def _key(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (id(args[0]),) + args[1:]


def precomputed(
    factory: Callable[..., Callable[..., Any]],
    factory_kwargs: dict[str, Any],
    calls: Sequence[tuple[Any, ...]],
    jobs: int,
) -> Callable[..., Any]:
    """Run ``factory(**factory_kwargs)(*call)`` for each call on ``jobs`` processes.

    Returns a validator that takes the same arguments and returns, or raises,
    what the worker's call did. Results are keyed by the identity of the first
    argument (the object's JSON dict, which the graph holds for the whole run)
    and the remaining arguments, so the pass can make its calls in any order.
    A call that was not precomputed raises ``KeyError``, which a pass reports
    as a systemic failure.
    """
    chunksize = max(1, len(calls) // (jobs * 8))
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_init_worker, initargs=(factory, factory_kwargs)
    ) as pool:
        outcomes = list(pool.map(_call_in_worker, calls, chunksize=chunksize))
    results = {_key(call): outcome for call, outcome in zip(calls, outcomes, strict=True)}

    def _replay(*args: Any) -> Any:
        ok, value = results[_key(args)]
        if ok:
            return value
        raise value

    return _replay
