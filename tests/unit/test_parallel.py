"""The process pool returns, and raises, what one process would."""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from rashid._parallel import precomputed
from rashid.cli import main
from rashid.runner import validate

pytestmark = pytest.mark.unit


class Unpicklable(Exception):
    def __reduce__(self) -> Any:
        raise TypeError("cannot pickle")


def _factory(*, offset: int) -> Any:
    def _validate(data: dict[str, Any], suffix: str = "") -> Any:
        if data.get("fail") == "lookup":
            raise LookupError(f"no schema for {data['n']}")
        if data.get("fail") == "unpicklable":
            raise Unpicklable(f"broken {data['n']}")
        return f"{data['n'] + offset}{suffix}"

    return _validate


def test_results_come_back_for_each_call_in_any_order() -> None:
    objects = [{"n": n} for n in range(50)]
    replay = precomputed(_factory, {"offset": 100}, [(o, "x") for o in objects], jobs=3)
    assert [replay(o, "x") for o in reversed(objects)] == [
        f"{n + 100}x" for n in reversed(range(50))
    ]


def test_an_exception_is_raised_at_the_call_that_raised_it() -> None:
    good, bad = {"n": 1}, {"n": 2, "fail": "lookup"}
    replay = precomputed(_factory, {"offset": 0}, [(good,), (bad,)], jobs=2)
    assert replay(good) == "1"
    with pytest.raises(LookupError, match="no schema for 2"):
        replay(bad)


def test_an_unpicklable_exception_keeps_its_message() -> None:
    bad = {"n": 3, "fail": "unpicklable"}
    replay = precomputed(_factory, {"offset": 0}, [(bad,)], jobs=2)
    with pytest.raises(RuntimeError, match="broken 3"):
        replay(bad)


def test_results_are_keyed_by_object_identity_not_equality() -> None:
    """Two equal dicts are two objects, and a call for one never answers the other."""
    first, twin = {"n": 1}, {"n": 1}
    replay = precomputed(_factory, {"offset": 0}, [(first,)], jobs=2)
    assert replay(first) == "1"
    with pytest.raises(KeyError):
        replay(twin)


def test_validate_refuses_jobs_below_one(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="jobs must be at least 1"):
        validate(tmp_path, jobs=0)


def test_cli_refuses_jobs_below_one(tmp_path: Any) -> None:
    result = CliRunner().invoke(main, ["check", "--jobs", "0", str(tmp_path)])
    assert result.exit_code == 2
    assert "--jobs" in result.output
