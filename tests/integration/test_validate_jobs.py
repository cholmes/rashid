"""``jobs`` changes how many processes run, never what the report says."""

from __future__ import annotations

from typing import Any

import pytest

from rashid import validate
from tests.conftest import CatalogBuilder, mutate_json

pytestmark = pytest.mark.integration

PROJECTION_URI = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"


def _break_item(item: dict[str, Any]) -> None:
    item["bbox"] = ["west", "south"]  # STR-001
    item.setdefault("stac_extensions", []).append(PROJECTION_URI)
    item["properties"]["proj:code"] = 4326  # EXT-001: proj:code is a string


def test_reports_match_for_one_and_several_processes(catalog: CatalogBuilder) -> None:
    for name in ("roads", "rivers", "rails"):
        collection = catalog.collection(name)
        for year in (2023, 2024, 2025):
            collection.item(f"{name}-{year}")
    root = catalog.write()
    mutate_json(root / "roads" / "roads-2024" / "roads-2024.json", _break_item)
    mutate_json(root / "rivers" / "collection.json", lambda d: d.pop("title"))  # SCH-001

    reports = [
        validate(root, data=False, schema=True, schema_allow_network=False, jobs=jobs)
        for jobs in (1, 2, 4)
    ]

    rule_ids = {f.rule_id for f in reports[0].findings}
    assert {"PTL-STR-001", "PTL-EXT-001", "PTL-SCH-001"} <= rule_ids
    assert reports[1].to_dict() == reports[0].to_dict()
    assert reports[2].to_dict() == reports[0].to_dict()


def test_an_injected_validator_runs_in_this_process(catalog: CatalogBuilder) -> None:
    """A closure cannot be pickled, so jobs must not try to send it to a worker."""
    catalog.collection("roads").item("roads-2024")
    root = catalog.write()
    seen: list[str] = []

    def record(data: dict[str, Any]) -> list[Any]:
        seen.append(str(data.get("id")))
        return []

    validate(root, data=False, structural_validator=record, jobs=2)
    assert sorted(seen) == ["roads", "roads-2024", "root"]
