"""Version metadata and Julia-pin consistency tests."""

import json
import os
import re
from pathlib import Path

import pytest

import quanestimation
from quanestimation.base.Common import Common

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.4.0"
JULIA_VERSION = "0.4.0"


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def test_version_metadata() -> None:
    """The package and its version files agree."""
    assert quanestimation.__version__ == VERSION
    assert Common.JULIA_PACKAGE_VERSION == JULIA_VERSION

    assert re.search(r'^version = "%s"$' % re.escape(VERSION), _read("pyproject.toml"), re.M)
    assert re.search(r'version="%s"' % re.escape(VERSION), _read("setup.py"))
    assert re.search(r"^version = %s$" % re.escape(VERSION), _read("setup.cfg"), re.M)


def test_julia_package_pin() -> None:
    """The Julia core pin is exposed by the bridge and, when configured,
    by juliapkg as well."""
    assert Common.JULIA_PACKAGE_VERSION == JULIA_VERSION
    for name in ("juliapkg.json", "juliacalldeps.json"):
        cfg = json.loads(_read(name))
        assert cfg["julia"] == "^1.11"
        if "QuanEstimation" in cfg.get("packages", {}):
            assert cfg["packages"]["QuanEstimation"].startswith("0.4")


@pytest.mark.skipif(
    not os.environ.get("QuanEstimation_DEV_PATH"),
    reason="QuanEstimation_DEV_PATH is not set",
)
def test_julia_bridge_loads_and_caches() -> None:
    """The bridge loads the pinned core and is cached."""
    first = Common.load_julia()
    second = Common.load_julia()
    assert first is second
