"""Version metadata and Julia-pin consistency tests."""

import json
import re
from importlib.resources import files
from pathlib import Path

import quanestimation
from quanestimation.base.Common import Common

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.5.0"


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def test_version_metadata() -> None:
    """The package and its version files agree."""
    assert quanestimation.__version__ == VERSION
    assert Common.JULIA_PACKAGE_VERSION == VERSION

    assert re.search(r'^version = "%s"$' % re.escape(VERSION), _read("pyproject.toml"), re.M)
    assert re.search(r'version="%s"' % re.escape(VERSION), _read("setup.py"))
    assert re.search(r"^version = %s$" % re.escape(VERSION), _read("setup.cfg"), re.M)


def test_julia_package_pin() -> None:
    """The installed package exposes matching bridge and juliapkg pins."""
    assert Common.JULIA_PACKAGE_VERSION == VERSION
    resource = files("quanestimation").joinpath("juliapkg.json")
    cfg = json.loads(resource.read_text(encoding="utf-8"))
    assert cfg["julia"] == "^1.11"
    assert cfg["packages"]["QuanEstimation"] == {
        "uuid": "088c8dff-a786-4a66-974c-03d3f6773f87",
        "version": "0.5",
    }


def test_julia_bridge_loads_and_caches() -> None:
    """The bridge loads the pinned core and is cached."""
    first = Common.load_julia()
    second = Common.load_julia()
    assert first is second
