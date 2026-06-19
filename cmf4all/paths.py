"""Package path helpers."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = PACKAGE_ROOT / "cmf_zoo"
DEFAULT_METADATA_FILE = DEFAULT_PROJECT_ROOT / "data" / "metadata" / "surveys.yaml"
