"""Pytest configuration for odra_test (pythonpath set in pyproject.toml)."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iqm_token():
    return os.environ.get("IQM_TOKEN", "").strip()


@pytest.fixture
def iqm_token():
    return _iqm_token()
