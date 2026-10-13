import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlamanip.config import load_config  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def grid(cfg):
    return WorkspaceGrid.from_config(cfg.workspace)
