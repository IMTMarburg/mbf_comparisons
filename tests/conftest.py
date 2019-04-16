#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import pytest
import sys
from pathlib import Path
from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,  # noqa:F401
    both_ppg_and_no_ppg,
    pytest_runtest_makereport,  # noqa:F401
)  # noqa:F401


def pytest_generate_tests(metafunc):
    if "both_ppg_and_no_ppg" in metafunc.fixturenames:
        metafunc.parametrize("both_ppg_and_no_ppg", [True, False], indirect=True)


root = Path(__file__).parent.parent
sys.path.append(str(root / "src"))
