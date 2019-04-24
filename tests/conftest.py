#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import pytest
import sys
from pathlib import Path
from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,
    both_ppg_and_no_ppg,
    pytest_runtest_makereport,
)
from mbf_qualitycontrol.testing.fixtures import (  # noqa: F401
    new_pipegraph_no_qc,
    both_ppg_and_no_ppg_no_qc,
)


def pytest_generate_tests(metafunc):
    if "both_ppg_and_no_ppg" in metafunc.fixturenames:
        metafunc.parametrize("both_ppg_and_no_ppg", [True, False], indirect=True)


root = Path(__file__).parent.parent
sys.path.append(str(root / "src"))
