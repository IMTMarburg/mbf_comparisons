# -*- coding: utf-8 -*-

__version__ = "0.1"

from .comparisons import Comparisons
from .methods import (
    Log2FC,
    TTest,
    TTestPaired,
    EdgeRUnpaired,
    DESeq2Unpaired,
    EdgeRPaired,
)
from . import venn


__all__ = [
    Comparisons,
    Log2FC,
    TTest,
    TTestPaired,
    EdgeRUnpaired,
    DESeq2Unpaired,
    venn,
    EdgeRPaired,
]
