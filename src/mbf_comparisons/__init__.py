# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .maxquant_parser import (  # noqa:F401
    TableBasedDataset,
    MaxQuantDataset,
    Normalization_Raw,
    Normalization_MedianCenter,
    Normalization_MedianZero,
    Normalization_Quantile,
    Normalization_InverseNormalTransformation,
    Zero_Ignore,
    Zero_NaN,
    Zero_Filter,
    Zero_NaNIfOtherMeasured,
    Contrast_TTest,
    Contrast_TTestPaired,
    Contrast_Limma,
    Contrast_LimmaPaired,
    Load_AllFeatureTable,
)

from .annotator import Comparison
from .comparisons import Log2FC, TTest, TTestPaired, EdgeRUnpaired, DESeq2Unpaired


def do_export(name):
    if "_" in name:
        start = name
        start = start[: start.find("_")]
        return start in ("Load", "Contrast", "Zero", "Normalization")
    return False


__all__ = (
    ["TableBasedDataset", "MaxQuantDataset"]
    + [key for (key, value) in globals().items() if do_export(key)]
    + [Comparison, Log2FC, TTest, TTestPaired, EdgeRUnpaired, DESeq2Unpaired]
)
