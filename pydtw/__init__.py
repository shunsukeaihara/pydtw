# -*- coding: utf-8 -*-
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pydtw")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "unknown"

from .dtw import (
    dtw1d, dtw2d,
    dtw1d_dist, dtw2d_dist,
    pruned_dtw1d, pruned_dtw2d,
    constrained_dtw1d, constrained_dtw2d,
)

__all__ = [
    "__version__",
    "dtw1d", "dtw2d",
    "dtw1d_dist", "dtw2d_dist",
    "pruned_dtw1d", "pruned_dtw2d",
    "constrained_dtw1d", "constrained_dtw2d",
]
