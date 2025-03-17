"""Automatically get the seascape around a set of points."""

import xscape.accessors
from .core import get_glorys_ds, create_xscp_da
from .utils import generate_points
from .plotting import plotting

__all__ = [
    "get_glorys_ds",
    "create_xscp_da",
    "generate_points",
    "plotting",
    ]