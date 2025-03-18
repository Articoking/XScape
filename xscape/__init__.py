"""Automatically get the seascape around a set of points."""

from . import accessors
from .core import get_glorys_ds, get_glorys_var, create_xscp_da
from .utils import generate_points
from . import plotting

__all__ = [
    "get_glorys_ds",
    "get_glorys_var",
    "create_xscp_da",
    "generate_points",
    "plotting",
    "accessors"
    ]