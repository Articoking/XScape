"""Plotting functions."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

import pandas as pd

def plot_points(
    points: pd.DataFrame,
    ax:GeoAxes = None
    ) -> None:
    """
    Scatterplot of a series of points.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    ax : GeoAxes, optional
        cartopy GeoAxes object on which to plot the points. If none specified,
        uses the currently active matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()
    
    ax.coastlines()
    ax.scatter(
        points['lon'],
        points['lat'],
        transform=ccrs.PlateCarree(),
        marker='x',
        c='black',
    )