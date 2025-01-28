import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

import pandas as pd

import xscape

def plot_points(points: pd.DataFrame, ax:GeoAxes = None) -> None:
    if ax is None:
        ax = plt.gca()
    
    ax.coastlines()
    ax.scatter(points['lon'], points['lat'], transform=ccrs.PlateCarree(), marker='x', c='black')