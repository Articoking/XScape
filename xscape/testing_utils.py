"""Plotting functions."""

from typing import List

import xarray as xr
import pandas as pd
import numpy as np
import copernicusmarine as cmems
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

import xscape.utils as utils

def random_datetime64_generator(
    n_datetimes: int,
    start_date: np.datetime64,
    end_date: np.datetime64,
    ) -> np.ndarray:
    """
    Generates an array of random datetime64 values within a given range.

    Parameters
    ----------
    n_datetimes : int
        Number of random datetime64 values to generate.
    start_date : np.datetime64
        The earliest possible datetime.
    end_date : np.datetime64
        The latest possible datetime.

    Returns
    -------
    np.ndarray
        A 1D array of random datetime64 values.
    """
    # Convert datetime64 to integers (seconds since epoch)
    start_int = start_date.astype('datetime64[s]').astype(np.int64)
    end_int = end_date.astype('datetime64[s]').astype(np.int64)
    
    # Generate random integers in the given range
    random_ints = np.random.randint(start_int, end_int, size=n_datetimes)
    
    # Convert back to datetime64
    return random_ints.astype('datetime64[s]')
    

def generate_points(
    n_points: int,
    lon_range: tuple,
    lat_range: tuple,
    time_range: tuple | None = None,
    ) -> pd.DataFrame:
    """
    Randomly generates a series of points.

    Parameters
    ----------
    n_points : int
        The number of points to generate.
    lon_range, lat_range : tuple
        Lat. and lon. ranges defining the area in which to generate points.
    time_range : tuple of np.datetime64, optional
        Range of times to generate timestamps.

    Returns
    -------
    points : pd.DataFrame
        A pandas DataFrame object with "lat" and "lon" columns containing the
        points as rows. If `time_range` is provided, contains an additional
        "time" column.
    """
    lat_limit = 90 # Only allow latitudes in [-90, 90]
    lon_limit = 180 # Only allow longitudes in [-180, 180]

    min_lon, max_lon = lon_range
    min_lat, max_lat = lat_range

    # See issue #10
    if min_lat > max_lat:
        lat_range = abs(lat_limit - min_lat) + abs(max_lat - lat_limit)
        rel_lats = np.random.uniform(0, lat_range, size=(n_points,))
        lats = np.where(
            rel_lats <= max_lat,
            rel_lats - lat_limit,
            rel_lats + min_lat
        )
    else:
        lats = np.random.uniform(min_lat, max_lat, size=(n_points,))

    if min_lon > max_lon:
        lon_range = abs(lon_limit - min_lon) + abs(max_lon - lon_limit)
        rel_lons = np.random.uniform(0, lon_range, size=(n_points,))
        lons = np.where(
            rel_lons <= max_lon,
            rel_lons - lon_limit,
            rel_lons + min_lon
        )
    else:
        lons = np.random.uniform(min_lon, max_lon, size=(n_points,))
    
    points = pd.DataFrame({
        'lat': lats,
        'lon': lons
    })

    if time_range is not None:
        min_time, max_time = time_range
        points["time"] = random_datetime64_generator(
            n_points,
            min_time, 
            max_time
            )
    return points

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

GLORYS_GRIDSIZE = 1/12

def get_glorys_ds(
    points: pd.DataFrame,
    seascape_size: float,
    variables: List[str],
    start_datetime: str,
    end_datetime: str,
    ) -> xr.Dataset:
    """
    Gets GLORYS data for the specified region/time.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    seascape_size : float
        Size (in degrees) of the seascape around each point.
    variables : list of str
        GLORYS variable names to include in the returned data.
    start_datetime : str
        Earliest date for which to get data.
    end_datetime : str
        Latest date for which to get data.

    Returns
    -------
    xr.Dataset
        Dataset in the same format as that returned by `copernicusmarine`.
    """

    gridsize = GLORYS_GRIDSIZE
    extent = utils.get_request_extent(
        points,
        seascape_size,
        gridsize
        )

    data_request = {
    "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "variables": variables,
    "start_datetime": start_datetime,
    "end_datetime" : end_datetime,
    "maximum_latitude": extent["maximum_latitude"],
    "minimum_latitude": extent["minimum_latitude"],
    "maximum_longitude": extent["maximum_longitude"],
    "minimum_longitude": extent["minimum_longitude"],
    }
    glorys_da = cmems.open_dataset(**data_request)\
        .rename({
            "latitude": "lat",
            "longitude": "lon"
        })
    return glorys_da

def get_glorys_var(
    points: pd.DataFrame,
    seascape_size: float,
    variable: str,
    start_datetime: str,
    end_datetime: str,
    ) -> xr.Dataset:
    """
    Gets GLORYS data *for a single variable* for the specified region/time.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    seascape_size : float
        Size (in degrees) of the seascape around each point.
    variables : list of str
        GLORYS variable names to include in the returned data.
    start_datetime : str
        Earliest date for which to get data.
    end_datetime : str
        Latest date for which to get data.

    Returns
    -------
    xr.Dataset
        Dataset in the same format as that returned by `copernicusmarine`.
    """
    glorys_da = get_glorys_ds(
        points = points,
        seascape_size = seascape_size,
        variables = [variable],
        start_datetime = start_datetime,
        end_datetime = end_datetime
    )[variable]

    return glorys_da