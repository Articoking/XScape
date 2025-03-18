"""Utility functions for XScape."""

import numpy as np
import pandas as pd
import xarray as xr

def generate_points(
    n_points: int,
    lon_range: tuple,
    lat_range: tuple
    ) -> pd.DataFrame:
    """
    Randomly generates a series of points.

    Parameters
    ----------
    n_points : int
        The number of points to generate.
    lon_range, lat_range : tuple
        Lat. and lon. ranges defining the area in which to generate points.

    Returns
    -------
    points : pd.DataFrame
        A pandas DataFrame object with "lat" and "lon" columns containing the
        points as rows.
    """
    min_lon, max_lon = lon_range
    min_lat, max_lat = lat_range

    # See issue #10
    if min_lon > max_lon or min_lat > max_lat:
        raise NotImplementedError(
            "Reverse lat/lon ranges are not yet allowed."
        )

    lats = np.random.uniform(min_lat, max_lat, size=(n_points,))
    lons = np.random.uniform(min_lon, max_lon, size=(n_points,))
    points = pd.DataFrame({
        'lat': lats,
        'lon': lons
    })
    return points

def get_request_extent(
    points: pd.DataFrame,
    seascape_size: float,
    gridsize: float
    ) -> dict:
    """
    Calculates the area needed to cover all points and their seascapes.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    seascape_size : float
        Size (in degrees) of the seascape around each point.
    gridsize : float
        Size (in degrees) of each pixel in the original background field.

    Returns
    -------
    dict
        `copernicusmarine`-style dictionary of max/min lat/lon.

    See Also
    --------
    generate_points
    """

    if seascape_size < 0:
        raise ValueError("seascape_size cannot be negative.")
    
    # Sizes in degrees
    return {
    'maximum_latitude': points['lat'].max() + gridsize + seascape_size/2,
    'minimum_latitude': points['lat'].min() - gridsize - seascape_size/2,
    'maximum_longitude': points['lon'].max() + gridsize + seascape_size/2,
    'minimum_longitude': points['lon'].min() - gridsize - seascape_size/2,
    }

def get_gridcenter_points(
    points: pd.DataFrame, 
    var_da: xr.DataArray,
    ) -> pd.DataFrame:
    """
    Gets the corresponding pixel coordinates for a series of points.

    Returns a DataFrame with points as rows, which correspond to the coordinates of the
    pixels of `var_da` in which each point in `points` is.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    var_da : xr.DataArray
        Gridded background field on whose grid to project the points.

    Returns
    -------
    pd.DataFrame
        A DataFrame in the same format as `points` with the center coordinates
        of pixels in `var_da`
    """

    # Function to find the nearest grid point
    def find_nearest(value, grid):
        return grid[np.abs(grid - value).argmin()]

    c_points = points.copy()
    c_points['lat'] = points['lat'].apply(lambda x: find_nearest(x, var_da['lat'].values))
    c_points['lon'] = points['lon'].apply(lambda x: find_nearest(x, var_da['lon'].values))
    return c_points.drop_duplicates()

def calculate_horizontal_gridsize(
    var_da: xr.DataArray,
    ) -> float:
    """
    Calculates the horizontal pixel size of a gridded DataArray.

    Automatically calculates the mean of the difference between gridpoints for
    both lat and lon and then averages those two values.

    Parameters
    ----------
    var_da : xr.DataArray
        Data array gridded in "lat" and "lon" coordinates. Coordinates must be
        in degrees.

    Returns
    -------
    float
        Calculated gridsize (in degrees)
    """

    lat_coord = "ss_lat" if "ss_lat" in var_da.coords else "lat"
    lon_coord = "ss_lon" if "ss_lon" in var_da.coords else "lon"

    lat_gridsize = np.diff(var_da[lat_coord].values).mean()
    lon_gridsize = np.diff(var_da[lon_coord].values).mean()
    # TODO (#2): Allow different sizes in lat and lon
    gridsize = (lat_gridsize + lon_gridsize) / 2
    return gridsize

