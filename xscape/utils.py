"""Utility functions for XScape."""

import numpy as np
import pandas as pd
import xarray as xr

VERTICAL_DIM_NAMES = [
    "depth",
    "DEPTH",
    "Depth",
    "height",
    "Height",
    "HEIGHT",
    # TODO: Add any other standard names
    ]

VERTICAL_COORD_NAMES = VERTICAL_DIM_NAMES

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

def get_gridcenter_time(
    dates: pd.DataFrame,
    var_da: xr.DataArray,
    ) -> pd.DataFrame:
    """
    Gets the nearest time coordinates in a grid for a series of datetime values.

    Parameters
    ----------
    dates :pd.DataFrame
        DataFrame with a "time" column.
    var_da : xr.DataArray
        Gridded data array with a "time" dimension.

    Returns
    -------
    pd.DataFrame
        A DataFrame equivalent to `dates` with the "time" column replaced with
        the closest available time in `var_da` for each datetime in `dates`.
        Any duplicate rows after the operation are dropped.
    
    Raises
    ------
    AttributeError
        If `dates` has no "time" column or if `var_da` has no "time" coordinate.
    """
    
    # Function to find the nearest time
    def find_nearest_time(value, time_grid):
        return time_grid[np.abs(time_grid - np.datetime64(value)).argmin()]
    
    if "time" not in var_da.coords:
        raise AttributeError("No time coordinate found in background DataArray.")
    if "time" not in dates.columns:
        raise AttributeError("No time column found in the provided DataFrame.")

    gridded_dates = dates.copy()
    time_values = var_da["time"].values
    
    gridded_dates["time"] = dates["time"].apply(lambda x: find_nearest_time(x, time_values))
    
    return gridded_dates.drop_duplicates()

def get_vert_dimname(
    var_da: xr.DataArray,
    ) -> str | None:
    """
    Gets the name of the dimension representing the vertical dimension.

    Parameters
    ----------
    var_da : xr.DataArray
        Gridded data array.

    Returns
    -------
    str | None
        Name of the dimension if it exists.
    """
    list_dims = list(var_da.dims)
    dim_set = set(VERTICAL_DIM_NAMES).intersection(list_dims)
    vert_dimname = None if len(dim_set) == 0 else next(iter(dim_set))
    return vert_dimname

def get_vert_coordname(
    var_da: xr.DataArray,
    ) -> str | None:
    """
    Gets the name of the coordinate representing the vertical dimension.

    Parameters
    ----------
    var_da : xr.DataArray
        Gridded data array.

    Returns
    -------
    str | None
        Name of the dimension if it exists.
    """
    list_coords = list(var_da.coords)
    coord_set = set(VERTICAL_COORD_NAMES).intersection(list_coords)
    vert_dimname = None if len(coord_set) == 0 else next(iter(coord_set))
    return vert_dimname

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

    lat_coord = "ss_rlat" if "ss_lat" in var_da.dims else "lat"
    lon_coord = "ss_rlon" if "ss_lon" in var_da.dims else "lon"

    lat_gridsize = np.diff(var_da[lat_coord].values).mean()
    lon_gridsize = np.diff(var_da[lon_coord].values).mean()
    # TODO (#2): Allow different sizes in lat and lon
    gridsize = (lat_gridsize + lon_gridsize) / 2
    return gridsize

def calculate_timestep_duration(
    var_da: xr.DataArray,
    ) -> np.timedelta64:
    """
    Calculates the duration of each timestep in `var_da`.

    Parameters
    ----------
    var_da : xr.DataArray
        DataArray with a regularly spaced "time" coordinate.

    Returns
    -------
    np.timedelta64
        Average timedelta between each timestamp in `var_da`'s "time" coordinate.
    """
    time_coord = "ss_rtime" if "ss_time" in var_da.dims else "time"
    ts_duration = np.diff(var_da[time_coord].values).mean()
    return ts_duration

def create_empty_seascape(
    ss_rlon_vals: np.ndarray,
    ss_rlat_vals: np.ndarray,
    ss_rtime_vals: np.ndarray | None = None
    ) -> xr.DataArray:
    """
    Creates an empty seascape according to prescribed relative coordinates.

    Parameters
    ----------
    ss_rlon_vals , ss_rlat_vals, ss_rtime_vals : np.ndarray
        Relative grid values.

    Returns
    -------
    xr.DataArray
        Seascape-like DataArray filled with NaN values.
    """
    data = np.full((len(ss_rlon_vals), len(ss_rlat_vals)), np.nan)
    coords = {
        "lon": ss_rlon_vals,
        "lat": ss_rlat_vals,
    }
    dims = ["lon", "lat"]
    if ss_rtime_vals is not None:
        data = np.tile(
            np.expand_dims(data, axis=-1),
            (1, 1, len(ss_rtime_vals))\
            )
        coords["time"] = ss_rtime_vals
        dims.append("time")

    seascape = xr.DataArray(
        data = data,
        coords = coords,
        dims = dims,
    )
    return seascape