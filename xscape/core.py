"""Core functionality of XScape."""

import math
from typing import List

import xarray as xr
import pandas as pd
import numpy as np
import copernicusmarine as cmems
from tqdm import tqdm

GLORYS_GRIDSIZE = 1/12

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
    # Sizes in degrees
    return {
    'maximum_latitude': points['lat'].max() + gridsize + seascape_size/2,
    'minimum_latitude': points['lat'].min() - gridsize - seascape_size/2,
    'maximum_longitude': points['lon'].max() + gridsize + seascape_size/2,
    'minimum_longitude': points['lon'].min() - gridsize - seascape_size/2,
    }

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
    extent = get_request_extent(
        points,
        seascape_size,
        gridsize
        )

    data_request = {
    'dataset_id': 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
    'variables': variables,
    'start_datetime': start_datetime,
    'end_datetime' : end_datetime,
    'maximum_latitude': extent['maximum_latitude'],
    'minimum_latitude': extent['minimum_latitude'],
    'maximum_longitude': extent['maximum_longitude'],
    'minimum_longitude': extent['minimum_longitude'],
    }
    glorys_da = cmems.open_dataset(**data_request)\
        .rename({
            'latitude': 'lat',
            'longitude': 'lon'
        })
    return glorys_da 

def get_gridcenter_points(
    points: pd.DataFrame, 
    var_da:xr.DataArray
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


def create_xscp_da(
    points: pd.DataFrame,
    seascape_size: float,
    var_da:xr.DataArray,
    ) -> xr.DataArray:
    """
    Crops and packages together a series of seascapes.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns.
    seascape_size : float
        Size (in degrees) of the seascape around each point.
    var_da : xr.DataArray
        Gridded background field from which we extract the seascapes.

    Returns
    -------
    xr.DataArray
        A DataArray indexed by `seascape_idx`, `ss_lon` and `ss_lat`. The latter
        two coordinates correspond to a relative reference frame centered on
        each seascape.

    Raises
    ------
    NotImplementedError
        If a seascape is completely empty (e.g. if a point is on land).
    """

    # Calculate gridsize
    lat_gridsize = np.diff(var_da.lat.values).mean()
    lon_gridsize = np.diff(var_da.lon.values).mean()
    # TODO: Allow different sizes in lat and lon
    gridsize = (lat_gridsize + lon_gridsize) / 2

    c_points = get_gridcenter_points(points, var_da)
    
    n_seascapes = c_points.shape[0]
    n_ss_gridpoints = math.ceil(seascape_size / gridsize)
    if not (n_ss_gridpoints % 2): n_ss_gridpoints += 1 # Must be odd to have a center pixel.

    # Calculate values in relative seascape grid
    half_range = (n_ss_gridpoints // 2) * gridsize
    ss_rlat_vals = np.linspace(-half_range, half_range, n_ss_gridpoints)
    ss_rlon_vals = np.linspace(-half_range, half_range, n_ss_gridpoints)

    # Calculate values of seascape and
    # stack them in a seascape_idx dimension

    ss_list = []

    for ss_idx, c_point in c_points.iterrows():
        c_point_lon = c_point['lon']
        c_point_lat = c_point['lat']
        seascape = var_da.sel(
            lat=slice(
                c_point_lat-(seascape_size+gridsize)/2,
                c_point_lat+(seascape_size+gridsize)/2
                ),
            lon=slice(
                c_point_lon-(seascape_size+gridsize)/2,
                c_point_lon+(seascape_size+gridsize)/2
                )
            )
        if seascape.size == 0:
            # TODO: Add error handling for empty seascapes
            raise NotImplementedError(f"Empty seascape for index {ss_idx}")
        else:
            # Change global coords to relative ss coords
            seascape = seascape.assign_coords(
                lat=ss_rlat_vals,
                lon=ss_rlon_vals
            )
            ss_list.append(seascape)


    xscp_data = xr.concat(
        ss_list, 
        pd.RangeIndex(
            n_seascapes,
            name='seascape_idx'
            )
        )

    # Construct DataArray
    xscp_da = xr.DataArray(
        data=xscp_data,
        coords={
            # Center pixel coordinates for each ss
            'c_lon': ('seascape_idx', c_points['lon']),
            'c_lat': ('seascape_idx', c_points['lat']),
            # Relative lat/lon with center pixel at (0,0)
            'ss_rlon': ('ss_lon', ss_rlon_vals),
            'ss_rlat': ('ss_lat', ss_rlat_vals),
            # Real-world coordinates for each pixel in each ss
            'ss_lon': (('seascape_idx','ss_lon'),\
                       c_points["lat"].values[:, np.newaxis] + ss_rlat_vals),
            'ss_lat': (('seascape_idx','ss_lat'),\
                       c_points["lon"].values[:, np.newaxis] + ss_rlon_vals),
        },
        dims=['seascape_idx', 'ss_lon', 'ss_lat'],
        name=f"{var_da.name}"
        # TODO: Add attrs
    )

    return xscp_da.chunk("auto")