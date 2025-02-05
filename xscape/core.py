import math

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
    Generates a `n_points`-long DataFrame with randomly selected geographical points in the specified
    longitude and latitude ranges.
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
    variables: list,
    start_datetime,
    end_datetime,
    ) -> xr.Dataset:
    """
    Performs the data request to get the GLORYS data corresponding
    to the selected variables in the geographical extent, for the
    specified date.

    Returns a `xr.Dataset` object
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
    Returns a DataFrame with a list of (lat,lon) points, which correspond to the coordinates of the
    pixels of `var_da` in which each point in `points` is.
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