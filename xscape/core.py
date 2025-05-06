"""Core functionality of XScape."""

import math
from typing import List
import warnings

import xarray as xr
import pandas as pd
import numpy as np
import copernicusmarine as cmems

import xscape.utils as utils

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

def create_xscp_da(
    points: pd.DataFrame,
    seascape_size: float,
    var_da:xr.DataArray,
    seascape_timerange: np.timedelta64 | None = None,
    get_column: bool = False
    ) -> xr.DataArray:
    """
    Crops and packages together a series of seascapes.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame of points as rows with "lat" and "lon" columns. If
        `seascape_timerange` is specified it must also have a "time" column.
    seascape_size : float
        Size (in degrees) of the seascape around each point.
    var_da : xr.DataArray
        Gridded background field from which we extract the seascapes.
    seascape_timerange: np.timedelta64, optional
        Duration of the seascape around each point's timestamp. If not an exact
        multiple of the timestep duration, take the minimum number of timesteps
        so that the specified duration is covered.
    get_column: bool, optional
        Whether to include a vertical dimension in the seascape. If True,
        `var_da` must have a dimension and coordinate named "depth" or "height".

    Returns
    -------
    xr.DataArray
        A DataArray indexed by `seascape_idx`, `ss_lon` and `ss_lat`. The latter
        two coordinates correspond to a relative reference frame centered on
        each seascape.

    Raises
    ------
    ValueError
        When `var_da` has a time dimension but no `seascape_timerange` is
        specified.
    """

    if (seascape_timerange is None) and ("time" in var_da.dims):
        raise ValueError(
            "var_da has a time dimension but seascape_timerange was not "\
            "specified."
            )
    
    vert_dimname = utils.get_vert_dimname(var_da)
    vert_coordname = utils.get_vert_coordname(var_da)

    gridsize = utils.calculate_horizontal_gridsize(var_da)
    n_ss_gridpoints = math.ceil(seascape_size / gridsize)
    if not (n_ss_gridpoints % 2):
        n_ss_gridpoints += 1 # Must be odd to have a center pixel.
    c_points = utils.get_gridcenter_points(points, var_da)
    
    # Calculate values in relative seascape grid
    half_range = (n_ss_gridpoints // 2) * gridsize
    ss_rlat_vals = np.linspace(-half_range, half_range, n_ss_gridpoints)
    ss_rlon_vals = np.linspace(-half_range, half_range, n_ss_gridpoints)

    if seascape_timerange is not None:
        c_points = utils.get_gridcenter_time(c_points, var_da)
        ss_timestep_duration = utils.calculate_timestep_duration(var_da)
        n_ss_timesteps = math.ceil(seascape_timerange / ss_timestep_duration)
        if not (n_ss_timesteps % 2):
            n_ss_timesteps += 1 # Must be odd to have a center time.
        
        half_t_range = (n_ss_timesteps // 2) * ss_timestep_duration
        ss_rtime_vals = np.linspace(
            -half_t_range.astype('timedelta64[ns]').astype(int),
            half_t_range.astype('timedelta64[ns]').astype(int),
            n_ss_timesteps
            ).astype('timedelta64[ns]')
    else:
        ss_rtime_vals = None

    n_seascapes = c_points.shape[0]

    if get_column or (vert_dimname is None):
        background_da = var_da
    else:
        # Select vertical level
        warning_msg = "Automatically selecting the first level in dimension: " \
            f"{vert_dimname}.\n" \
            "Consider setting get_column=True or select a vertical level manually."
        warnings.warn(warning_msg, stacklevel=2)
        background_da = var_da.isel({vert_dimname: 0})
        
    
    # Extract values of data in seascape and
    # stack them in a seascape_idx dimension
    ss_list = []

    for _, c_point in c_points.iterrows():
        c_point_lon = c_point["lon"]
        c_point_lat = c_point["lat"]
        sel_dict = dict(
            lat=slice(
                c_point_lat-n_ss_gridpoints*gridsize/2,
                c_point_lat+n_ss_gridpoints*gridsize/2
                ),
            lon=slice(
                c_point_lon-n_ss_gridpoints*gridsize/2,
                c_point_lon+n_ss_gridpoints*gridsize/2
                )
        )

        if seascape_timerange is not None:
            c_point_time = c_point["time"]
            sel_dict["time"] = slice(
                c_point_time-n_ss_timesteps*ss_timestep_duration/2,
                c_point_time+n_ss_timesteps*ss_timestep_duration/2
                )
            
        seascape = background_da.sel(sel_dict)
        
        try:
            # Change global coords to relative ss coords
            coord_dict = dict(
                lat=ss_rlat_vals,
                lon=ss_rlon_vals
            )
            if seascape_timerange is not None:
                coord_dict["time"] = ss_rtime_vals
            seascape = seascape.assign_coords(coord_dict)
            
        except ValueError:
            # Add empty seascape to prevent size mismatches later
            # See issue #7
            warning_msg = "Creating empty seascape for c_point: "\
                f"(lat={c_point["lat"]}, lon={c_point["lon"]})." \
                "This may be due to the corresponding point being outside " \
                "var_da's grid or too close to its edge."
            warnings.warn(warning_msg, stacklevel=2)
            seascape = utils.create_empty_seascape(
                ss_rlon_vals=ss_rlon_vals,
                ss_rlat_vals=ss_rlat_vals,
                ss_rtime_vals=ss_rtime_vals
            )

        ss_list.append(seascape)

    xscp_data = xr.concat(
        ss_list, 
        pd.RangeIndex(
            n_seascapes,
            name="seascape_idx"
            )
        )

    # Construct xr.DataArray
    xscp_coords = {
            # Center pixel coordinates for each ss
            "c_lon": ("seascape_idx", c_points["lon"]),
            "c_lat": ("seascape_idx", c_points["lat"]),
            # Relative lat/lon with center pixel at (0,0)
            "ss_rlon": ("ss_lon", ss_rlon_vals),
            "ss_rlat": ("ss_lat", ss_rlat_vals),
            # Real-world coordinates for each pixel in each ss
            "ss_lon": (
                ("seascape_idx","ss_lon"),
                c_points["lon"].values[:, np.newaxis] + ss_rlon_vals
                ),
            "ss_lat": (
                ("seascape_idx","ss_lat"),
                c_points["lat"].values[:, np.newaxis] + ss_rlat_vals
                ),
        }
    xscp_dims = ["seascape_idx", "ss_lon", "ss_lat"]
    if seascape_timerange is not None:
        xscp_coords["c_time"] = ("seascape_idx", c_points["time"])
        xscp_coords["ss_rtime"] =  ("ss_time", ss_rtime_vals)
        xscp_coords["ss_time"] = (
            ("seascape_idx","ss_time"),
            c_points["time"].values[:, np.newaxis] + ss_rtime_vals
            )
        xscp_dims.append("ss_time")
    if get_column:
        xscp_coords[vert_coordname] = (
            vert_dimname,
            background_da[vert_coordname].data
            )
        xscp_dims.append(vert_dimname)

    
    # Make sure the data dimensions are ordered properly
    xscp_data = xscp_data.transpose(*[
        "seascape_idx",
        "lat",
        "lon",
        "time",
        vert_dimname
        ], missing_dims='ignore') # In case there is no time/vertical dimension
    
    xscp_attrs = {"seascape_gridsize": gridsize} # See issue #13
    if seascape_timerange is not None:
        xscp_attrs["seascape_timestep"] = ss_timestep_duration

    xscp_da = xr.DataArray(
        data=xscp_data,
        coords=xscp_coords,
        dims=xscp_dims,
        name=f"{var_da.name}",
        attrs = xscp_attrs,
    )

    return xscp_da.chunk("auto")