"""Testing the creation of XScape DataArrays."""

import numpy as np
import pandas as pd
import xarray as xr
import xscape as xscp
import pytest

@pytest.fixture(scope="module")
def sample_points():
    """
    Setup sample points.
    
    In conjuction with the sample DataArray, we have two points in the center
    pixel (0,0) and also a point outside of the grid (-3,-3)
    """
    points = pd.DataFrame({
        "lat": [-3, -1, 0, 0.1, 1],
        "lon": [-3, 1, 0, -0.1, -1]
        })
    yield points

@pytest.fixture(scope="module")
def sample_points_with_time():
    """
    Setup sample points with time indeces.
    """
    points = pd.DataFrame({
        "lat": [-3, -1, 0, 0, 0.1, 1],
        "lon": [-3, 1, 0, 0, -0.1, -1],
        "time": [
            np.datetime64("2020-01-04"),
            np.datetime64("2020-01-04"),
            np.datetime64("2020-01-09"),
            np.datetime64("2020-01-13"),
            np.datetime64("2020-01-13"),
            np.datetime64("2020-01-19"),
            ]
        })
    yield points

def get_2d_gridded_da():
    """Create a sample Xarray dataset."""
    # Create dimensions for the dataset
    lat = np.arange(-2, 3, dtype=float)
    lon = np.arange(-2, 3, dtype=float)

    # Create some random temperature data
    temperature = 15 + 8 * np.random.randn(len(lat), len(lon))

    # Define the data in an xarray dataset
    da = xr.DataArray(
        data=temperature,
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    # define dataset attributes
    da.attrs.update(
        {
            "title": "Sea Surface Temperature",
            "description": "Generated for testing XScape.",
            "units": "Celsius",
        }
    )
    return da

def get_3d_gridded_da():
    """Create a sample Xarray dataset with a time dimension."""
    # Create dimensions for the dataset
    lat = np.arange(-2, 3, dtype=float)
    lon = np.arange(-2, 3, dtype=float)
    time = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-31"),
        np.timedelta64(1, "D"))

    # Create some random temperature data
    temperature = 15 + 8 * np.random.randn(len(lat), len(lon), len(time))

    # Define the data in an xarray dataset
    da = xr.DataArray(
        data=temperature,
        coords={
            "lat": lat,
            "lon": lon,
            "time": time,
        },
    )
    # define dataset attributes
    da.attrs.update(
        {
            "title": "Sea Surface Temperature",
            "description": "Generated for testing XScape.",
            "units": "Celsius",
        }
    )
    return da

@pytest.fixture(scope="module")
def sample_var_da():
    """Setup test data."""
    var_da = get_2d_gridded_da()
    yield var_da

@pytest.fixture(scope="module")
def sample_var_da_with_time():
    var_da = get_3d_gridded_da()
    yield var_da

def test_create_xscp_da(
    sample_var_da,
    sample_points,
    ):
    """Tests creating an XScape DataArray."""
    with pytest.warns(
        UserWarning,
        match=r"Creating empty seascape *"):

        xscp_da = xscp.create_xscp_da(
            points = sample_points,
            seascape_size = 3,
            var_da = sample_var_da,
        )

    assert xscp_da.sizes["seascape_idx"] == 4
    assert xscp_da.sizes["ss_lat"] == 3
    assert xscp_da.sizes["ss_lon"] == 3
    
    # Testing selection
    point = sample_points.iloc[2]
    seascape = xscp_da.xscp.ss_sel(point)
    assert seascape.c_lon == 0
    assert seascape.c_lat == 0

    # Test selecting outside of range
    with pytest.raises(ValueError):
        point = sample_points.iloc[0]
        xscp_da.xscp.ss_sel(point)

def test_create_xscp_da_single_pixel(
    sample_var_da,
    sample_points,
    ):
    """Tests creating an XScape DataArray with a single pixel seascape."""
    xscp_da = xscp.create_xscp_da(
        points = sample_points,
        seascape_size = 0.5,
        var_da = sample_var_da,
    )

    assert xscp_da.sizes["seascape_idx"] == 4
    assert xscp_da.sizes["ss_lat"] == 1
    assert xscp_da.sizes["ss_lon"] == 1
    
    # Testing selection
    point = sample_points.iloc[2]
    seascape = xscp_da.xscp.ss_sel(point)
    assert seascape.c_lon == 0
    assert seascape.c_lat == 0

    # Test selecting outside of range
    with pytest.raises(ValueError):
        point = sample_points.iloc[0]
        xscp_da.xscp.ss_sel(point)

def test_create_xscp_da_timerange(
    sample_var_da_with_time,
    sample_points_with_time,
    ):
    """Tests creating an XScape DataArray with a seascape timerange."""
    xscp_da = xscp.create_xscp_da(
        points = sample_points_with_time,
        seascape_size = 0.5,
        var_da = sample_var_da_with_time,
        seascape_timerange=np.timedelta64(60, "h") # 2.5 days
    )

    assert xscp_da.sizes["seascape_idx"] == 5
    assert xscp_da.sizes["ss_lat"] == 1
    assert xscp_da.sizes["ss_lon"] == 1
    assert xscp_da.sizes["ss_time"] == 3
    
    # Testing selection
    point = sample_points_with_time.iloc[2]
    seascape = xscp_da.xscp.ss_sel(point)
    assert seascape.c_lon == 0
    assert seascape.c_lat == 0
    assert seascape.c_time == np.datetime64("2020-01-09")

    # Test selecting outside of range
    with pytest.raises(ValueError):
        point = sample_points_with_time.iloc[0]
        xscp_da.xscp.ss_sel(point)
    
    # Test selecting outside of timerange
    with pytest.raises(ValueError):
        point = sample_points_with_time.iloc[0].copy()
        point['time'] = np.datetime64("2020-01-14")
        xscp_da.xscp.ss_sel(point)