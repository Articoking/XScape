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
    points = pd.DataFrame({"lat": [-3, -1, 0, 0.1, 1], "lon": [-3, 1, 0, -0.1, -1]})
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

@pytest.fixture(scope="module")
def sample_var_da():
    """Setup test data."""
    var_da = get_2d_gridded_da()
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