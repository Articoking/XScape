"""Testing the creation of XScape DataArrays."""

import numpy as np
import xarray as xr
import xscape as xscp
import pytest

def get_2d_gridded_ones_da():
    """Create a sample Xarray dataset."""
    # Create dimensions for the dataset
    lat = np.arange(-2, 3, dtype=float)
    lon = np.arange(-2, 3, dtype=float)

    # Create a flat array full of 1s
    temperature = np.ones((len(lat), len(lon)))

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
def sample_ones_var_da():
    """Setup test data."""
    var_da = get_2d_gridded_ones_da()
    yield var_da

def test_km_grid_from_angular(
        sample_points,
        sample_ones_var_da,
    ):
    """Test creating a km grid Xscape DA from simple data."""
    

    # Test selecting outside of timerange
    with pytest.raises(ValueError):
        """
        Single pixel seascapes can't be converted to km grid since we don't
        have enough data to do the interpolation.
        """
        xscp_da = xscp.create_xscp_da(
            points = sample_points,
            seascape_size = 0.5,
            var_da = sample_ones_var_da,
        )
        
        km_xscp_da = xscp_da.xscp.to_km_grid(
            gridsize=1,
            extent=5,
        )

    # Three-pixel seascape
    with pytest.warns(
        UserWarning,
        match=r"Creating empty seascape *"):
        xscp_da = xscp.create_xscp_da(
            points = sample_points,
            seascape_size = 3,
            var_da = sample_ones_var_da,
        )
    
    km_xscp_da = xscp_da.xscp.to_km_grid(
        gridsize=1,
        extent=5,
    )

    assert "ss_x" in km_xscp_da.dims
    assert "ss_y" in km_xscp_da.dims
    assert "ss_x" in km_xscp_da.coords
    assert "ss_y" in km_xscp_da.coords
    assert "c_lat" in km_xscp_da.coords
    assert "c_lon" in km_xscp_da.coords
    assert "ss_lat" not in km_xscp_da.dims
    assert "ss_lon" not in km_xscp_da.dims
    assert "ss_lat" not in km_xscp_da.coords
    assert "ss_lon" not in km_xscp_da.coords

    assert km_xscp_da.sizes["seascape_idx"] == 4
    assert km_xscp_da.sizes["ss_x"] == 5
    assert km_xscp_da.sizes["ss_y"] == 5

    km_da_values = km_xscp_da.values
    assert np.nanmean(km_da_values) == 1
    eps = 0.0001 # Slight numerical errors may occur
    assert np.logical_or(abs(km_da_values - 1) <= eps, np.isnan(km_da_values)).all()
    assert np.isnan(km_da_values).any()
    assert not np.isnan(km_da_values).all()
    