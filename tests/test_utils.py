import pandas as pd

import xscape as xscp
import pytest

n_points = 10
min_lat = 0
max_lat = 10
min_lon = -90
max_lon = 90

@pytest.fixture(scope="module")
def sample_points():
    points = pd.DataFrame({"lat": [-1, 0, 1], "lon": [1, 0, -1]})
    yield points

def test_generate_points():
    """Test simple point generation."""
    points = xscp.generate_points(
        n_points,
        lon_range=(min_lon, max_lon),
        lat_range=(min_lat, max_lat),
    )
    assert points.shape[0] == n_points
    assert points['lat'].min() >= min_lat
    assert points['lat'].max() <= max_lat
    assert points['lon'].min() >= min_lon
    assert points['lon'].max() <= max_lon

def test_generate_points_reverse():
    """Test generating with reverse bounds."""
    with pytest.raises(NotImplementedError):
        xscp.generate_points(
            n_points,
            lon_range=(max_lon, min_lon),
            lat_range=(max_lat, min_lon),
        )

def test_get_request_extent(sample_points):
    """Test request extent."""
    extent = xscp.utils.get_request_extent(
        sample_points,
        seascape_size = 1,
        gridsize = 1,
    )
    assert extent["maximum_latitude"] == 2.5
    assert extent["maximum_longitude"] == 2.5
    assert extent["minimum_latitude"] == -2.5
    assert extent["minimum_longitude"] == -2.5

def test_get_request_extent_error(sample_points):
    """Test request extent with invalid SS size."""
    with pytest.raises(ValueError):
        xscp.utils.get_request_extent(
            sample_points,
            seascape_size = -1,
            gridsize = 1,
        )