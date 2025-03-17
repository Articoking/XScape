import xscape
import pytest

n_points = 10
min_lat = 0
max_lat = 10
min_lon = -90
max_lon = 90

def test_generate_points():
    """Test simple point generation."""
    points = xscape.generate_points(
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
        points = xscape.generate_points(
            n_points,
            lon_range=(max_lon, min_lon),
            lat_range=(max_lat, min_lon),
        )