import pytest
import pandas as pd

@pytest.fixture
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