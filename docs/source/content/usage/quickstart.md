# Quickstart guide

To use XScape, you only need two things:

- A list of points in a pandas DataFrame with `lat` and `lon` as columns.
- An Xarray DataArray containing the "background" variable in a regular latitude/longitude grid.

Let's start by generating some points in the South Atlantic Ocean

```python
import xarray as xr
import xscape as xscp

n_points = 10
points = xscp.testing_utils.generate_points(
    n_points,
    lon_range = (-50, 20),
    lat_range = (-60, 0)
    )
```

In this example, we'll use XScape functions to get Sea Surface Temperature data from GLORYS.
To do that, we need to first define the size of the seascape we want to get around each point, in this case 1°.
XScape uses this information to avoid requesting an area much larger than that covered by the points.

```python
seascape_size = 1 # degrees

glorys_var = xscp.testing_utils.get_glorys_var(
    points,
    seascape_size,
    'thetao', # Temperature variable name
    start_datetime = "2000-01-01",
    end_datetime = "2000-12-31",
    )

glorys_var = glorys_var.isel(time=0, depth=0) # See note below
```

**NOTE:** XScape can also create seascapes with time and depth dimensions, through the `seascape_timerange` and `get_column` parameters in `create_xscp_da()` respectively. Consult the [API documentation](../api.md) for more detail.

Finally, we create our "XScape DataArray" using the `create_xscp_da()` function:

```python
xscp_da = xscp.create_xscp_da(
    points,
    seascape_size,
    glorys_var
)
```

This is a regular Xarray DataArray object with a special format that makes working with seascapes quite simple.
For example, you can easily select the seascape that corresponds to the first point in your list:

```python
first_point = points.iloc[0]
seascape = xscp_da.xscp.ss_sel(first_point)
```

For more information on the format of XScape DataArrays refer to the [dedicated documentation page](da_format.md).
