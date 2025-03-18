# XScape

![XScape](docs/XScape.png)

A Python library to recover the seascape around a set of geographical points, based on [Xarray](https://github.com/pydata/xarray).

## Why XScape?

**TODO:** write this.

## Installation

Until the package is officially published, the only way of installing XScape is by cloning the repository.
Once you have it on your own machine, go into the XScape directory and run  `poetry install`.
The whole process can be done in three commands:

```bash
git clone https://github.com/Articoking/XScape.git
cd XScape
poetry install
```

If you want to **contribute to the library,** you will likely want to install the extra developer dependencies.
To do that, use the `--with dev` option when running the installation.

## Basic usage

To use XScape, you only need two things:

- A list of points in a pandas DataFrame with `lat` and `lon` as columns.
- An Xarray DataArray containing the "background" variable in a regular latitude/longitude grid.

Let's start by generating some points in the South Atlantic Ocean

```python
import xarray as xr
import xscape as xscp

n_points = 10
points = xscp.generate_points(
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

glorys_var = xscp.get_glorys_var(
    points,
    seascape_size,
    'thetao', # Temperature variable name
    start_datetime = "2000-01-01",
    end_datetime = "2000-12-31",
)

glorys_var = glorys_var.isel(time=0, depth=0) # See note below
```

**NOTE:** XScape currently has no way of obtaining seascapes with time or depth as dimensions, but that will be added in the future.
For now, only 2D horizontal maps can be used to create seascapes.

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

For more information on the format of XScape DataArrays refer to the dedicated documentation page (**TODO:** add link).
