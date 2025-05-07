# XScape DataArray format

XScape relies on special Xarray `DataArray` objects with a very specific format.
These objects can be created rather simply by using the `xscp.create_xscp_da` function, which will output arrays with the format described below.

## Dimensions

An XScape `DataArray` has three dimensions: `seascape_idx`, `ss_lon` and `ss_lat` (and optionally `ss_time` and `depth`).

The former gives a unique index to each seascape, while the latter two index each pixel in a given seascape.
Additionally, `ss_lon` and `ss_lat` will always have an odd length.
This is to guarantee that there will always be a single "center pixel" in the seascape (more on this below).

When constructing a seascape with a time dimension, the same logic applies to `ss_time` as does to `ss_lon` and `ss_lat`.

Lastly, `depth` always corresponds to the full vertical dimension in the original background field from which the seascapes were taken.

## Coordinates

There are six coordinates, divided into three groups:

- **Center point coordinates:** `c_lat` and `c_lon`, which are indexed only by `seascape_idx` and contain the geographical coordinates of the center pixel of the seascape.
- **Relative lat/lon:** `ss_rlat` and `ss_rlon`, which locate each pixel in a relative lat/lon grid where the center pixel (i.e. (`c_lat`°N, `c_lon`°W)) corresponds to (0,0).
- **Absolute lat/lon:** `ss_lat` and `ss_lon`, which provide the real-world location of each pixel in a seascape.

The same idea applies to the time dimension if it exists: an absolute `ss_time` coordinate with timestamps for each spatial field in the seascape, and a relative `ss_rtime` showing the time delta between each timestamp and the one closest to the point's time.

When creating an XScape `DataArray` with `xscp.create_xscp_da`, each point in `points` is mapped to its closest pixel in `var_da`'s grid.
If many points are mapped to the same pixel, XScape won't waste resources making multiple copies of the corresponding seascape.
That is why you may find that the `seascape_idx` dimension has a smaller length than the number of `points`.
During the creation process no checks are made to verify that every point is actually in `var_da`'s grid.

When selecting a specific seascape using the `xscp.ss_sel(point)` method on a `DataArray`, XScape finds the seascape whose center pixel is the shortest distance from the specified `point`, and verifies that it is indeed inside the center pixel.
If a `point` is not inside the closest seascape's center pixel, then it means that said `point` does not correspond to any seascape in the `DataArray`, and so XScape will raise a `ValueError`.
