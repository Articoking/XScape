"""Accessors to extend Xarray functionality."""

# https://docs.xarray.dev/en/stable/internals/extending-xarray.html

import numpy as np
import pandas as pd
import xarray as xr

import xscape.utils as utils

@xr.register_dataarray_accessor("xscp")
class XScapeDAAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._c_points = None
        if "seascape_gridsize" in xarray_obj.attrs.keys():
            self._gridsize = xarray_obj.attrs["seascape_gridsize"]
        else:
            self._gridsize = None

    @property
    def gridsize(self):
        """Horizontal pixel size of this DataArray."""
        if self._gridsize is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._gridsize = utils.calculate_horizontal_gridsize(self._obj)
        return self._gridsize
    
    @property
    def c_points(self):
        """DataFrame of center points of each seascape."""
        if self._c_points is None:
            # Reconstruct from "c_lon" and "c_lat" coordinates
            self._c_points = pd.DataFrame({
                    "lon": self._obj["c_lon"].values,
                    "lat": self._obj["c_lat"].values
                }, 
                index = self._obj["seascape_idx"].values
                )  # Preserve `seascape_idx` as index if needed
            
        return self._c_points

    def ss_sel(
        self,
        point: pd.Series,
        ) -> xr.DataArray:
        """
        Return the corresponding seascape for the specified point.

        Calculates the corresponding seascape index and performs `.isel()` on
        the calling object to retrieve it.

        Parameters
        ----------
        point : pd.Series
            Coordinates of the point in a series with "lat" and
            "lon" values.

        Returns
        -------
        xr.DataArray
            XScape-style DataArray containing only one seascape.
        """

        """
        Euclidean distance. Not accurate for long distances but in this case we
        would have at most gridsize/sqrt(2) degrees of distance.
        """
        distances = np.sqrt(
            (self.c_points['lat'] - point['lat'])**2
            + (self.c_points['lon'] - point['lon'])**2
            )

        # Get the index of the closest point
        closest_point_idx = distances.idxmin()

        # Check that `point` actually is in the seascape
        if distances[closest_point_idx] >= (self.gridsize / np.sqrt(2)):
            raise ValueError(
                "The specified point does not correspond to any seascape."
                )
        else:
            return self._obj.isel(seascape_idx=closest_point_idx)