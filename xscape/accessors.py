"""Accessors to extend Xarray functionality."""

# https://docs.xarray.dev/en/stable/internals/extending-xarray.html

import math

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj, Transformer
from scipy.interpolate import griddata

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
        if "seascape_timestep" in xarray_obj.attrs.keys():
            self._timestep = xarray_obj.attrs["seascape_timestep"]
        else:
            self._timestep = None

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
            if "c_time" in self._obj.coords:
                self._c_points["time"] = self._obj["c_time"].values
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
        
        Raises
        ------
        ValueError
            If the point does not correspond to the center of any seascape.
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
        
        # If time-referenced, choose appropriate seascape
        # NOTE: many seascapes may share the same c_point but have different times.
        if ("time" in point.index) and ("c_time" in self._obj.coords):
            c_point = self.c_points.iloc[closest_point_idx]

            # Filtering rows that match the lat/lon of `c_point`
            matching_rows = self.c_points[
                (self.c_points["lat"] == c_point["lat"]) \
                & (self.c_points["lon"] == c_point["lon"])
                ]

            # Finding the row with the closest time
            delta_ts = (matching_rows["time"] - c_point["time"]).abs()
            closest_point_idx = delta_ts.idxmin()
            # Check that `point` actually is in the seascape
            if delta_ts[closest_point_idx] >= (self._timestep):
                raise ValueError(
                    "The specified point does not correspond to any seascape's timestamp."
                    )

        return self._obj.isel(seascape_idx=closest_point_idx)
    
    
    def to_km_grid(
        self,
        gridsize: float,
        extent: float,
        ) -> xr.DataArray:
        """
        Convert an XScape DataArray to a kilometric grid.

        Parameters
        ----------
        gridsize : float
            Size of the new grid in kilometers.
        extent : float
            extent of the new grid in kilometers.

        Returns
        -------
        xr.DataArray
            XScape DataArray regridded in the specified a kilometric grid.
        """

        assert gridsize <= extent

        patches = []
        n_ss_gridpoints = math.ceil(extent / gridsize)
        if not (n_ss_gridpoints % 2):
            n_ss_gridpoints += 1 # Must be odd to have a center pixel.
        half_range = (n_ss_gridpoints // 2) * gridsize
        lin = np.linspace(-half_range, half_range, n_ss_gridpoints)
        km_x, km_y = np.meshgrid(lin, lin)


        for ss_idx in range(self._obj.sizes["seascape_idx"]):
            c_lat = self.c_points["lat"].iloc[ss_idx]
            c_lon = self.c_points["lon"].iloc[ss_idx]
            # Projector: Azimuthal Equidistant centered at the patch center
            proj_aeqd = Proj(proj='aeqd', lat_0=c_lat, lon_0=c_lon, units='km')
            transformer = Transformer.from_proj("epsg:4326", proj_aeqd, always_xy=True)

            # Get lat/lon for the patch
            lat = self._obj["ss_lat"].isel(seascape_idx=ss_idx).values
            lon = self._obj["ss_lon"].isel(seascape_idx=ss_idx).values
            lon2d, lat2d = np.meshgrid(lon, lat)

            # Flatten and project
            x_km, y_km = transformer.transform(lon2d.flatten(), lat2d.flatten())
            data_patch = self._obj.isel(seascape_idx=ss_idx).values
            points = np.stack([x_km.ravel(), y_km.ravel()], axis=1)
            values = data_patch.ravel()

            grid_patch = griddata(points, values, (km_x, km_y), method='linear')

            patches.append(grid_patch)

        # Stack into new DataArray
        out = xr.DataArray(
            data=np.stack(patches),
            dims=("seascape_idx", "ss_y", "ss_x"),
            coords={
                "c_lat": self._obj.coords["c_lat"],
                "c_lon": self._obj.coords["c_lon"],
                "ss_y": lin,
                "ss_x": lin,
            },
            name=self._obj.name if self._obj.name else None,
            attrs=self._obj.attrs
        )
        out.attrs["is_kilometric"] = True
        out.attrs["seascape_gridsize"] = gridsize
        return out