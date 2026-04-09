"""Basic distance + height bounds filter.

Keeps points within a given radial distance from the sensor origin and
between z_min and z_max. This is the simple fallback used when no LUT
corridor is available, now exposed as its own filter for comparison.
"""

from typing import Dict, List
import open3d as o3d
import numpy as np
from .helpers import SingletonMeta


class BasicBoundsFilter(metaclass=SingletonMeta):
    """Filter that keeps points within max_distance and z bounds."""

    def __init__(self, max_distance: float, z_min: float, z_max: float):
        self.max_distance = max_distance
        self.z_min = z_min
        self.z_max = z_max

    def filterBasicBounds(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply simple distance and z bounds to the point cloud."""
        points = np.asarray(pcd.points)
        if points.size == 0:
            return pcd

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        distance = np.sqrt(x**2 + y**2)
        mask = (
            (distance <= self.max_distance)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )

        return pcd.select_by_index(np.where(mask)[0])
