"""
Pizza Slice ROI Filter - Filters point cloud based on angular and distance bounds
Defined by maximum distance, minimum angle, maximum angle, and z-filter for vertical points
"""
from typing import Dict, List
import open3d as o3d
import numpy as np
from .helpers import SingletonMeta


class PizzaSliceFilter(metaclass=SingletonMeta):
    """
    Pizza Slice Region of Interest filter that keeps points within a sector
    defined by angular bounds and maximum distance.
    """

    def __init__(
        self,
        max_distance: float,
        min_angle: float,
        max_angle: float,
        z_min: float,
        z_max: float,
        ground_level: float,
        point_num: int,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        horizontal_plane_gradient: float,
    ):
        """
        Parameters
        ----------
        max_distance : float
            Maximum radial distance from the lidar origin (in meters)
        min_angle : float
            Minimum azimuth angle (in radians) - typically negative for left side
        max_angle : float
            Maximum azimuth angle (in radians) - typically positive for right side
        z_min : float
            Minimum height (z-coordinate) to keep
        z_max : float
            Maximum height (z-coordinate) to keep
        ground_level : float
            Level below which to consider ground points
        point_num : int
            Minimum number of points considered for ground removal
        distance_threshold : float
            RANSAC distance threshold for ground plane fitting
        ransac_n : int
            RANSAC sample size
        num_iterations : int
            RANSAC number of iterations
        horizontal_plane_gradient : float
            Gradient threshold for horizontal plane detection
        """
        self.max_distance = max_distance
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.z_min = z_min
        self.z_max = z_max
        self.ground_level = ground_level
        self.point_num = point_num
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.horizontal_plane_gradient = horizontal_plane_gradient

    def filterPizzaSlice(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Filter point cloud to keep only points within the pizza slice ROI.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud
            
        Returns
        -------
        o3d.geometry.PointCloud
            Filtered point cloud
        """
        points = np.asarray(pcd.points)
        
        # Calculate distance and angle for each point
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Radial distance in xy plane
        distance = np.sqrt(x**2 + y**2)
        
        # Azimuth angle (atan2 gives angle in [-pi, pi])
        angle = np.arctan2(y, x)
        
        # Create mask for pizza slice (distance, angle, and z bounds)
        distance_mask = distance <= self.max_distance
        
        # Handle angle wrapping (angles can wrap around at ±π)
        # This creates a sector from min_angle to max_angle
        if self.min_angle <= self.max_angle:
            angle_mask = (angle >= self.min_angle) & (angle <= self.max_angle)
        else:
            # Wrap-around case (e.g., -π to π)
            angle_mask = (angle >= self.min_angle) | (angle <= self.max_angle)
        
        z_mask = (z >= self.z_min) & (z <= self.z_max)
        
        # Combine all masks
        mask = distance_mask & angle_mask & z_mask
        
        # Select points that pass the filter
        filtered_pcd = pcd.select_by_index(np.where(mask)[0])
        
        return filtered_pcd

    def removeGround(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove ground plane from the point cloud using RANSAC.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud
            
        Returns
        -------
        o3d.geometry.PointCloud
            Point cloud with ground removed
        """
        points = np.asarray(pcd.points)
        
        ground_candidates = np.where(points[:, 2] < self.ground_level)[0]
        
        if len(ground_candidates) > self.point_num:
            ground_pcd = pcd.select_by_index(ground_candidates)
            
            plane_model, inliers = ground_pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if abs(plane_model[2]) > self.horizontal_plane_gradient:
                pcd = pcd.select_by_index(ground_candidates[inliers], invert=True)
        
        return pcd

    def removeCar(self, pcd: o3d.geometry.PointCloud, 
                   car_x: List[float], car_y: List[float]) -> o3d.geometry.PointCloud:
        """
        Remove points that fall on the car itself.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud
        car_x : List[float]
            [x_min, x_max] for car dimensions
        car_y : List[float]
            [y_min, y_max] for car dimensions
            
        Returns
        -------
        o3d.geometry.PointCloud
            Point cloud with car points removed
        """
        points = np.asarray(pcd.points)
        
        x_min, x_max = car_x
        y_min, y_max = car_y
        
        mask = ~((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                 (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
        
        return pcd.select_by_index(np.where(mask)[0])
