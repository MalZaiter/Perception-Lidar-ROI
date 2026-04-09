"""
LUT Bounding Box Filter - Uses cone detection to create corridors for filtering
Builds a lookup table based on cone positions and track width
"""
from typing import Dict, List, Tuple, Optional
import open3d as o3d
import numpy as np
from .helpers import SingletonMeta
from .ConeClassifier import ConeClassifier


class LUTBoundingBoxFilter(metaclass=SingletonMeta):
    """
    Look-Up Table Bounding Box filter that uses cone positions to define
    a corridor of interest. Works across frames:
    - Frame 0: Detect cones, compute centerline and width
    - Frame N: Use stored centerline + width for fast filtering
    """

    def __init__(
        self,
        max_distance: float,
        z_min: float,
        z_max: float,
        ground_level: float,
        point_num: int,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        horizontal_plane_gradient: float,
        cone_radius: float = 0.1,
        cone_height: float = 0.3,
        # Match ConeClassifier settings used in lidar_processor_node
        min_cone_points: int = 5,
        l2_loss_threshold: float = 0.05,
        lin_loss_percentage: float = 0.1,
        max_track_half_width: float = 2.0,
        max_cone_lateral: float = 2.5,
        max_lut_size: int = 50,
    ):
        """
        Parameters
        ----------
        max_distance : float
            Maximum radial distance from lidar
        z_min : float
            Minimum height to keep
        z_max : float
            Maximum height to keep
        ground_level : float
            Level below which to consider ground points
        point_num : int
            Minimum number of points for ground removal
        distance_threshold : float
            RANSAC distance threshold
        ransac_n : int
            RANSAC sample size
        num_iterations : int
            RANSAC number of iterations
        horizontal_plane_gradient : float
            Horizontal plane gradient threshold
        cone_radius : float
            Cone radius for classification
        cone_height : float
            Cone height for classification
        min_cone_points : int
            Minimum points to consider for cone detection
        l2_loss_threshold : float
            L2 loss threshold for cone classification
        lin_loss_percentage : float
            Linearization loss threshold for cone classification
        """
        self.max_distance = max_distance
        self.z_min = z_min
        self.z_max = z_max
        self.ground_level = ground_level
        self.point_num = point_num
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.horizontal_plane_gradient = horizontal_plane_gradient

        # Cone classifier for detecting cones
        self.cone_classifier = ConeClassifier(
            radius=cone_radius,
            height=cone_height,
            minPoints=min_cone_points,
            l2LossTh=l2_loss_threshold,
            linLossPerc=lin_loss_percentage,
        )

        # Store LUT data across frames
        self.lut_centerline: Dict[float, float] = {}  # {x: y_center}
        self.lut_width: Dict[float, float] = {}  # {x: half_width}
        # History of cone centers used to build a stable corridor
        self.detected_cones: List[np.ndarray] = []

        # Heuristics to make the corridor robust against outliers
        # Maximum allowed half-width of the track (meters). Wider pairs
        # are treated as outliers and ignored when building the LUT.
        self.max_track_half_width: float = max_track_half_width

        # Maximum lateral distance |y| at which a cone is still
        # considered part of the track when building the LUT.
        self.max_cone_lateral: float = max_cone_lateral

        # Diagnostics storage for corridor filtering stats
        self._lut_filter_diagnostics = {}
        
        # Maximum LUT size - enforce to prevent unbounded memory growth
        self._max_lut_size = max_lut_size

    def detectCones(
        self, pcd: o3d.geometry.PointCloud, clustering_results: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Detect cones from clustered points.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Original point cloud
        clustering_results : List[np.ndarray]
            List of cluster indices from clustering algorithm
            
        Returns
        -------
        List[np.ndarray]
            List of detected cone centers (3D coordinates)
        """
        cones = []
        points = np.asarray(pcd.points)

        for cluster_indices in clustering_results:
            if len(cluster_indices) > 0:
                cluster_points = points[cluster_indices]

                # Try to fit a cone to this cluster
                is_cone, cone_center = self.cone_classifier.isCone(cluster_points)

                if is_cone[0] and cone_center is not None:
                    cones.append(cone_center[0])

        self.detected_cones = cones
        return cones

    def buildLUT(self, cones: List[np.ndarray], resolution: float = 0.1) -> None:
        """
        Build Look-Up Table (LUT) from detected cone positions.
        Creates a centerline and width data structure for fast filtering.
        
        For a racing track:
        - Left cones (odd indices) at y > 0
        - Right cones (even indices) at y < 0
        - Compute centerline: y_center = (y_left + y_right) / 2
        - Track width: w = (y_left - y_right) / 2
        
        Parameters
        ----------
        cones : List[np.ndarray]
            List of cone centers
        resolution : float
            Resolution of the LUT (step size in x direction)
        """
        # Gate cones by lateral distance and range so off-track
        # detections do not widen the corridor.
        gated_cones: List[np.ndarray] = []
        for c in cones:
            x_c, y_c, _ = c
            dist = float(np.hypot(x_c, y_c))
            if abs(y_c) > self.max_cone_lateral:
                continue
            if dist > self.max_distance:
                continue
            gated_cones.append(c)

        if len(gated_cones) < 2:
            return

        # Sort cones by x coordinate
        cones_sorted = sorted(gated_cones, key=lambda c: c[0])
        cones_array = np.array(cones_sorted)

        x_min = cones_array[:, 0].min()
        x_max = cones_array[:, 0].max()

        # Build LUT by x position
        x_positions = np.arange(x_min, x_max + resolution, resolution)

        for x in x_positions:
            # Find cones near this x position (within tolerance)
            tolerance = resolution * 1.5
            nearby_cones = cones_array[
                np.abs(cones_array[:, 0] - x) <= tolerance
            ]

            if len(nearby_cones) >= 2:
                # Separate left (y > 0) and right (y < 0) cones
                left_cones = nearby_cones[nearby_cones[:, 1] > 0]
                right_cones = nearby_cones[nearby_cones[:, 1] < 0]

                if len(left_cones) > 0 and len(right_cones) > 0:
                    # Compute centerline and width
                    y_left = left_cones[:, 1].mean()
                    y_right = right_cones[:, 1].mean()

                    y_center = (y_left + y_right) / 2.0
                    half_width = np.abs(y_left - y_right) / 2.0

                    # Ignore implausibly wide corridors so a bad
                    # cone pair does not include distant noise.
                    if half_width > self.max_track_half_width:
                        continue

                    self.lut_centerline[x] = y_center
                    self.lut_width[x] = half_width

        # Enforce maximum LUT size: when it exceeds 50, remove 10 oldest entries
        # This prevents unbounded growth while maintaining corridor continuity
        if len(self.lut_centerline) > self._max_lut_size:
            # Sort by x position (oldest/smallest X values first)
            x_positions_sorted = sorted(self.lut_centerline.keys())
            x_to_remove = x_positions_sorted[:10]  # Remove first 10 (oldest)
            
            for x_old in x_to_remove:
                self.lut_centerline.pop(x_old, None)
                self.lut_width.pop(x_old, None)

    def filterWithLUT(self, pcd: o3d.geometry.PointCloud,
                      margin: float = 3.0) -> o3d.geometry.PointCloud:
        """
        Filter point cloud using the Look-Up Table (corridor filtering).
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud
        margin : float
            Additional lateral margin around the corridor (in meters).
            Default is 3.0 m to allow for curves and slight
            misalignments while still rejecting far-off noise.
            
        Returns
        -------
        o3d.geometry.PointCloud
            Filtered point cloud
        """
        if not self.lut_centerline or not self.lut_width:
            # If LUT not built, return all points within basic bounds
            return self._filterBasicBounds(pcd)

        points = np.asarray(pcd.points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Initialize mask (keep all points initially)
        mask = np.ones(len(points), dtype=bool)

        # Create LUT position array once (not per-point)
        lut_x_positions = np.array(sorted(self.lut_centerline.keys()))
        if len(lut_x_positions) == 0:
            # If LUT not built, return all points within basic bounds
            return self._filterBasicBounds(pcd)

        # For each point, check if it's within the corridor
        for i in range(len(points)):
            x_pos = x[i]

            # Find nearest x in LUT using sorted array
            idx = np.searchsorted(lut_x_positions, x_pos, side='left')
            
            # Handle edge cases
            if idx >= len(lut_x_positions):
                idx = len(lut_x_positions) - 1
            elif idx > 0:
                # Check which is closer: idx or idx-1
                if np.abs(lut_x_positions[idx - 1] - x_pos) < np.abs(lut_x_positions[idx] - x_pos):
                    idx = idx - 1
            
            nearest_x = lut_x_positions[idx]

            y_center = self.lut_centerline[nearest_x]
            half_width = self.lut_width[nearest_x]

            # Check if point is within corridor (centerline ± width + margin)
            max_y_deviation = half_width + margin
            y_offset = y[i] - y_center

            if np.abs(y_offset) > max_y_deviation:
                mask[i] = False

        # Also apply z and distance bounds
        distance = np.sqrt(x**2 + y**2)
        bound_mask = (
            (distance <= self.max_distance)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )
        mask = mask & bound_mask

        # Store diagnostics for this filtering pass
        self._lut_filter_diagnostics = {
            'input_count': len(points),
            'corridor_pass': int(np.sum(mask & ~bound_mask + bound_mask)) if len(self.lut_centerline) > 0 else len(points),
            'bound_pass': int(np.sum(bound_mask)),
            'final_output': int(np.sum(mask)),
            'margin': margin,
            'lut_size': len(self.lut_centerline),
        }

        return pcd.select_by_index(np.where(mask)[0])

    def _filterBasicBounds(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply basic distance and z bounds when LUT is not available.
        
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
                num_iterations=self.num_iterations,
            )

            if abs(plane_model[2]) > self.horizontal_plane_gradient:
                pcd = pcd.select_by_index(ground_candidates[inliers], invert=True)

        return pcd

    def removeCar(
        self, pcd: o3d.geometry.PointCloud,
        car_x: List[float], car_y: List[float]
    ) -> o3d.geometry.PointCloud:
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

        mask = ~(
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
        )

        return pcd.select_by_index(np.where(mask)[0])

    def clearLUT(self) -> None:
        """Clear the LUT and detected cones."""
        self.lut_centerline.clear()
        self.lut_width.clear()
        self.detected_cones.clear()
