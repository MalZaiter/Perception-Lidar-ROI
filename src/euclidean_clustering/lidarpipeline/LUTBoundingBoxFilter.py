"""
LUT Bounding Box Filter - Uses cone detection to create corridors for filtering
Builds a lookup table based on cone positions and track width
"""
from typing import Dict, List
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
    # Shared ground-removal / bounds params
    max_distance: float,
    z_min: float,
    z_max: float,
    ground_level: float,
    point_num: int,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    horizontal_plane_gradient: float,
    # Cone gating
    max_track_half_width: float,
    max_cone_lateral: float,
    # Cone classifier
    cone_radius: float,
    cone_height: float,
    min_cone_points: int,
    l2_loss_threshold: float,
    lin_loss_percentage: float,
    # LUT building
    lut_resolution: float,
    lut_ema_alpha: float,
    lut_max_width_change: float,
    lut_tolerance_multiplier: float,
    # LUT filtering
    lut_filter_margin: float,
    lut_filter_x_margin_before: float,
    lut_filter_x_margin_after: float,
):
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

        # LUT building parameters from YAML
        self.lut_resolution: float = lut_resolution
        self.lut_ema_alpha: float = lut_ema_alpha
        self.lut_max_width_change: float = lut_max_width_change
        self.lut_tolerance_multiplier: float = lut_tolerance_multiplier
        
        # LUT filtering parameters from YAML
        self.lut_filter_margin: float = lut_filter_margin
        self.lut_filter_x_margin_before: float = lut_filter_x_margin_before
        self.lut_filter_x_margin_after: float = lut_filter_x_margin_after

        # Diagnostics storage for corridor filtering stats
        self._lut_filter_diagnostics = {}

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

    def buildLUT(self, cones: List[np.ndarray]) -> None:
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
        x_positions = np.arange(x_min, x_max + self.lut_resolution, self.lut_resolution)

        for x in x_positions:
            # Find cones near this x position (within tolerance)
            # Multiplier allows cones spaced 3-5m apart to pair
            tolerance = self.lut_resolution * self.lut_tolerance_multiplier
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

                    # EMA smoothing and rate limiting to reject abrupt changes
                    if x in self.lut_width:
                        # Check for abrupt width changes (possible mislabeling or outliers)
                        if abs(half_width - self.lut_width[x]) > self.lut_max_width_change:
                            continue  # Skip this entry; don't update
                        # Apply EMA smoothing to existing entries
                        self.lut_centerline[x] = self.lut_ema_alpha * y_center + (1 - self.lut_ema_alpha) * self.lut_centerline[x]
                        self.lut_width[x] = self.lut_ema_alpha * half_width + (1 - self.lut_ema_alpha) * self.lut_width[x]
                    else:
                        # First time seeing this X position
                        self.lut_centerline[x] = y_center
                        self.lut_width[x] = half_width

        # Apply sliding window LUT: keep only entries within ±max_distance of cone center
        # This makes LUT spatially consistent instead of temporally arbitrary
        if len(self.lut_centerline) > 0:
            # Compute center of current cones using median (robust to outliers)
            if len(cones_array) > 0:
                cone_x = cones_array[:, 0]
                x_center = np.median(cone_x)  # Median more robust than mean for outliers
                # Define window around current cone center
                x_min_window = x_center - self.max_distance
                x_max_window = x_center + self.max_distance
                # Remove LUT entries outside the window
                x_keys_to_remove = [
                    x for x in self.lut_centerline.keys()
                    if x < x_min_window or x > x_max_window
                ]
                for x_old in x_keys_to_remove:
                    self.lut_centerline.pop(x_old, None)
                    self.lut_width.pop(x_old, None)

    def filterWithLUT(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Filter point cloud using the Look-Up Table (corridor filtering).
        Uses margin and x-bounds from YAML parameters.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud
            
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

        lut_x_sorted = np.array(sorted(self.lut_centerline.keys()))
        lut_y_arr = np.array([self.lut_centerline[xi] for xi in lut_x_sorted])
        lut_w_arr = np.array([self.lut_width[xi] for xi in lut_x_sorted])

        lut_x_min = float(lut_x_sorted[0])
        lut_x_max = float(lut_x_sorted[-1])

        x_in_range = (x >= lut_x_min - self.lut_filter_x_margin_before) & (x <= lut_x_max + self.lut_filter_x_margin_after)

        idx = np.searchsorted(lut_x_sorted, x, side='left')
        idx = np.clip(idx, 0, len(lut_x_sorted) - 1)

        idx_prev = np.maximum(idx - 1, 0)
        use_prev = (idx > 0) & (np.abs(lut_x_sorted[idx_prev] - x) < np.abs(lut_x_sorted[idx] - x))
        idx = np.where(use_prev, idx_prev, idx)

        y_center = lut_y_arr[idx]
        half_width = lut_w_arr[idx]

        corridor_mask = np.abs(y - y_center) <= (half_width + self.lut_filter_margin)

        distance = np.sqrt(x**2 + y**2)
        bound_mask = (
            (distance <= self.max_distance)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )

        mask = corridor_mask & bound_mask & x_in_range
        # Store diagnostics for this filtering pass
        self._lut_filter_diagnostics = {
            'input_count': len(points),
            'corridor_pass': int(np.sum(corridor_mask & x_in_range)),
            'bound_pass': int(np.sum(bound_mask)),
            'final_output': int(np.sum(mask)),
            'margin': self.lut_filter_margin,
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

            # FIX A: Add inlier ratio guard to ensure plane fitting actually succeeded
            inlier_ratio = len(inliers) / max(len(ground_candidates), 1)
            if (abs(plane_model[2]) > self.horizontal_plane_gradient
                    and inlier_ratio > 0.05):
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
