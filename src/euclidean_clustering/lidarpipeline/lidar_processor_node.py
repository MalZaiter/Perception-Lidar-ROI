import time
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from .Filter import Filter
from .EuclideanClustering import clustering
import pandas as pd
from .ConeClassifier import ConeClassifier
from .PizzaSliceFilter import PizzaSliceFilter
from .LUTBoundingBoxFilter import LUTBoundingBoxFilter
from .BasicBoundsFilter import BasicBoundsFilter
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt



class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('filter_algorithm', 'box'),  # 'box', 'pizza', 'lut'
                ('x', [-20.0, 20.0]),
                ('y', [-20.0, 20.0]),
                ('z', [-2.0, 2.0]),


                ('car_x', [-1.0, 1.0]),
                ('car_y', [-0.5, 0.5]),


                ('ground_level', 0.2),
                ('point_num', 10),
                ('distance_threshold', 0.15),
                ('ransac_n', 3),
                ('num_iterations', 200),
                ('horizontal_plane_gradient', 0.8),

                # Pizza slice filter params
                ('pizza_max_distance', 20.0),
                ('pizza_min_angle_deg', -45.0),
                ('pizza_max_angle_deg', 45.0),
                ('pizza_z', [-2.0, 2.0]),

                # LUT bounding box filter params
                ('lut_max_distance', 20.0),
                ('lut_z', [-2.0, 2.0]),
                ('lut_max_cone_lateral', 2.5),
                ('lut_max_track_half_width', 2.0),

                ('voxel_size', 0.03),

                # Optional CSV log file for per-frame timings
                # If empty, a default timings_<filter_algorithm>.csv will be used
                ('timing_log_path', ''),

                # Optional directory to save filtered frames as PCD files.
                # If empty, no frames are written.
                ('save_frame_dir', ''),

                # Maximum number of frames to save per run when
                # save_frame_dir is set. If 1, behavior matches the
                # original single-frame export.
                ('save_frame_limit', 1),

                ('cluster_distance_threshold', 0.5),
                ('cluster_min_size', 5),
                ('cluster_max_size', 500)
            ]
        )

        # Original axis-aligned box filter
        self.filter = Filter(
            viewableBounds={"x": self.get_parameter('x').value, "y": self.get_parameter('y').value, "z": self.get_parameter('z').value},
            carDimensions={"x": self.get_parameter('car_x').value, "y": self.get_parameter('car_y').value},
            ground_level= self.get_parameter('ground_level').get_parameter_value().double_value,
            point_num= self.get_parameter('point_num').get_parameter_value().integer_value,
            distance_threshold= self.get_parameter('distance_threshold').get_parameter_value().double_value,
            ransac_n= self.get_parameter('ransac_n').get_parameter_value().integer_value,
            num_iterations= self.get_parameter('num_iterations').get_parameter_value().integer_value,
            horizontal_plane_gradient= self.get_parameter('horizontal_plane_gradient').get_parameter_value().double_value,
        )

        # Pizza slice ROI filter (sector in polar coordinates)
        pizza_max_distance = self.get_parameter('pizza_max_distance').get_parameter_value().double_value
        pizza_min_angle_deg = self.get_parameter('pizza_min_angle_deg').get_parameter_value().double_value
        pizza_max_angle_deg = self.get_parameter('pizza_max_angle_deg').get_parameter_value().double_value
        pizza_z = self.get_parameter('pizza_z').value

        self.pizza_filter = PizzaSliceFilter(
            max_distance=pizza_max_distance,
            min_angle=np.deg2rad(pizza_min_angle_deg),
            max_angle=np.deg2rad(pizza_max_angle_deg),
            z_min=pizza_z[0],
            z_max=pizza_z[1],
            ground_level=self.get_parameter('ground_level').get_parameter_value().double_value,
            point_num=self.get_parameter('point_num').get_parameter_value().integer_value,
            distance_threshold=self.get_parameter('distance_threshold').get_parameter_value().double_value,
            ransac_n=self.get_parameter('ransac_n').get_parameter_value().integer_value,
            num_iterations=self.get_parameter('num_iterations').get_parameter_value().integer_value,
            horizontal_plane_gradient=self.get_parameter('horizontal_plane_gradient').get_parameter_value().double_value,
        )

        # LUT-based corridor filter (falls back to simple distance/z bounds
        # until a LUT is built)
        lut_max_distance = self.get_parameter('lut_max_distance').get_parameter_value().double_value
        lut_z = self.get_parameter('lut_z').value
        lut_max_cone_lateral = self.get_parameter('lut_max_cone_lateral').get_parameter_value().double_value
        lut_max_track_half_width = self.get_parameter('lut_max_track_half_width').get_parameter_value().double_value

        self.lut_filter = LUTBoundingBoxFilter(
            max_distance=lut_max_distance,
            z_min=lut_z[0],
            z_max=lut_z[1],
            ground_level=self.get_parameter('ground_level').get_parameter_value().double_value,
            point_num=self.get_parameter('point_num').get_parameter_value().integer_value,
            distance_threshold=self.get_parameter('distance_threshold').get_parameter_value().double_value,
            ransac_n=self.get_parameter('ransac_n').get_parameter_value().integer_value,
            num_iterations=self.get_parameter('num_iterations').get_parameter_value().integer_value,
            horizontal_plane_gradient=self.get_parameter('horizontal_plane_gradient').get_parameter_value().double_value,
            max_cone_lateral=lut_max_cone_lateral,
            max_track_half_width=lut_max_track_half_width,
        )

        # Simple distance+height bounds filter (no LUT corridor) used for
        # the "lut_basic" algorithm option.
        self.basic_bounds_filter = BasicBoundsFilter(
            max_distance=lut_max_distance,
            z_min=lut_z[0],
            z_max=lut_z[1],
        )


        self.cone= ConeClassifier(
            radius=0.1, 
            height=0.3, 
            minPoints=1, 
            l2LossTh=0.1, 
            linLossPerc=0.2
        )

        self.processing_times = []
        self.reduction_percentages = []

        # Frame counter and timing log path
        self.frame_index = 0
        algo_param = self.get_parameter('filter_algorithm').get_parameter_value().string_value
        log_path_param = self.get_parameter('timing_log_path').get_parameter_value().string_value
        if not log_path_param:
            log_path_param = f'timings_{algo_param}.csv'
        self.timing_log_path = log_path_param
        self.get_logger().info(f'Per-frame timings will be logged to: {self.timing_log_path}')

        # Diagnostics log (LUT state, corridor filtering, point counts)
        self.diagnostics_log_path = f'diagnostics_{algo_param}.csv'
        self.get_logger().info(f'Diagnostics will be logged to: {self.diagnostics_log_path}')

        # Optional frame saving
        self.save_frame_dir = self.get_parameter('save_frame_dir').get_parameter_value().string_value
        self.save_frame_limit = self.get_parameter('save_frame_limit').get_parameter_value().integer_value
        self._frames_saved = 0


        
        # Large queue depth so a slow pipeline can still process every
        # message from the bag without dropping frames.
        self.sub = self.create_subscription(PointCloud2,'/velodyne_points', self.callback, 2000)

        # Primary filtered cloud (selected algorithm) and per-algorithm topics
        self.pub = self.create_publisher(PointCloud2, '/filtered_points', 10)
        self.pub_box = self.create_publisher(PointCloud2, '/filtered_points_box', 10)
        self.pub_pizza = self.create_publisher(PointCloud2, '/filtered_points_pizza', 10)
        self.pub_lut = self.create_publisher(PointCloud2, '/filtered_points_lut', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cluster_markers', 10)
        self.get_logger().info('Lidar Processor Node (ROS 2) has started.')


    def _run_filter_pipeline(self, algo_key: str, base_points: np.ndarray, car_x, car_y):
        """Run ROI + car/ground removal + voxel filter for a given algorithm.

        Returns a dict with the processed point cloud and counts/timings.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(base_points.copy())

        # For the pizza algorithm, remove ground on the full field of view
        # *before* applying the pizza slice ROI.
        if algo_key == 'pizza':
            pcd = self.pizza_filter.removeGround(pcd)

        # Measure full pipeline time for this algorithm on this frame
        pipe_start = time.perf_counter_ns()

        # ROI step timing (subset of the full pipeline)
        roi_start = time.perf_counter_ns()
        if algo_key == 'box':
            pcd = self.filter.filterViewableArea(pcd)
            label = 'Old Box'
        elif algo_key == 'pizza':
            pcd = self.pizza_filter.filterPizzaSlice(pcd)
            label = 'Pizza Slice'
        elif algo_key == 'lut':
            pcd = self.lut_filter.filterWithLUT(pcd)
            label = 'LUT Bounding Box'
        elif algo_key == 'lut_basic':
            # Simple distance + height bounds (no corridor, no cones)
            pcd = self.basic_bounds_filter.filterBasicBounds(pcd)
            label = 'Basic Distance-Z Bounds'
        else:
            self.get_logger().warn(f'Unknown filter key "{algo_key}", defaulting to box.')
            pcd = self.filter.filterViewableArea(pcd)
            label = 'Old Box (default)'
        roi_end = time.perf_counter_ns()
        roi_time_ms = (roi_end - roi_start) / 1e6

        after_view = len(pcd.points)
        if after_view == 0:
            # No points after ROI; pipeline time is just the ROI step
            pipe_end = time.perf_counter_ns()
            pipeline_time_ms = (pipe_end - pipe_start) / 1e6
            return {
                'algo_key': algo_key,
                'label': label,
                'roi_time_ms': roi_time_ms,
                'pipeline_time_ms': pipeline_time_ms,
                'pcd': pcd,
                'after_view': after_view,
                'after_car': 0,
                'after_ground': 0,
                'after_voxel': 0,
            }

        # Car removal
        if algo_key == 'box':
            pcd = self.filter.removeCar(pcd)
        elif algo_key == 'pizza':
            pcd = self.pizza_filter.removeCar(pcd, car_x, car_y)
        elif algo_key in ('lut', 'lut_basic'):
            pcd = self.lut_filter.removeCar(pcd, car_x, car_y)

        after_car = len(pcd.points)

        # Ground removal
        if algo_key == 'box':
            pcd = self.filter.removeGround(pcd)
        elif algo_key == 'pizza':
            # Ground already removed before ROI for pizza; no-op here.
            pass
        elif algo_key in ('lut', 'lut_basic'):
            pcd = self.lut_filter.removeGround(pcd)

        after_ground = len(pcd.points)

        voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        after_voxel = len(pcd.points)

        pipe_end = time.perf_counter_ns()
        pipeline_time_ms = (pipe_end - pipe_start) / 1e6

        result_dict = {
            'algo_key': algo_key,
            'label': label,
            'roi_time_ms': roi_time_ms,
            'pipeline_time_ms': pipeline_time_ms,
            'pcd': pcd,
            'after_view': after_view,
            'after_car': after_car,
            'after_ground': after_ground,
            'after_voxel': after_voxel,
        }
        
        # Capture LUT diagnostics if available
        if algo_key in ('lut', 'lut_basic') and hasattr(self.lut_filter, '_lut_filter_diagnostics'):
            result_dict['lut_diagnostics'] = self.lut_filter._lut_filter_diagnostics
        
        return result_dict


    def callback(self, msg):
        self.get_logger().info('--- New Packet Received ---')
        start_time = time.perf_counter_ns()

        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_list = [[p[0], p[1], p[2]] for p in gen]

        points = np.array(points_list, dtype=np.float64)

        if points.size == 0:
            self.get_logger().warn("Empty cloud received")
            return

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(points)

        pcd_raw = self.filter.removeIntensity(pcd_raw)
        base_points = np.asarray(pcd_raw.points)

        initial_count = len(base_points)
        self.get_logger().info(f'1. Raw Points: {initial_count}')

        car_x = self.get_parameter('car_x').value
        car_y = self.get_parameter('car_y').value
        # Select which algorithm to run for this pass
        filter_algorithm = self.get_parameter('filter_algorithm').get_parameter_value().string_value
        algo_key = filter_algorithm if filter_algorithm in ('box', 'pizza', 'lut', 'lut_basic') else 'box'

        # Run the chosen algorithm on this frame
        result = self._run_filter_pipeline(algo_key, base_points, car_x, car_y)

        # Log stats for this algorithm, including full pipeline time
        self.get_logger().info(
            f'2. {result["label"]}: ROI={result["after_view"]} pts '
            f'(ROI time: {result["roi_time_ms"]:.3f} ms), '
            f'After Car={result["after_car"]}, After Ground={result["after_ground"]}, '
            f'After Voxel={result["after_voxel"]} '
            f'(Pipeline time: {result["pipeline_time_ms"]:.3f} ms)'
        )

        # Publish filtered cloud(s)
        def publish_result(pub, res, name: str):
            if res['after_voxel'] > 0:
                pts = np.asarray(res['pcd'].points).astype(np.float32)
                msg_out = pc2.create_cloud_xyz32(msg.header, pts)
                pub.publish(msg_out)
                self.get_logger().info(f'Published {name} filtered cloud.')
            else:
                self.get_logger().warn(f'ZERO points remaining after {name} filter')

        # Always publish on the generic /filtered_points topic
        publish_result(self.pub, result, result['label'])

        # Also publish on the algorithm-specific topic for convenience
        if algo_key == 'box':
            publish_result(self.pub_box, result, 'Old Box')
        elif algo_key == 'pizza':
            publish_result(self.pub_pizza, result, 'Pizza Slice')
        elif algo_key == 'lut':
            publish_result(self.pub_lut, result, 'LUT Bounding Box')
        elif algo_key == 'lut_basic':
            publish_result(self.pub_lut, result, 'LUT Basic Bounds')

        # Optionally save this filtered frame to disk once per run
        self._maybe_save_frame(algo_key, result)

        after_car_count = result['after_car']
        after_ground_count = result['after_ground']

        if result['after_voxel'] == 0:
            self.get_logger().error('ZERO points remaining after filter; skipping clustering.')
            end_time = time.perf_counter_ns()
            duration = end_time - start_time
            ns = duration / 1e9
            fps = 1.0 / ns if ns > 0 else 0.0
            # No clustering / classification performed
            result['cluster_time_ms'] = 0.0
            result['classify_time_ms'] = 0.0
            result['initial_pts'] = initial_count
            result['num_clusters'] = 0
            result['total_time_sec'] = ns
            if algo_key == 'lut':
                result['lut_diagnostics'] = self.lut_filter._lut_filter_diagnostics
            # Log to CSV even if there are no points after voxel
            self._log_timing_to_file(result, 0, msg, start_time, end_time)
            self._log_diagnostics_to_file(result, msg)
            self.get_logger().info(f'Total processing Time: {ns} s ({fps:.2f} Hz)')
            return

        # Clustering on filtered cloud
        pts_primary = np.asarray(result['pcd'].points).astype(np.float32)
        df = pd.DataFrame(pts_primary, columns=["X", "Y", "Z"])

        processor = clustering(df)

        cluster_distance_threshold = self.get_parameter('cluster_distance_threshold').get_parameter_value().double_value
        cluster_min_size = self.get_parameter('cluster_min_size').get_parameter_value().integer_value
        cluster_max_size = self.get_parameter('cluster_max_size').get_parameter_value().integer_value

        # Measure pure Euclidean clustering time
        cluster_start = time.perf_counter_ns()
        clusters = processor.euclidean_clustering(
            distance_threshold=cluster_distance_threshold,
            cluster_parameters={"min_size": cluster_min_size, "max_size": cluster_max_size}
        )

        # If using the LUT filter, use current clusters to update the
        # cone-based corridor LUT so that filterWithLUT works across frames.
        num_detected_cones = 0
        if algo_key == 'lut' and len(clusters) > 0:
            cluster_arrays = [np.array(list(idx_set), dtype=int) for idx_set in clusters.values()]
            cones = self.lut_filter.detectCones(result['pcd'], cluster_arrays)
            num_detected_cones = len(cones)
            if cones:
                self.lut_filter.buildLUT(cones)
        cluster_end = time.perf_counter_ns()
        cluster_time_ms = (cluster_end - cluster_start) / 1e6
        result['cluster_time_ms'] = cluster_time_ms
        result['num_detected_cones'] = num_detected_cones

        self.get_logger().info(f'Found {len(clusters)} clusters (objects)')
        # Publish cone markers for clustered objects
        marker_array = MarkerArray()

        safe_lifetime = 0.12

        # Measure cone classification + marker creation/publish time
        classify_start = time.perf_counter_ns()

        # Count how many clusters are classified as cones
        num_cones = 0

        for cluster_id, indices in clusters.items():
            cluster_np = df.iloc[list(indices)][["X", "Y", "Z"]].values.astype(np.float64)
            classified_cones, _ = self.cone.isCone(cluster_np)

            if not classified_cones[0]:
                continue

            num_cones += 1

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_np)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center()

            marker = Marker()
            marker.header = msg.header
            marker.ns = "cone_detection"
            marker.id = int(cluster_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            marker.scale.x, marker.scale.y, marker.scale.z = 0.3, 0.3, 0.3
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 0.0, 1.0

            marker.lifetime = rclpy.duration.Duration(seconds=0, nanoseconds=int(safe_lifetime * 1e9)).to_msg()

            marker_array.markers.append(marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

        classify_end = time.perf_counter_ns()
        classify_time_ms = (classify_end - classify_start) / 1e6
        result['classify_time_ms'] = classify_time_ms
        result['num_cones'] = num_cones

        self.get_logger().info(
            f'Cone classification: {num_cones} cones out of {len(clusters)} clusters'
        )

        # Reduction stats for this algorithm
        redper = 100.0 * (1 - after_ground_count / after_car_count) if after_car_count > 0 else 0
        self.reduction_percentages.append(redper)

        # Overall timing for this frame
        end_time = time.perf_counter_ns()
        duration = end_time - start_time
        ns = duration / 1e9

        fps = 1.0 / ns if ns > 0 else 0.0

        self.get_logger().info(f'Total processing Time: {ns} s ({fps:.2f} Hz)')

        # Log detailed timings to CSV
        self._log_timing_to_file(result, len(clusters), msg, start_time, end_time)
        
        # Store all diagnostics data before logging
        result['initial_pts'] = initial_count
        result['num_clusters'] = len(clusters)
        result['total_time_sec'] = ns
        
        # Capture LUT filter diagnostics if using LUT algorithm
        if algo_key == 'lut':
            result['lut_diagnostics'] = self.lut_filter._lut_filter_diagnostics
        
        # Log diagnostics (point counts, LUT state, clusters, timing) to CSV
        self._log_diagnostics_to_file(result, msg)

        self.processing_times.append(ns)

        if len(self.processing_times) > 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
            fpsmax = 1.0 / min_time if min_time > 0 else 0.0
            fpsmin = 1.0 / max_time if max_time > 0 else 0.0

            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0

            self.get_logger().info(
                f'Overall performance (Last {len(self.processing_times)} frames):\n'
                f'  Min:     {min_time:.4f}s ({fpsmax:.1f} Hz)\n'
                f'  Max:     {max_time:.4f}s ({fpsmin:.1f} Hz)\n'
                f'  Avg:     {avg_time:.4f}s ({avg_fps:.1f} Hz)'
            )


    def _log_timing_to_file(self, result, num_clusters: int, msg: PointCloud2, start_time_ns: int, end_time_ns: int) -> None:
        """Append per-frame timing information to a CSV file.

        Columns: frame, stamp_sec, algo, label, roi_time_ms, pipeline_time_ms,
        cluster_time_ms, classify_time_ms, total_time_ms, after_view, after_car,
        after_ground, after_voxel, num_clusters, num_cones
        """
        log_path = self.timing_log_path
        if not log_path:
            return

        # Determine if we need to write the header
        header_needed = not os.path.exists(log_path)

        # ROS message timestamp
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) / 1e9

        total_time_ms = (end_time_ns - start_time_ns) / 1e6

        self.frame_index += 1

        with open(log_path, 'a') as f:
            if header_needed:
                f.write(
                    'frame,stamp_sec,algo,label,roi_time_ms,pipeline_time_ms,cluster_time_ms,classify_time_ms,total_time_ms,'
                    'after_view,after_car,after_ground,after_voxel,num_clusters,num_cones\n'
                )

            f.write(
                f'{self.frame_index},{stamp_sec:.6f},{result["algo_key"]},"{result["label"]}",'  # noqa: E501
                f'{result["roi_time_ms"]:.3f},{result["pipeline_time_ms"]:.3f},{result.get("cluster_time_ms", 0.0):.3f},{result.get("classify_time_ms", 0.0):.3f},{total_time_ms:.3f},'  # noqa: E501
                f'{result["after_view"]},{result["after_car"]},{result["after_ground"]},{result["after_voxel"]},{num_clusters},{result.get("num_cones", 0)}\n'  # noqa: E501
            )

    def _log_diagnostics_to_file(self, result: dict, msg: PointCloud2) -> None:
        """Log detailed diagnostics (point counts, LUT state, clustering, timing) to CSV.
        
        Tracks progression and bottlenecks: LUT size, cluster count, cone count, processing times.
        Columns: frame,stamp_sec,algo,initial_pts,after_roi,after_car,after_ground,
                 after_voxel,num_clusters,num_detected_cones,num_classified_cones,lut_size,cluster_time_ms,
                 classify_time_ms,total_time_ms,lut_input,lut_corridor_pass,lut_bounds_pass,lut_final
        """
        try:
            log_path = self.diagnostics_log_path
            if not log_path:
                self.get_logger().warn('Diagnostics log path not set, skipping diagnostics logging')
                return

            # Determine if we need to write the header
            header_needed = not os.path.exists(log_path)

            # ROS message timestamp
            stamp = msg.header.stamp
            stamp_sec = float(stamp.sec) + float(stamp.nanosec) / 1e9

            with open(log_path, 'a') as f:
                if header_needed:
                    f.write(
                        'frame,stamp_sec,algo,initial_pts,after_roi,after_car,after_ground,after_voxel,'
                        'num_clusters,num_detected_cones,num_classified_cones,lut_size,cluster_time_ms,'
                        'classify_time_ms,total_time_ms,lut_input,lut_corridor_pass,lut_bounds_pass,lut_final\n'
                    )

                # Extract timing diagnostics
                cluster_time = result.get('cluster_time_ms', 0.0)
                classify_time = result.get('classify_time_ms', 0.0)
                num_detected_cones = result.get('num_detected_cones', 0)
                num_classified_cones = result.get('num_cones', 0)
                num_clusters = result.get('num_clusters', 0)
                total_time_sec = result.get('total_time_sec', 0.0)
                total_time_ms = total_time_sec * 1000.0

                # Extract LUT diagnostics if available
                lut_input = lut_corridor = lut_bounds = lut_final = lut_size = 0
                if 'lut_diagnostics' in result:
                    diag = result['lut_diagnostics']
                    lut_input = diag.get('input_count', 0)
                    lut_corridor = diag.get('corridor_pass', 0)
                    lut_bounds = diag.get('bound_pass', 0)
                    lut_final = diag.get('final_output', 0)
                    lut_size = diag.get('lut_size', 0)

                f.write(
                    f'{self.frame_index},{stamp_sec:.6f},{result["algo_key"]},'
                    f'{result.get("initial_pts", 0)},{result["after_view"]},{result["after_car"]},'
                    f'{result["after_ground"]},{result["after_voxel"]},'
                    f'{num_clusters},{num_detected_cones},{num_classified_cones},{lut_size},'
                    f'{cluster_time:.3f},{classify_time:.3f},{total_time_ms:.3f},'
                    f'{lut_input},{lut_corridor},{lut_bounds},{lut_final}\n'
                )
                f.flush()
        except Exception as e:
            self.get_logger().error(f'Failed to log diagnostics to file: {e}')

    def _maybe_save_frame(self, algo_key: str, result: dict) -> None:
        """Save filtered frames to PCD files if configured.

        If save_frame_limit == 1 (default), this behaves like the
        original single-frame export (one file per run named by
        algorithm). If save_frame_limit > 1, multiple frames are saved
        with an index suffix.
        """
        if not self.save_frame_dir:
            return
        # Only save if there are points after voxel filtering
        if result.get('after_voxel', 0) == 0:
            return

        # Respect the maximum number of frames to save
        if self.save_frame_limit > 0 and self._frames_saved >= self.save_frame_limit:
            return

        try:
            os.makedirs(self.save_frame_dir, exist_ok=True)

            if self.save_frame_limit <= 1:
                # Backwards-compatible single-frame filename
                filename = f'frame_{algo_key}.pcd'
            else:
                # Multi-frame export with zero-padded index
                filename = f'frame_{algo_key}_{self._frames_saved:03d}.pcd'

            path = os.path.join(self.save_frame_dir, filename)
            o3d.io.write_point_cloud(path, result['pcd'])
            self._frames_saved += 1
            self.get_logger().info(f'Saved filtered frame to: {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save frame PCD: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
