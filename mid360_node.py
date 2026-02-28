import rclpy 
from rclpy.node import Node

import numpy as np
import open3d as o3d
import time

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import sensor_msgs_py.point_cloud2 as pc2


class preprocessConfig:
    # MAP_PATH = ''
    # CAD_UNIT = 0.001

    X_MIN, X_MAX = -3.0, 3.0
    Y_MIN, Y_MAX = -0.5, 5.0
    Z_MIN, Z_MAX = 0.0, 1.0

    # ICP_MAX_CORRESPONDENCE_DISTANCE = 0.5
    # ICP_MAX_ITERATIONS = 50
    # ICP_RELATIVE_FITNESS = 1e-6
    # ICP_RELATIVE_RMSE = 1e-6

    SUBTRACTION_RADIUS = 0.5

    VOXEL_SIZE = 0.05

    GROUND_Z_FALLBACK = 0.05
    GROUND_SEARCH_BAND =0.03
    GROUND_Z_THRESHOLD = 0.05

    SOR_K_NEIGHBORS = 30
    SOR_STD_RATIO = 1.2

def ros2_poinycloud2_to_open3d(msg: PointCloud2)-> o3d.geometry.PointCloud:
    # Convert ROS PointCloud2 to Open3D PointCloud
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    return pcd

def open3d_to_ros_pointcloud2(pcd: o3d.geometry.PointCloud, frame_id: str , stamp) -> PointCloud2:
    # Convert Open3D PointCloud to ROS PointCloud2
    points = np.asarray(pcd.points, dtype=np.float32)
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud_xyz32(header, points.tolist())

def passthrough_filter(pcd: o3d.geometry.PointCloud, cfg: preprocessConfig) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    
    mask = (
        (pts[:, 0] >= cfg.X_MIN) & (pts[:, 0] <= cfg.X_MAX) &
        (pts[:, 1] >= cfg.Y_MIN) & (pts[:, 1] <= cfg.Y_MAX) &
        (pts[:, 2] >= cfg.Z_MIN) & (pts[:, 2] <= cfg.Z_MAX)
    )

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    return filtered

def voxel_downsample(pcd: o3d.geometry.PointCloud, config: preprocessConfig) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE)

def ground_removal(pcd: o3d.geometry.PointCloud, cfg: preprocessConfig) -> tuple:
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd, o3d.geometry.PointCloud()

    above_mask = pts[:, 2] > cfg.GROUND_Z_THRESHOLD
    ground_mask = ~above_mask

    above = o3d.geometry.PointCloud()
    above.points = o3d.utility.Vector3dVector(pts[above_mask])

    ground = o3d.geometry.PointCloud()
    ground.points = o3d.utility.Vector3dVector(pts[ground_mask])

    return above, ground

def statistical_outlier_removal(pcd: o3d.geometry.PointCloud, cfg: preprocessConfig) -> o3d.geometry.PointCloud:
    if len(pcd.points) < cfg.SOR_K_NEIGHBORS:
        return pcd

    clean_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.SOR_K_NEIGHBORS,
        std_ratio=cfg.SOR_STD_RATIO
    )
    return clean_pcd

def preprocess_pipeline(raw_pcd: o3d.geometry.PointCloud,cfg: preprocessConfig, verbose: bool=False) -> dict:
    def log(stage, pcd):
        if verbose:
            print(f"[{stage:20s}] {len(pcd.points):>6} pts")
    
    log("Input", raw_pcd)

    pcd = passthrough_filter(raw_pcd, cfg)
    log("Passthrough", pcd)

    pcd = voxel_downsample(pcd, cfg)
    log("Voxel Downsample", pcd)

    pcd, ground_pcd = ground_removal(pcd, cfg)
    log("Ground Removal", pcd)

    pcd = statistical_outlier_removal(pcd, cfg)
    log("Statistical Outlier Removal", pcd)

    return {
        "processed": pcd,
        "ground": ground_pcd,
        "raw_count": len(raw_pcd.points),
        "final_count":len(pcd.points)
    }

# ROS2 Node

class PreprocessingNode(Node):
    def __init__(self):
        super().__init__('preprocessing_node')
        self.cfg = preprocessConfig()

        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/preprocessed/lidar')
        self.declare_parameter('verbose', False)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.cloud_callback,
            10)
        
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

    def cloud_callback(self, msg):

        raw_pcd = ros2_poinycloud2_to_open3d(msg)

        if len(raw_pcd.points) == 0:
            self.get_logger().warn("Received empty point cloud")
            return
        
        result = preprocess_pipeline(raw_pcd, self.cfg, verbose=self.verbose)

        if self.verbose:
            self.get_logger().info(
                f"Processing: {result['raw_count']} -> {result['final_count']} points"
                f"{100*result['final_count']/max(result['raw_count'],1):.1f}%"
            )

        out_msg = open3d_to_ros_pointcloud2(
            result['processed'], 
            frame_id=msg.header.frame_id, 
            stamp=msg.header.stamp
        )
        self.pub.publish(out_msg)
        


def main(args=None):
    rclpy.init(args=args)
    preprocessing_node = PreprocessingNode()
    
    try:
        rclpy.spin(preprocessing_node)
    except KeyboardInterrupt:
        pass
    finally:
        preprocessing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()