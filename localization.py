"""
ABU Robocon 2026 - Kung Fu Quest
R2 Localization — ICP against CAD map (gamefield.pcd)

Estimates full 3D pose (x, y, z, roll, pitch, yaw) of R2
inside the Meihua Forest using Point-to-Plane ICP.

═══════════════════════════════════════════════════════════════
  COORDINATE FRAME  (matches lidar_preprocessing.py)
═══════════════════════════════════════════════════════════════
  ROS X  = CAD_Z / 1000              (forward,   0 – 6.235 m)
  ROS Y  = CAD_X / 1000              (lateral,   0 – 12.10 m)
  ROS Z  = (2470 - CAD_Y) / 1000 - 0.171  (up, floor = 0)

  Floor      = Z  0.000 m
  Block tops = Z  0.363 m  ← KFS sit here
  Walls top  = Z  0.484 m

  Start zone (Martial Club / R2 entry): X ≈ 0.0 m
  Forest entry blocks 1-3:              X ≈ 1.25 m
  Forest exit  blocks 10-12:            X ≈ 3.43 m
═══════════════════════════════════════════════════════════════

Pipeline (runs at LiDAR rate, 10 Hz):
    Raw Livox cloud
        → Passthrough filter (full field volume)
        → Voxel downsample
        → Point-to-Plane ICP against CAD map
        → Extract 6-DOF pose from 4×4 transform
        → Publish PoseStamped + TF

Startup:
    1. Load gamefield.pcd, remap, offset → reference map
    2. Set initial pose from known start zone (~10cm accuracy)
    3. First ICP run with coarse correspondence distance
       to snap to correct position
    4. All subsequent frames use previous pose as warm start
       with tight correspondence distance → fast convergence

Requirements:
    pip install open3d numpy rclpy sensor_msgs geometry_msgs tf2_ros

Usage:
    # Standalone test (no ROS2):
    python3 localization.py

    # ROS2 node:
    ros2 run your_package localization --ros-args \\
        -p map_path:=/path/to/gamefield.pcd \\
        -p verbose:=true
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationConfig:

    # ── CAD map ───────────────────────────────────────────────────────────────
    MAP_PATH        = "gamefield.pcd"
    CAD_SCALE       = 0.001              # mm → m
    CAD_Y_MAX       = 2470.0             # mm — used for flip
    GROUND_OFFSET_Z = 0.171              # m  — floor at Z=0 after flip

    # ── Map preprocessing ─────────────────────────────────────────────────────
    MAP_VOXEL_SIZE  = 0.05               # 5cm voxel for reference map
    LIVE_VOXEL_SIZE = 0.05               # 5cm voxel for live scan

    # ── Passthrough filter (full field, ROS frame after offset) ───────────────
    # Wider than preprocessing — we want all structural features for ICP
    X_MIN, X_MAX =  0.0,   6.5
    Y_MIN, Y_MAX =  0.0,  12.5
    Z_MIN, Z_MAX = -0.10,  0.70         # floor to above block tops

    # ── Initial pose (R2 start zone, Martial Club) ────────────────────────────
    # R2 starts at X≈0, center of team's field half
    # Adjust Y to your team's side: ~3.0m (team A) or ~9.0m (team B)
    INIT_X   =  0.10                    # m — just inside field
    INIT_Y   =  6.05                    # m — center of field width (tune to your side)
    INIT_Z   =  0.00                    # m — floor level
    INIT_ROLL  = 0.0                    # rad
    INIT_PITCH = 0.0                    # rad
    INIT_YAW   = 0.0                    # rad — facing +X (into field)

    # ── ICP — first frame (coarse, snaps from ~10cm initial error) ────────────
    ICP_INIT_MAX_CORR  = 0.50           # 50cm — wide net for first frame
    ICP_INIT_MAX_ITER  = 100

    # ── ICP — subsequent frames (tight, warm-started from previous pose) ──────
    ICP_TRACK_MAX_CORR = 0.15           # 15cm — tight for tracking
    ICP_TRACK_MAX_ITER = 30

    # ── ICP convergence ───────────────────────────────────────────────────────
    ICP_REL_FITNESS    = 1e-6
    ICP_REL_RMSE       = 1e-6

    # ── Pose quality thresholds ───────────────────────────────────────────────
    MIN_FITNESS        = 0.30           # below this → pose unreliable, warn
    MAX_RMSE           = 0.10           # above this → pose unreliable, warn

    # ── ROS topics ────────────────────────────────────────────────────────────
    INPUT_TOPIC        = "/livox/lidar"
    POSE_TOPIC         = "/r2/pose"
    MAP_FRAME          = "map"
    ROBOT_FRAME        = "r2_base_link"


# ─────────────────────────────────────────────────────────────────────────────
#  CAD MAP  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceMap:
    """
    Loads gamefield.pcd, remaps to ROS frame, applies ground offset,
    estimates normals (required for Point-to-Plane ICP), and crops
    to the working volume.
    """

    def __init__(self, cfg: LocalizationConfig):
        self.cfg = cfg
        self.pcd     = None   # full reference map with normals
        self._loaded = False

    def load(self, map_path: str = None) -> bool:
        path = map_path or self.cfg.MAP_PATH
        print(f"[ReferenceMap] Loading: {path}")

        try:
            raw = o3d.io.read_point_cloud(path)
        except Exception as e:
            print(f"[ReferenceMap] ERROR: {e}")
            return False

        if len(raw.points) == 0:
            print("[ReferenceMap] ERROR: empty cloud.")
            return False

        print(f"[ReferenceMap] {len(raw.points)} pts (raw CAD mm)")

        # ── Remap CAD → ROS frame ─────────────────────────────────────────────
        cad = np.asarray(raw.points)
        ros = np.zeros_like(cad, dtype=np.float32)
        ros[:, 0] = cad[:, 2] * self.cfg.CAD_SCALE                          # X = CAD_Z
        ros[:, 1] = cad[:, 0] * self.cfg.CAD_SCALE                          # Y = CAD_X
        ros[:, 2] = (self.cfg.CAD_Y_MAX - cad[:, 1]) * self.cfg.CAD_SCALE   # Z = flip(CAD_Y)
        ros[:, 2] -= self.cfg.GROUND_OFFSET_Z                                # floor = 0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ros)

        # ── Voxel downsample ──────────────────────────────────────────────────
        pcd = pcd.voxel_down_sample(self.cfg.MAP_VOXEL_SIZE)

        # ── Crop to working volume ────────────────────────────────────────────
        pcd = self._crop(pcd)
        print(f"[ReferenceMap] After crop: {len(pcd.points)} pts")

        # ── Estimate surface normals (required for Point-to-Plane ICP) ────────
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.20,   # 20cm search radius
                max_nn=30
            )
        )
        # Orient normals consistently (toward the sensor / upward)
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([3.0, 6.0, 5.0])   # above field center
        )

        self.pcd     = pcd
        self._loaded = True
        print("[ReferenceMap] Ready.\n")
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def _crop(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        c   = self.cfg
        mask = (
            (pts[:, 0] >= c.X_MIN) & (pts[:, 0] <= c.X_MAX) &
            (pts[:, 1] >= c.Y_MIN) & (pts[:, 1] <= c.Y_MAX) &
            (pts[:, 2] >= c.Z_MIN) & (pts[:, 2] <= c.Z_MAX)
        )
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[mask])
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  POSE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """Build a 4×4 homogeneous transform from (x,y,z,roll,pitch,yaw)."""
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def matrix_to_xyzrpy(T: np.ndarray) -> dict:
    """Extract (x, y, z, roll, pitch, yaw) from a 4×4 transform."""
    xyz  = T[:3, 3]
    rpy  = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    return {
        'x':     float(xyz[0]),
        'y':     float(xyz[1]),
        'z':     float(xyz[2]),
        'roll':  float(rpy[0]),
        'pitch': float(rpy[1]),
        'yaw':   float(rpy[2]),
    }


def matrix_to_quaternion(T: np.ndarray) -> np.ndarray:
    """Return quaternion [qx, qy, qz, qw] from a 4×4 transform."""
    return Rotation.from_matrix(T[:3, :3]).as_quat()   # [x, y, z, w]


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALIZER
# ─────────────────────────────────────────────────────────────────────────────

class Localizer:
    """
    Maintains robot pose estimate and updates it via ICP at each scan.

    State machine:
        INIT    — waiting for first valid ICP result
        TRACKING — ICP warm-started from previous pose each frame
        LOST    — ICP fitness dropped below threshold, needs recovery
    """

    STATE_INIT     = "INIT"
    STATE_TRACKING = "TRACKING"
    STATE_LOST     = "LOST"

    def __init__(self, ref_map: ReferenceMap, cfg: LocalizationConfig):
        self.ref_map   = ref_map
        self.cfg       = cfg
        self.state     = self.STATE_INIT
        self.pose_T    = xyzrpy_to_matrix(    # current best pose estimate
            cfg.INIT_X, cfg.INIT_Y, cfg.INIT_Z,
            cfg.INIT_ROLL, cfg.INIT_PITCH, cfg.INIT_YAW
        )
        self.last_fitness = 0.0
        self.last_rmse    = 999.0
        self.frame_count  = 0

    def update(self, live_pcd: o3d.geometry.PointCloud) -> dict:
        """
        Run ICP with the live scan, update pose, return result dict.

        Args:
            live_pcd : preprocessed live point cloud (ROS frame, meters)

        Returns dict:
            'pose'      : {'x','y','z','roll','pitch','yaw'}
            'T'         : 4×4 transform matrix
            'fitness'   : ICP fitness (0–1, higher=better)
            'rmse'      : ICP inlier RMSE (lower=better)
            'state'     : current state string
            'reliable'  : bool — True if pose estimate is trustworthy
        """
        self.frame_count += 1

        # Preprocess live scan
        live = self._prepare_live(live_pcd)
        if live is None or len(live.points) < 50:
            return self._result(reliable=False, reason="too few points after prep")

        # Choose ICP parameters based on state
        if self.state == self.STATE_INIT:
            max_corr = self.cfg.ICP_INIT_MAX_CORR
            max_iter = self.cfg.ICP_INIT_MAX_ITER
        else:
            max_corr = self.cfg.ICP_TRACK_MAX_CORR
            max_iter = self.cfg.ICP_TRACK_MAX_ITER

        # Run Point-to-Plane ICP
        result = o3d.pipelines.registration.registration_icp(
            source=live,
            target=self.ref_map.pcd,
            max_correspondence_distance=max_corr,
            init=self.pose_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=self.cfg.ICP_REL_FITNESS,
                relative_rmse=self.cfg.ICP_REL_RMSE
            )
        )

        self.last_fitness = result.fitness
        self.last_rmse    = result.inlier_rmse

        # Check quality
        reliable = (
            result.fitness >= self.cfg.MIN_FITNESS and
            result.inlier_rmse <= self.cfg.MAX_RMSE
        )

        if reliable:
            self.pose_T = result.transformation
            self.state  = self.STATE_TRACKING
        else:
            # Don't update pose — keep last good estimate
            if self.state == self.STATE_TRACKING:
                self.state = self.STATE_LOST

        return self._result(reliable=reliable)

    def reset_to_start(self):
        """Reset pose to initial start zone estimate. Call on retry."""
        cfg = self.cfg
        self.pose_T = xyzrpy_to_matrix(
            cfg.INIT_X, cfg.INIT_Y, cfg.INIT_Z,
            cfg.INIT_ROLL, cfg.INIT_PITCH, cfg.INIT_YAW
        )
        self.state = self.STATE_INIT
        print("[Localizer] Reset to start zone pose.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare_live(self, pcd: o3d.geometry.PointCloud):
        """Downsample live scan and estimate normals for Point-to-Plane ICP."""
        if len(pcd.points) == 0:
            return None

        pcd = pcd.voxel_down_sample(self.cfg.LIVE_VOXEL_SIZE)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.20, max_nn=30
            )
        )
        return pcd

    def _result(self, reliable: bool, reason: str = "") -> dict:
        pose = matrix_to_xyzrpy(self.pose_T)
        if not reliable and reason:
            print(f"[Localizer] Unreliable: {reason}")
        return {
            'pose':     pose,
            'T':        self.pose_T.copy(),
            'fitness':  self.last_fitness,
            'rmse':     self.last_rmse,
            'state':    self.state,
            'reliable': reliable,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE SCAN PREPROCESSING  (lightweight version for localization)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_live_scan(raw_pcd: o3d.geometry.PointCloud,
                      cfg: LocalizationConfig) -> o3d.geometry.PointCloud:
    """
    Passthrough filter on live scan.
    Keep floor + walls + blocks — all structural features help ICP.
    Do NOT remove ground here (ground plane is a useful ICP feature).
    """
    pts = np.asarray(raw_pcd.points)
    if pts.shape[0] == 0:
        return raw_pcd

    mask = (
        (pts[:, 0] >= cfg.X_MIN) & (pts[:, 0] <= cfg.X_MAX) &
        (pts[:, 1] >= cfg.Y_MIN) & (pts[:, 1] <= cfg.Y_MAX) &
        (pts[:, 2] >= cfg.Z_MIN) & (pts[:, 2] <= cfg.Z_MAX)
    )
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationNode(Node):
    """
    ROS2 node:
        Subscribes : /livox/lidar        (raw Livox PointCloud2)
        Publishes  : /r2/pose            (geometry_msgs/PoseStamped)
        Broadcasts : TF  map → r2_base_link
    """

    def __init__(self):
        super().__init__('r2_localization_node')

        self.cfg = LocalizationConfig()

        self.declare_parameter('map_path',  self.cfg.MAP_PATH)
        self.declare_parameter('init_x',    self.cfg.INIT_X)
        self.declare_parameter('init_y',    self.cfg.INIT_Y)
        self.declare_parameter('init_yaw',  self.cfg.INIT_YAW)
        self.declare_parameter('verbose',   False)

        self.cfg.MAP_PATH = self.get_parameter('map_path').value
        self.cfg.INIT_X   = self.get_parameter('init_x').value
        self.cfg.INIT_Y   = self.get_parameter('init_y').value
        self.cfg.INIT_YAW = self.get_parameter('init_yaw').value
        self.verbose      = self.get_parameter('verbose').value

        # Load reference map
        self.ref_map = ReferenceMap(self.cfg)
        if not self.ref_map.load():
            self.get_logger().error("Failed to load reference map. Shutting down.")
            raise RuntimeError("Map load failed.")

        # Create localizer
        self.localizer = Localizer(self.ref_map, self.cfg)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, self.cfg.POSE_TOPIC, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscriber
        self.sub = self.create_subscription(
            PointCloud2,
            self.cfg.INPUT_TOPIC,
            self.cloud_callback,
            10
        )

        self.get_logger().info(
            f"R2 Localization ready\n"
            f"  Map    : {self.cfg.MAP_PATH}\n"
            f"  Input  : {self.cfg.INPUT_TOPIC}\n"
            f"  Output : {self.cfg.POSE_TOPIC}\n"
            f"  Init   : x={self.cfg.INIT_X:.2f} y={self.cfg.INIT_Y:.2f} yaw={self.cfg.INIT_YAW:.2f}"
        )

    def cloud_callback(self, msg: PointCloud2):
        # Convert to Open3D
        pts = [
            [p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
        ]
        if not pts:
            return

        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float32))

        # Passthrough filter
        live_pcd = prepare_live_scan(raw_pcd, self.cfg)

        # Localize
        result = self.localizer.update(live_pcd)

        if self.verbose:
            p = result['pose']
            self.get_logger().info(
                f"[{result['state']:8s}] "
                f"x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f} "
                f"yaw={np.degrees(p['yaw']):.1f}°  "
                f"fit={result['fitness']:.3f}  rmse={result['rmse']:.4f}m"
            )

        if not result['reliable']:
            self.get_logger().warn(
                f"Pose unreliable! fitness={result['fitness']:.3f} rmse={result['rmse']:.4f}"
            )

        # Publish pose
        self._publish_pose(result, msg.header.stamp)

    def _publish_pose(self, result: dict, stamp):
        p   = result['pose']
        quat = matrix_to_quaternion(result['T'])

        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = stamp
        pose_msg.header.frame_id = self.cfg.MAP_FRAME
        pose_msg.pose.position.x = p['x']
        pose_msg.pose.position.y = p['y']
        pose_msg.pose.position.z = p['z']
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # TF: map → r2_base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = stamp
        tf_msg.header.frame_id = self.cfg.MAP_FRAME
        tf_msg.child_frame_id  = self.cfg.ROBOT_FRAME
        tf_msg.transform.translation.x = p['x']
        tf_msg.transform.translation.y = p['y']
        tf_msg.transform.translation.z = p['z']
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(tf_msg)


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST  (no ROS2 needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_localization(map_path: str = "gamefield.pcd"):
    """
    Validates the full localization pipeline without ROS2.

    Test cases:
        1. Correct initial pose → should converge immediately
        2. Perturbed pose (+15cm, +10cm, +5° yaw) → should recover
        3. Large perturbation (+50cm) → should still converge with coarse ICP
    """
    import os
    if not os.path.exists(map_path):
        print(f"[!] {map_path} not found. Place it in the same folder.")
        return

    print("=" * 60)
    print("ABU Robocon 2026 — R2 Localization Test")
    print("=" * 60)

    cfg = LocalizationConfig()
    cfg.MAP_PATH = map_path

    # Load reference map
    ref_map = ReferenceMap(cfg)
    if not ref_map.load():
        return

    # Sample a patch of the reference map as a synthetic live scan
    # (simulates what the Livox would see from a known position)
    map_pts = np.asarray(ref_map.pcd.points)

    def make_live_scan(true_x, true_y, true_yaw, noise_std=0.01) -> o3d.geometry.PointCloud:
        """Simulate a live scan from a given pose by sampling nearby map points."""
        rng = np.random.default_rng(42)
        # Take points within 3m of the true position
        dists = np.sqrt((map_pts[:,0]-true_x)**2 + (map_pts[:,1]-true_y)**2)
        nearby = map_pts[dists < 3.0]
        # Add sensor noise
        noisy = nearby + rng.normal(0, noise_std, nearby.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(noisy.astype(np.float32))
        return pcd

    # ── Test cases ────────────────────────────────────────────────────────────
    true_x, true_y = 2.0, 6.0   # somewhere in the middle of the field

    test_cases = [
        {
            'name'  : "Correct initial pose",
            'init_x': true_x,
            'init_y': true_y,
            'init_yaw': 0.0,
        },
        {
            'name'  : "Small perturbation (+15cm X, +10cm Y, +5° yaw)",
            'init_x': true_x + 0.15,
            'init_y': true_y + 0.10,
            'init_yaw': np.radians(5),
        },
        {
            'name'  : "Large perturbation (+50cm X, +30cm Y)",
            'init_x': true_x + 0.50,
            'init_y': true_y + 0.30,
            'init_yaw': np.radians(10),
        },
    ]

    live_pcd = make_live_scan(true_x, true_y, 0.0)
    print(f"Simulated live scan: {len(live_pcd.points)} pts")
    print(f"True pose: x={true_x:.3f} y={true_y:.3f} yaw=0.0°\n")

    for tc in test_cases:
        print(f"── {tc['name']} ──")

        # Create fresh localizer with this test's initial pose
        cfg.INIT_X   = tc['init_x']
        cfg.INIT_Y   = tc['init_y']
        cfg.INIT_YAW = tc['init_yaw']
        localizer = Localizer(ref_map, cfg)

        # Run ICP
        result = localizer.update(live_pcd)
        p = result['pose']

        # Error vs true pose
        err_x   = abs(p['x'] - true_x)
        err_y   = abs(p['y'] - true_y)
        err_yaw = abs(np.degrees(p['yaw']))

        print(f"  Result : x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f} "
              f"yaw={np.degrees(p['yaw']):.1f}°")
        print(f"  Error  : Δx={err_x:.3f}m  Δy={err_y:.3f}m  Δyaw={err_yaw:.1f}°")
        print(f"  Quality: fitness={result['fitness']:.3f}  "
              f"rmse={result['rmse']:.4f}m  state={result['state']}")
        converged = result['reliable'] and err_x < 0.05 and err_y < 0.05
        print(f"  {'✓ CONVERGED' if converged else '✗ DID NOT CONVERGE'}\n")

    # ── Visualize final test case ──────────────────────────────────────────────
    print("Opening visualization (last test case)...")
    print("  Gray  = reference map")
    print("  Green = live scan at true pose")
    print("  Red   = live scan at initial (perturbed) pose")
    print("  Blue  = live scan after ICP correction\n")

    # Reference map
    ref_vis = ref_map.pcd
    ref_vis.paint_uniform_color([0.6, 0.6, 0.6])

    # Live scan at true pose
    live_true = make_live_scan(true_x, true_y, 0.0)
    live_true.paint_uniform_color([0.1, 0.9, 0.1])

    # Live scan at perturbed initial pose (last test case)
    T_init = xyzrpy_to_matrix(
        tc['init_x'], tc['init_y'], 0.0,
        0.0, 0.0, tc['init_yaw']
    )
    live_perturbed = make_live_scan(tc['init_x'], tc['init_y'], tc['init_yaw'])
    live_perturbed.paint_uniform_color([0.9, 0.1, 0.1])

    # Live scan after ICP (at corrected pose)
    live_corrected = make_live_scan(true_x, true_y, 0.0)
    live_corrected = live_corrected.transform(result['T'])
    live_corrected.paint_uniform_color([0.1, 0.3, 0.9])

    o3d.visualization.draw_geometries(
        [ref_vis, live_true, live_perturbed, live_corrected],
        window_name=(
            "ABU 2026 Localization | "
            "Gray=map  Green=true  Red=initial  Blue=ICP corrected"
        ),
        width=1400, height=800
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main_ros2():
    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    test_localization("gamefield.pcd")

    # To run as ROS2 node, comment above and uncomment:
    # main_ros2()