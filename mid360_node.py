"""
ABU Robocon 2026 - Kung Fu Quest
LiDAR Preprocessing Pipeline — using real gamefield.pcd
Robot 2 (R2) - Meihua Forest KFS Detection

═══════════════════════════════════════════════════════
  CAD MAP COORDINATE FRAME (gamefield.pcd)
═══════════════════════════════════════════════════════
  Units   : millimeters (mm)
  CAD X   : 0 to 12100 mm  → field width  (remapped to ROS Y)
  CAD Y   : 0 to 2470  mm  → vertical up  (remapped to ROS Z)
  CAD Z   : 0 to 6235  mm  → field depth  (remapped to ROS X)

  ROS frame after remapping (meters):
    ROS X  = CAD_Z / 1000   (forward,  0 to  6.235 m)
    ROS Y  = CAD_X / 1000   (lateral,  0 to 12.100 m)
    ROS Z  = CAD_Y / 1000   (up,       0 to  2.470 m)

  Key Z heights (ROS frame, before ground offset):
    Z = 2.020 m  →  field floor surface
    Z = 2.420 m  →  block top surface  (KFS sit here)
    Z = 2.470 m  →  boundary walls top
    Block height above floor = 0.400 m

  After ground offset (subtract 2.020 m from Z):
    Z = 0.000 m  →  floor
    Z = 0.400 m  →  block tops
    Z = 0.450 m  →  boundary walls top

  Forest region (ROS Y):  ~3.1 m  to  ~8.2 m
  Full field split: team A Y=0-6.05m, team B Y=6.05-12.1m
═══════════════════════════════════════════════════════

Pipeline:
    [ONCE]  Load gamefield.pcd
            -> remap CAD -> ROS frame (mm to m)
            -> apply ground offset
            -> crop to forest volume
            -> downsample
            -> build KDTree for background subtraction
    [10Hz]  Receive Livox PointCloud2
            -> passthrough filter
            -> ICP registration to CAD map
            -> background subtraction
            -> voxel downsample
            -> ground removal
            -> statistical outlier removal
            -> publish clean cloud (KFS candidates only)

Requirements:
    pip install open3d numpy rclpy sensor_msgs
"""

import numpy as np
import open3d as o3d
# import rclpy
# from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header
# import sensor_msgs_py.point_cloud2 as pc2


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — values derived from gamefield.pcd analysis
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessConfig:

    # ── CAD map ───────────────────────────────────────────────────────────────
    MAP_PATH        = "gamefield.pcd"
    CAD_SCALE       = 0.001      # mm -> m

    # Ground offset: floor is at CAD_Y=2020mm -> ROS_Z=2.020m
    # Subtract this so floor = Z=0 in robot frame
    GROUND_OFFSET_Z = 2.020      # meters

    # ── Passthrough filter bounds (meters, ROS frame after ground offset) ─────
    # Forest region is ROS_Y ~ 3.1 to 8.2 m
    # Adjust Y_MIN/Y_MAX to your team's side of the field
    X_MIN, X_MAX =  0.0,   6.5   # field depth  (forward)
    Y_MIN, Y_MAX =  3.0,   8.5   # forest width (lateral)
    Z_MIN, Z_MAX = -0.05,  0.60  # floor to above KFS (block tops at 0.40m)

    # ── ICP registration ──────────────────────────────────────────────────────
    ICP_MAX_CORRESPONDENCE = 0.15
    ICP_MAX_ITERATIONS     = 50
    ICP_RELATIVE_FITNESS   = 1e-5
    ICP_RELATIVE_RMSE      = 1e-5

    # ── Background subtraction ────────────────────────────────────────────────
    SUBTRACTION_RADIUS = 0.05    # 5cm — covers Livox noise + field tolerances

    # ── Voxel downsampling ────────────────────────────────────────────────────
    VOXEL_SIZE = 0.03            # 3cm

    # ── Ground removal ────────────────────────────────────────────────────────
    GROUND_Z_THRESHOLD = 0.03   # 3cm above floor

    # ── Statistical Outlier Removal ───────────────────────────────────────────
    SOR_K_NEIGHBORS = 30
    SOR_STD_RATIO   = 1.2


# ─────────────────────────────────────────────────────────────────────────────
#  CAD MAP LOADER
# ─────────────────────────────────────────────────────────────────────────────

class CADMap:
    """
    Loads gamefield.pcd, remaps from CAD frame (mm, Y-up) to ROS frame (m, Z-up),
    applies ground offset so floor=0, crops to forest, builds KDTree.
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg        = cfg
        self.forest_map = None
        self.map_kdtree = None
        self._loaded    = False

    def load(self, map_path: str = None) -> bool:
        path = map_path or self.cfg.MAP_PATH
        print(f"[CADMap] Loading: {path}")

        try:
            raw = o3d.io.read_point_cloud(path)
        except Exception as e:
            print(f"[CADMap] ERROR: {e}")
            return False

        if len(raw.points) == 0:
            print("[CADMap] ERROR: empty cloud.")
            return False

        print(f"[CADMap] Loaded {len(raw.points)} points (CAD mm)")

        # Step 1: Remap CAD frame -> ROS frame, mm -> m
        #   CAD: X=width, Y=up, Z=depth
        #   ROS: X=forward, Y=lateral, Z=up
        cad = np.asarray(raw.points)
        ros = np.zeros_like(cad)
        ros[:, 0] = cad[:, 2] * self.cfg.CAD_SCALE   # ROS X = CAD Z
        ros[:, 1] = cad[:, 0] * self.cfg.CAD_SCALE   # ROS Y = CAD X
        ros[:, 2] = cad[:, 1] * self.cfg.CAD_SCALE   # ROS Z = CAD Y

        print(f"[CADMap] Remapped to ROS meters:")
        print(f"         X: {ros[:,0].min():.3f} to {ros[:,0].max():.3f} m  (fwd)")
        print(f"         Y: {ros[:,1].min():.3f} to {ros[:,1].max():.3f} m  (lat)")
        print(f"         Z: {ros[:,2].min():.3f} to {ros[:,2].max():.3f} m  (up)")

        # Step 2: Ground offset — shift Z so floor = 0
        ros[:, 2] -= self.cfg.GROUND_OFFSET_Z
        block_top = 2.420 - self.cfg.GROUND_OFFSET_Z
        print(f"[CADMap] Ground offset applied. Floor=Z=0, block tops at Z={block_top:.3f}m")

        # Step 3: Build full cloud
        full = o3d.geometry.PointCloud()
        full.points = o3d.utility.Vector3dVector(ros.astype(np.float32))

        # Step 4: Downsample
        full = full.voxel_down_sample(voxel_size=self.cfg.VOXEL_SIZE)

        # Step 5: Crop to forest volume
        self.forest_map = self._crop(full)
        print(f"[CADMap] Forest crop: {len(self.forest_map.points)} points")

        # Step 6: KDTree
        self.map_kdtree = o3d.geometry.KDTreeFlann(self.forest_map)

        self._loaded = True
        print("[CADMap] Ready.\n")
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def _crop(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        c = self.cfg
        mask = (
            (pts[:,0] >= c.X_MIN) & (pts[:,0] <= c.X_MAX) &
            (pts[:,1] >= c.Y_MIN) & (pts[:,1] <= c.Y_MAX) &
            (pts[:,2] >= c.Z_MIN) & (pts[:,2] <= c.Z_MAX)
        )
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[mask])
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING STEPS
# ─────────────────────────────────────────────────────────────────────────────

def ros2_to_open3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    pts = [
        [p[0], p[1], p[2]]
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
    ]
    pcd = o3d.geometry.PointCloud()
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float32))
    return pcd


def open3d_to_ros2(pcd, frame_id, stamp) -> PointCloud2:
    pts = np.asarray(pcd.points, dtype=np.float32)
    h = Header()
    h.stamp = stamp
    h.frame_id = frame_id
    return pc2.create_cloud_xyz32(h, pts.tolist())


def passthrough_filter(pcd, cfg):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    m = (
        (pts[:,0]>=cfg.X_MIN)&(pts[:,0]<=cfg.X_MAX)&
        (pts[:,1]>=cfg.Y_MIN)&(pts[:,1]<=cfg.Y_MAX)&
        (pts[:,2]>=cfg.Z_MIN)&(pts[:,2]<=cfg.Z_MAX)
    )
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[m])
    return out


def register_to_map(live_pcd, cad_map, cfg, initial_guess=None):
    """ICP align live scan to CAD map. Returns (aligned_pcd, transform)."""
    if initial_guess is None:
        initial_guess = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        source=live_pcd,
        target=cad_map.forest_map,
        max_correspondence_distance=cfg.ICP_MAX_CORRESPONDENCE,
        init=initial_guess,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg.ICP_MAX_ITERATIONS,
            relative_fitness=cfg.ICP_RELATIVE_FITNESS,
            relative_rmse=cfg.ICP_RELATIVE_RMSE
        )
    )
    return live_pcd.transform(result.transformation), result.transformation


def subtract_background(live_pcd, cad_map, cfg):
    """Remove all points that match the static CAD map. Keeps only new objects."""
    pts = np.asarray(live_pcd.points)
    if pts.shape[0] == 0 or cad_map.map_kdtree is None:
        return live_pcd
    keep = [
        pt for pt in pts
        if cad_map.map_kdtree.search_radius_vector_3d(pt, cfg.SUBTRACTION_RADIUS)[0] == 0
    ]
    out = o3d.geometry.PointCloud()
    if keep:
        out.points = o3d.utility.Vector3dVector(np.array(keep, dtype=np.float32))
    return out


def remove_ground(pcd, cfg):
    """Remove residual floor points. Returns (above_ground, ground)."""
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd, o3d.geometry.PointCloud()
    m = pts[:,2] > cfg.GROUND_Z_THRESHOLD
    above = o3d.geometry.PointCloud()
    above.points = o3d.utility.Vector3dVector(pts[m])
    gnd = o3d.geometry.PointCloud()
    gnd.points = o3d.utility.Vector3dVector(pts[~m])
    return above, gnd


def voxel_downsample(pcd, cfg):
    return pcd.voxel_down_sample(cfg.VOXEL_SIZE) if len(pcd.points) > 0 else pcd


def statistical_outlier_removal(pcd, cfg):
    if len(pcd.points) < cfg.SOR_K_NEIGHBORS:
        return pcd
    clean, _ = pcd.remove_statistical_outlier(cfg.SOR_K_NEIGHBORS, cfg.SOR_STD_RATIO)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(raw_pcd, cad_map, cfg, prior_pose=None, verbose=False):
    def log(stage, pcd):
        if verbose:
            print(f"  [{stage:25s}] {len(pcd.points):>6} pts")

    log("Input", raw_pcd)

    pcd = passthrough_filter(raw_pcd, cfg)
    log("Passthrough", pcd)

    if len(pcd.points) == 0:
        if verbose: print("  [!] Empty after passthrough.")
        return {"preprocessed": o3d.geometry.PointCloud(), "ground": o3d.geometry.PointCloud(),
                "robot_pose": np.eye(4), "raw_count": len(raw_pcd.points), "final_count": 0}

    robot_pose = prior_pose if prior_pose is not None else np.eye(4)
    if cad_map.is_loaded():
        pcd, robot_pose = register_to_map(pcd, cad_map, cfg, robot_pose)
        log("ICP Registration", pcd)
        pcd = subtract_background(pcd, cad_map, cfg)
        log("Background Subtract", pcd)

    pcd = voxel_downsample(pcd, cfg)
    log("Voxel Downsample", pcd)

    pcd, ground = remove_ground(pcd, cfg)
    log("Ground Removal", pcd)

    pcd = statistical_outlier_removal(pcd, cfg)
    log("SOR", pcd)

    return {
        "preprocessed": pcd,
        "ground":       ground,
        "robot_pose":   robot_pose,
        "raw_count":    len(raw_pcd.points),
        "final_count":  len(pcd.points),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class LidarPreprocessNode(Node):

    def __init__(self):
        super().__init__('lidar_preprocess_node')
        self.cfg = PreprocessConfig()
        self.declare_parameter('map_path',     self.cfg.MAP_PATH)
        self.declare_parameter('input_topic',  '/livox/lidar')
        self.declare_parameter('output_topic', '/r2/lidar/preprocessed')
        self.declare_parameter('verbose',       False)

        map_path     = self.get_parameter('map_path').value
        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.verbose = self.get_parameter('verbose').value

        self.cad_map = CADMap(self.cfg)
        if not self.cad_map.load(map_path):
            self.get_logger().warn("CAD map not loaded — no background subtraction.")

        self.prior_pose = np.eye(4)
        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)
        self.get_logger().info(f"Ready | map={map_path} | in={input_topic} | out={output_topic}")

    def callback(self, msg):
        raw = ros2_to_open3d(msg)
        if len(raw.points) == 0:
            return
        result = preprocess_pipeline(raw, self.cad_map, self.cfg,
                                     prior_pose=self.prior_pose, verbose=self.verbose)
        self.prior_pose = result['robot_pose']
        if self.verbose:
            self.get_logger().info(f"{result['raw_count']} -> {result['final_count']} pts")
        self.pub.publish(open3d_to_ros2(result["preprocessed"],
                                        msg.header.frame_id, msg.header.stamp))


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST — uses the real gamefield.pcd
# ─────────────────────────────────────────────────────────────────────────────

def test_with_real_map(map_path="gamefield.pcd"):
    import os
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("ABU Robocon 2026 — Preprocessing with real gamefield.pcd")
    print("=" * 60)

    if not os.path.exists(map_path):
        print(f"[!] Not found: {map_path}  — place gamefield.pcd in the same folder.")
        return None

    cfg = PreprocessConfig()
    cfg.MAP_PATH = map_path

    cad_map = CADMap(cfg)
    if not cad_map.load(map_path):
        return None

    # Inspect forest map bounds
    pts = np.asarray(cad_map.forest_map.points)
    block_top_z = 2.420 - cfg.GROUND_OFFSET_Z   # 0.400 m
    print(f"Forest cloud stats:")
    print(f"  Points : {len(pts)}")
    print(f"  X range: {pts[:,0].min():.3f} to {pts[:,0].max():.3f} m")
    print(f"  Y range: {pts[:,1].min():.3f} to {pts[:,1].max():.3f} m")
    print(f"  Z range: {pts[:,2].min():.3f} to {pts[:,2].max():.3f} m")
    print(f"  Block tops at Z = {block_top_z:.3f} m")

    # Simulate live scan: forest map + sensor noise + 2 KFS objects
    live_pts = pts.copy() + rng.normal(0, 0.006, pts.shape)

    # KFS 1 — somewhere in the middle of the forest
    mid_x = float(pts[:,0].mean())
    mid_y = float(pts[:,1].mean())
    kfs1 = rng.uniform(
        [mid_x - 0.55, mid_y - 0.55, block_top_z + 0.01],
        [mid_x - 0.45, mid_y - 0.45, block_top_z + 0.12],
        (100, 3)
    )
    # KFS 2 — offset from KFS 1
    kfs2 = rng.uniform(
        [mid_x + 0.45, mid_y + 0.45, block_top_z + 0.01],
        [mid_x + 0.55, mid_y + 0.55, block_top_z + 0.12],
        (100, 3)
    )

    all_live = np.vstack([live_pts, kfs1, kfs2])
    live_pcd = o3d.geometry.PointCloud()
    live_pcd.points = o3d.utility.Vector3dVector(all_live.astype(np.float32))
    print(f"\nSimulated live scan: {len(live_pcd.points)} pts  "
          f"(map+noise + {len(kfs1)+len(kfs2)} KFS pts)")

    print("\nPipeline:")
    result = preprocess_pipeline(live_pcd, cad_map, cfg, verbose=True)

    print(f"\n{'─'*60}")
    print(f"  Input  : {result['raw_count']} pts")
    print(f"  Output : {result['final_count']} pts  <- KFS candidates only")
    kfs_true = len(kfs1) + len(kfs2)
    if result['final_count'] > 0:
        print(f"  KFS retention: ~{100*result['final_count']/kfs_true:.0f}%")
    print(f"{'─'*60}")
    print("\n  Next: Euclidean Clustering on 'preprocessed' cloud.\n")

    # Visualize
    cad_vis = cad_map.forest_map
    cad_vis.paint_uniform_color([0.55, 0.55, 0.55])   # gray  = CAD map

    out_vis = result["preprocessed"]
    out_vis.paint_uniform_color([0.1, 0.9, 0.1])      # green = output

    kfs_ref = o3d.geometry.PointCloud()
    kfs_ref.points = o3d.utility.Vector3dVector(np.vstack([kfs1, kfs2]).astype(np.float32))
    kfs_ref.paint_uniform_color([0.9, 0.1, 0.1])      # red   = true KFS

    o3d.visualization.draw_geometries(
        [cad_vis, out_vis, kfs_ref],
        window_name="ABU 2026 | Gray=CAD map  Green=output  Red=true KFS",
        width=1400, height=800
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main_ros2():
    rclpy.init()
    node = LidarPreprocessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    test_with_real_map("gamefield.pcd")
    # main_ros2()