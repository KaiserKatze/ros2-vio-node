from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger
from launch.actions import Shutdown
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_prefix

import hashlib
import json
import pathlib
import os
import tempfile

debug = False


def file_fingerprint(path):
    """Always hash the content, even when LastModified and size are unchanged."""
    path = pathlib.Path(path)
    before = path.stat()
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_mtime_ns, before.st_size, before.st_ctime_ns) != (
        after.st_mtime_ns, after.st_size, after.st_ctime_ns
    ):
        raise RuntimeError(f"File changed while checking cache: {path}")
    return {"mtime_ns": after.st_mtime_ns, "size": after.st_size,
            "md5": digest.hexdigest()}


def file_snapshot(paths):
    return {str(path): file_fingerprint(path)
            for path in sorted({pathlib.Path(p).absolute() for p in paths})}


def find_source_root(workdir):
    """Support source/symlink launches and normal colcon installs."""
    override = os.environ.get("EUROC_VIO_SOURCE_DIR")
    if override:
        root = pathlib.Path(override).expanduser().resolve()
        if not (root / "src/VisualSim/VisualInertial.cpp").is_file():
            raise ValueError(f"Invalid EUROC_VIO_SOURCE_DIR: {root}")
        return root

    candidates = [pathlib.Path(__file__).resolve().parent.parent]
    cmake_cache = workdir / "build/euroc_vio/CMakeCache.txt"
    if cmake_cache.is_file():
        for line in cmake_cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                candidates.append(pathlib.Path(line.split("=", 1)[1]))
    for root in candidates:
        if (root / "src/VisualSim/VisualInertial.cpp").is_file():
            return root.resolve()
    matches = sorted({path.parent.parent.parent.resolve()
                      for path in (workdir / "src").rglob("VisualInertial.cpp")
                      if path.parent.name == "VisualSim"
                      and path.parent.parent.name == "src"})
    if len(matches) > 1:
        raise ValueError("Multiple source trees found; set EUROC_VIO_SOURCE_DIR")
    return matches[0] if matches else None


def estimation_dependencies(params, prefix, source_root):
    paths = [pathlib.Path(__file__).absolute(),
             prefix / "lib/euroc_vio/VisualInertial"]
    paths.extend(pathlib.Path(value) for key, value in params.items()
                 if key.startswith("path_"))
    header_roots = [prefix / "include/euroc_vio"]
    if source_root is not None:
        paths.extend(source_root / name for name in (
            "src/VisualSim/VisualInertial.cpp", "launch/mono.py",
            "CMakeLists.txt", "package.xml"))
        header_roots.extend([source_root / "include", source_root / "src/VisualSim"])
    for root in header_roots:
        paths.extend(path for path in root.rglob("*")
                     if path.is_file() and path.suffix in {".h", ".hpp", ".hxx", ".inl"})
    return paths


class TrajectoryCache:
    """A manifest is valid only after a successful, complete estimation run."""

    def __init__(self, path, params, dependencies, outputs):
        self.path = pathlib.Path(path)
        self.params = params
        self.dependencies = dependencies
        self.outputs = outputs
        self.inputs = self.input_snapshot()
        self.previous_outputs = None

    def input_snapshot(self):
        return {"parameters": self.params, "files": file_snapshot(self.dependencies())}

    def output_snapshot(self):
        for path in self.outputs:
            # A header-only trajectory is not a usable result.
            with path.open(encoding="utf-8") as stream:
                if not any(line.strip() and not line.startswith("#") for line in stream):
                    raise ValueError(f"Empty trajectory: {path}")
        return file_snapshot(self.outputs)

    def is_current(self):
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            return (isinstance(record, dict) and record.get("version") == 1
                    and record.get("inputs") == self.inputs
                    and record.get("outputs") == self.output_snapshot())
        except (OSError, ValueError):
            return False

    def begin(self):
        # Invalidate before running: a failure must not leave a valid manifest.
        self.path.unlink(missing_ok=True)
        self.previous_outputs = {
            str(path.absolute()): file_fingerprint(path) if path.is_file() else None
            for path in self.outputs
        }

    def finish(self, save=True):
        if self.input_snapshot() != self.inputs:
            raise RuntimeError("Estimation dependencies changed during the run")
        outputs = self.output_snapshot()
        if self.previous_outputs is None or any(
            fingerprint == self.previous_outputs.get(path)
            for path, fingerprint in outputs.items()
        ):
            raise RuntimeError("TrajectoryFactory did not refresh every trajectory")
        if not save:
            return
        record = {"version": 1, "inputs": self.inputs, "outputs": outputs}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name, suffix=".tmp", delete=False
            ) as stream:
                temporary = pathlib.Path(stream.name)
                json.dump(record, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting trajectory analysis ...")

    path_home = pathlib.Path.home()
    path_workdir = path_home / "vio_ws"
    # path_workdir = pathlib.PosixPath("/mnt", "e", "Documents")

    mav0_path = path_workdir / "mav0"
    cam0_path = mav0_path / "cam1"
    imu0_path = mav0_path / "imu0"
    truth_path = mav0_path / "state_groundtruth_estimate0"
    path_estimation_csv = str(mav0_path / "estimated_motion_sim1.csv")
    path_cam0_yaml = str(cam0_path / "sensor.yaml")
    path_imu_csv = str(imu0_path / "data.csv")
    path_imu_yaml = str(imu0_path / "sensor.yaml")
    path_truth_csv = str(truth_path / "data.csv")
    path_truth_yaml = str(truth_path / "sensor.yaml")

    # 结果 CSV 文件的输出目录
    output_dir = str(path_workdir)

    # 声明希望在 TrajectoryFactory 中执行的评估器列表
    active_estimators = [
        # "FastEstimator",
        # "EulerEstimator",
        "RK4Estimator",
        # "Preintegrator",
        # "FuseEstimator",
    ]

    # 使用 GDB 查错
    prefix = ["xterm -fa 'Monospace' -fs 16 -e gdb -ex run --args"] if debug else []

    factory_params = {
        # 是否使用真实姿态进行初始化
        "use_true_init_pose": True,
        # 利用本方法 (单目视觉) 估计得到的角位移向量和单位化平移向量的数据文件
        "path_estimation_csv": path_estimation_csv,
        # 相机传感器参数
        "path_cam0_yaml": path_cam0_yaml,
        # IMU 数据文件
        "path_imu_csv": path_imu_csv,
        # IMU 传感器参数
        "path_imu_yaml": path_imu_yaml,
        # 真实数据文件
        "path_truth_csv": path_truth_csv,
        # 真实数据变换矩阵
        "path_truth_yaml": path_truth_yaml,
        # 单目视觉估计角位移置信度
        "confidence_angular_displacement": 16.54940287490809,
        # 单目视觉估计平移方向置信度
        "confidence_normalized_translation": 1e6,
        # 轨迹估计类的输出目录
        "output_dir": output_dir,
        # 启用的轨迹估计类列表
        "estimators": active_estimators,
        # 左目相机的投影矩阵
        "proj_left": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        # 右目相机的投影矩阵
        "proj_right": [1.0, 0.0, 0.0, -0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }

    # 核心节点：TrajectoryFactory，通过 on_exit 实现顺序执行
    factory_node = Node(
        package="euroc_vio",
        executable="VisualInertial",
        name="TrajectoryFactory",
        output="screen",
        parameters=[factory_params],
        prefix=prefix,
    )

    post_nodes = []

    # 定义各个估计输出对应的 ROS Topic 话题映射
    topic_mappings = {
        "FastEstimator": "/traj/fast_est",
        "EulerEstimator": "/traj/midpoint_est",
        "RK4Estimator": "/traj/rk4_est",
        "Preintegrator": "/traj/preintegrate_est",
        "FuseEstimator": "/traj/fuse_est",
    }

    # 动态启动各估计轨迹的数据加载与发布器
    for est_name in active_estimators:
        csv_filepath = os.path.join(output_dir, f"{est_name}.csv")
        post_nodes.append(
            Node(
                package="euroc_vio",
                executable="SimpleDataLoader",
                name=f"loader_{est_name}",
                output="screen",
                parameters=[
                    {
                        "csv_file": csv_filepath,
                        "topic_name": topic_mappings[est_name],
                        "skip_header": True,
                        "delim": ",",
                    }
                ],
            )
        )

    # 启动真值的数据加载与发布器
    post_nodes.append(
        Node(
            package="euroc_vio",
            executable="SimpleDataLoader",
            name="loader_ground_truth",
            output="screen",
            parameters=[
                {
                    "csv_file": path_truth_csv,
                    "topic_name": "/ground_truth",
                    "skip_header": True,
                    "delim": ",",
                }
            ],
        )
    )

    if not debug:
        post_nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                on_exit=Shutdown(),
            )
        )

    prefix_path = pathlib.Path(get_package_prefix("euroc_vio"))
    source_root = find_source_root(path_workdir)
    if source_root is None:
        logger.warning("Source tree not found; checking installed files only. "
                       "Set EUROC_VIO_SOURCE_DIR to also track C++ source changes.")
    cache = TrajectoryCache(
        path_workdir / ".mono_trajectory_cache.json",
        factory_params,
        lambda: estimation_dependencies(factory_params, prefix_path, source_root),
        [pathlib.Path(output_dir) / f"{name}.csv" for name in active_estimators],
    )
    if not debug and cache.is_current():
        logger.info("LastModified, MD5 and parameters unchanged; reusing trajectories.")
        return LaunchDescription(post_nodes)

    logger.info("Trajectory cache missing/changed, or debug enabled; running estimation.")
    cache.begin()

    def on_factory_exit(event, context):
        if event.returncode != 0:
            logger.error(f"TrajectoryFactory failed (exit {event.returncode}); cache not saved.")
            return [Shutdown(reason="Trajectory estimation failed")]
        try:
            # A debugger wrapper can hide the estimator's exit status.
            cache.finish(save=not debug)
        except (OSError, ValueError, RuntimeError) as error:
            logger.error(f"Trajectory cache validation failed: {error}")
            return [Shutdown(reason="Trajectory output validation failed")]
        logger.info("TrajectoryFactory finished. Starting subsequent nodes...")
        return post_nodes

    # 先注册退出处理器，避免很快结束的估计进程错过回调。
    pose_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=factory_node,
            on_exit=on_factory_exit,
        )
    )

    return LaunchDescription([pose_handler, factory_node])
