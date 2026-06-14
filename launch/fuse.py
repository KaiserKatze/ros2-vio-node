from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger
from launch.actions import Shutdown

import pathlib
import os

debug = False


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting trajectory analysis ...")

    path_home = pathlib.PosixPath(os.path.expanduser("~"))
    path_workdir = path_home / "vio_ws"

    mav0_path = path_workdir / "mav0"
    # mav0_path = pathlib.PosixPath("/mnt", "e", "Documents", "mav0")

    cam0_path = mav0_path / "cam1"
    imu0_path = mav0_path / "imu0"
    truth_path = mav0_path / "state_groundtruth_estimate0"
    path_estimation_csv = str(mav0_path / "estimated_motion.csv")
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
        # "RK4Estimator",
        # "Preintegrator",
        "FuseEstimator",
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
        "confidence_angular_displacement": 1e-4,
        # 单目视觉估计平移方向置信度
        "confidence_normalized_translation": 1e+6,
        # 轨迹估计类的输出目录
        "output_dir": output_dir,
        # 启用的轨迹估计类列表
        "estimators": active_estimators,
    }

    post_nodes = []

    # 定义各个估计输出对应的 ROS Topic 话题映射
    topic_mappings = {
        "FastEstimator": "/fast_est",
        "EulerEstimator": "/midpoint_est",
        "RK4Estimator": "/rk4_est",
        "Preintegrator": "/preintegrate_est",
        "FuseEstimator": "/fuse_est",
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

    # 核心节点：TrajectoryFactory，通过 on_exit 实现顺序执行
    factory_node = Node(
        package="euroc_vio",
        executable="VisualInertial",
        name="TrajectoryFactory",
        output="screen",
        parameters=[factory_params],
        prefix=prefix,
        on_exit=post_nodes,  # 等待 factory_node 结束后再启动后续节点
    )

    return LaunchDescription([factory_node])
