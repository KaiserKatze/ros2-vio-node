from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.logging import get_logger

import pathlib
import os

debug = False

def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    home_path = pathlib.PosixPath(os.path.expanduser("~"))

    mav0_path = home_path / "vio_ws" / "mav0"
    # mav0_path = pathlib.PosixPath("/mnt", "e", "Documents", "mav0")

    cam0_path = mav0_path / "cam1"
    imu0_path = mav0_path / "imu0"
    truth_path = mav0_path / "state_groundtruth_estimate0"
    path_estimation_csv = str(home_path / "vio_ws" / "estimated_motion.csv")
    path_cam0_yaml = str(cam0_path / "sensor.yaml")
    path_imu_csv = str(imu0_path / "data.csv")
    path_imu_yaml = str(imu0_path / "sensor.yaml")
    path_truth_csv = str(truth_path / "data.csv")
    path_truth_yaml = str(truth_path / "sensor.yaml")
    # 使用 GDB 查错
    prefix = ["xterm -e gdb -ex run --args"] if debug else []

    params = {
        # 是否使用 Python 工具 evo 实施 Sim(3) 变换
        "use_evo_sim3": False,
        # 是否在本方法中使用真实平移向量
        "use_true_translation_in_fast": False,
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
    }

    fuse_node = Node(
        package="euroc_vio",
        executable="VisualInertial",
        name="VisualInertial",
        output="screen",
        parameters=[params],
        prefix=prefix,  # 关键配置
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
    )

    return LaunchDescription(
        [
            rviz_node,
            fuse_node,
        ]
    )
