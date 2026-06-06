from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.logging import get_logger

import pathlib
import os


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    home_path = pathlib.PosixPath(os.path.expanduser("~"))

    # mav0_path = home_path / "vio_ws" / "mav0"
    mav0_path = pathlib.PosixPath("/mnt/e/Documents/mav0")

    cam0_path = mav0_path / "cam0"
    imu0_path = mav0_path / "imu0"
    truth_path = mav0_path / "state_groundtruth_estimate0"

    # 输出: 轨迹话题 /path_fast_est
    # 输出: 轨迹话题 /pose_fast_est
    # 输出: 轨迹话题 /path_fuse_est
    # 输出: 轨迹话题 /pose_fuse_est
    fuse_node = Node(
        package="euroc_vio",
        executable="VisualInertial",
        name="VisualInertial",
        output="screen",
        parameters=[
            {
                "use_evo_sim3": False,
                "use_true_translation_in_fast": True,
                "use_true_init_pose": True,
                "path_estimation_csv": str(
                    home_path / "vio_ws" / "estimated_motion.csv"
                ),
                "path_imu_csv": str(imu0_path / "data.csv"),
                "path_imu_yaml": str(imu0_path / "sensor.yaml"),
                "path_truth_csv": str(truth_path / "data.csv"),
                "path_cam0_yaml": str(cam0_path / "sensor.yaml"),
                "path_truth_yaml": str(truth_path / "sensor.yaml"),
            }
        ],
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
