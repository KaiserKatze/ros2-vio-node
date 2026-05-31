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
                "use_true_init_pose": True,
                "path_estimation_csv": str(
                    home_path / "vio_ws" / "estimated_motion.csv"
                ),
                # "path_imu_csv": "/mnt/e/Documents/mav0/imu0/data.csv",
                "path_imu_csv": str(
                    home_path / "vio_ws" / "mav0" / "imu0" / "data.csv"
                ),
                # "path_truth_csv": "/mnt/e/Documents/mav0/state_groundtruth_estimate0/data.csv",
                "path_truth_csv": str(
                    home_path
                    / "vio_ws"
                    / "mav0"
                    / "state_groundtruth_estimate0"
                    / "data.csv"
                ),
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
