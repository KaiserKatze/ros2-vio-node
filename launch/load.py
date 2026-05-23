from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription

from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    loader_node = Node(
        package="euroc_vio",
        executable="SimpleDataLoader",
        name="SimpleDataLoader",
        output="screen",
        parameters=[
            {
                "csv_file": "/home/kk/vio_ws/CameraTrajectory.txt",
                "topic_name": "/path_traj",
                "skip_header": True,
                "delim": " ",
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
            loader_node,
        ]
    )
