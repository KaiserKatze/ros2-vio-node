from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    # 输出: 轨迹话题 /path_fast_est
    # 输出: 轨迹话题 /pose_fast_est
    # 输出: 轨迹话题 /path_fuse_est
    # 输出: 轨迹话题 /pose_fuse_est
    fuse_node = Node(
        package="euroc_vio",
        executable="VisualInertial",
        name="VisualInertial",
        output="screen",
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
