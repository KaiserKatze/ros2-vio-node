from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription

from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    # 输入: 角速度向量和单位化平移向量
    # 输出: 轨迹话题 /path_fast_est
    # 输出: 轨迹话题 /pose_fast_est
    est_node = Node(
        package="euroc_vio",
        executable="EstimationLoader",
        name="EstimationLoader",
        output="screen",
    )

    # 输入: IMU 话题 /imu0
    # 输入: 轨迹话题 /path_fast_est
    # 输出: 轨迹话题 /path_fuse_est
    # 输出: 轨迹话题 /pose_fuse_est
    fuse_node = Node(
        package="euroc_vio",
        executable="FuseKalman",
        name="FuseKalman",
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
            est_node,
            fuse_node,
        ]
    )
