from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    sim_node = Node(
        package="euroc_vio",
        executable="VisualSim",
        name="visual_slam_path_publisher",
        output="screen",
    )

    # est_node = Node(
    #     package="euroc_vio",
    #     executable="EstimationLoader",
    #     name="visual_slam_path_publisher",
    #     output="screen",
    # )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
    )

    return LaunchDescription(
        [
            # 启动 IMU 仿真节点
            sim_node,
            # est_node,
            # 启动 RViz2 并自动加载配置
            rviz_node,
        ]
    )
