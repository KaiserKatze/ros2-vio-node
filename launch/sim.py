from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting EuRoC trajectory launch ...")

    workdir_path = get_package_share_path("euroc_vio")
    default_rviz_config_path = workdir_path / "rviz/sim.rviz"

    # sim_node = Node(
    #     package="euroc_vio",
    #     executable="imu_sim",
    #     name="simulator_publisher",
    #     output="screen",
    # )

    sim_node = Node(
        package="euroc_vio",
        executable="VisualSim",
        name="visual_slam_path_publisher",
        output="screen",
    )

    est_node = Node(
        package="euroc_vio",
        executable="EstimationLoader",
        name="visual_slam_path_publisher",
        output="screen",
    )

    rviz_arg = DeclareLaunchArgument(
        name="rvizconfig",
        default_value=str(default_rviz_config_path),
        description="Absolute path to rviz config file",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        # arguments=["-d", LaunchConfiguration("rvizconfig")],
    )

    return LaunchDescription(
        [
            rviz_arg,
            # 启动 IMU 仿真节点
            sim_node,
            est_node,
            # 启动 RViz2 并自动加载配置
            rviz_node,
        ]
    )
