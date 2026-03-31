from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch.logging import get_logger


def generate_launch_description():
    logger = get_logger('euroc_vio')
    logger.info("Starting EuRoC trajectory launch ...")

    workdir_path = get_package_share_path("euroc_vio")
    default_rviz_config_path = workdir_path / "rviz/trajectory.rviz"

    logger.info(f"Default RViz config path: {default_rviz_config_path}")

    bag_replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "/mnt/e/Documents/mav0/bag/V2_01_easy_ros2/V2_01_easy_ros2.db3",
        ],
        output="screen",
    )

    imu_arg_use_filter = DeclareLaunchArgument(
        name="use_filter",
        default_value="false",
        description="Whether to use the RC low-pass filter for IMU data (true/false)",
    )

    imu_arg_estimator = DeclareLaunchArgument(
        name="estimator",
        default_value="mahony",
        description="The estimator to use (rk4, mahony, madgwick)",
    )

    imu_node = Node(
        package="euroc_vio",
        executable="euroc_imu",
        name="trajectory_publisher",
        output="screen",
        arguments=[
            "--use_filter",
            LaunchConfiguration("use_filter"),
            "--estimator",
            LaunchConfiguration("estimator"),
        ],
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
        arguments=["-d", LaunchConfiguration("rvizconfig")],
    )

    return LaunchDescription(
        [
            # Declare launch arguments
            imu_arg_use_filter,
            imu_arg_estimator,
            rviz_arg,
            # 1. 播放 rosbag (非阻塞方式)
            bag_replay,
            # 2. 启动 IMU 预处理节点
            imu_node,
            # 3. 启动 RViz2 并自动加载配置
            rviz_node,
        ]
    )
