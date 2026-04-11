from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch.logging import get_logger


logger = get_logger('euroc_vio')
logger.info("Starting EuRoC trajectory launch ...")

workdir_path = get_package_share_path("euroc_vio")
default_rviz_config_path = workdir_path / "rviz/imu.rviz"

logger.info(f"Default RViz config path: {default_rviz_config_path}")

if not default_rviz_config_path.exists() or not default_rviz_config_path.is_file():
    raise FileNotFoundError(f"RViz config file not found at: {default_rviz_config_path}")


def generate_launch_description():
    bag_replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "/mnt/e/Documents/mav0/bag/V2_01_easy_ros2/V2_01_easy_ros2.db3",
            "--read-ahead-queue-size",
            "5000",
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
        default_value="rk4",
        description="The estimator to use (rk4, mahony, madgwick)",
    )

    imu_arg_init_position_x = DeclareLaunchArgument(
        name="initial_position_x",
        default_value="-0.98248653560578914",
        description="Initial position x (default: 0.0)",
    )

    imu_arg_init_position_y = DeclareLaunchArgument(
        name="initial_position_y",
        default_value="0.46277992113897914",
        description="Initial position y (default: 0.0)",
    )

    imu_arg_init_position_z = DeclareLaunchArgument(
        name="initial_position_z",
        default_value="1.4401002233560267",
        description="Initial position z (default: 0.0)",
    )

    imu_node = Node(
        package="euroc_vio",
        executable="euroc_imu",
        name="trajectory_publisher",
        output="screen",
        parameters=[
            {"use_filter": LaunchConfiguration("use_filter")},
            {"estimator": LaunchConfiguration("estimator")},
            {"initial_position_x": LaunchConfiguration("initial_position_x")},
            {"initial_position_y": LaunchConfiguration("initial_position_y")},
            {"initial_position_z": LaunchConfiguration("initial_position_z")},
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
            imu_arg_init_position_x,
            imu_arg_init_position_y,
            imu_arg_init_position_z,
            rviz_arg,
            # 1. 播放 rosbag (非阻塞方式)
            bag_replay,
            # 2. 启动 IMU 预处理节点
            imu_node,
            # 3. 启动 RViz2 并自动加载配置
            rviz_node,
        ]
    )
