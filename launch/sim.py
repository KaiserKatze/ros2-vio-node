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

    # sim_arg_init_ax = DeclareLaunchArgument(
    #     name="init_ax",
    #     default_value="1.0",
    #     description="Accelerometer reading",
    # )
    # sim_arg_init_ay = DeclareLaunchArgument(
    #     name="init_ay",
    #     default_value="0.0",
    #     description="Accelerometer reading",
    # )
    # sim_arg_init_az = DeclareLaunchArgument(
    #     name="init_az",
    #     default_value="0.0",
    #     description="Accelerometer reading",
    # )
    # sim_arg_init_gx = DeclareLaunchArgument(
    #     name="init_gx",
    #     default_value="0.0",
    #     description="Gyroscope reading",
    # )
    # sim_arg_init_gy = DeclareLaunchArgument(
    #     name="init_gy",
    #     default_value="0.0",
    #     description="Gyroscope reading",
    # )
    # sim_arg_init_gz = DeclareLaunchArgument(
    #     name="init_gz",
    #     default_value="3.1415926535898",
    #     description="Gyroscope reading",
    # )

    sim_node = Node(
        package="euroc_vio",
        executable="imu_sim",
        name="simulator_publisher",
        output="screen",
        # parameters=[
        #     {"init_ax": LaunchConfiguration("init_ax")},
        #     {"init_ay": LaunchConfiguration("init_ay")},
        #     {"init_az": LaunchConfiguration("init_az")},
        #     {"init_gx": LaunchConfiguration("init_gx")},
        #     {"init_gy": LaunchConfiguration("init_gy")},
        #     {"init_gz": LaunchConfiguration("init_gz")},
        # ],
    )

    # rviz_arg = DeclareLaunchArgument(
    #     name="rvizconfig",
    #     default_value=str(default_rviz_config_path),
    #     description="Absolute path to rviz config file",
    # )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        # arguments=["-d", LaunchConfiguration("rvizconfig")],
    )

    return LaunchDescription(
        [
            # sim_arg_init_ax,
            # sim_arg_init_ay,
            # sim_arg_init_az,
            # sim_arg_init_gx,
            # sim_arg_init_gy,
            # sim_arg_init_gz,
            # rviz_arg,
            # 启动 IMU 仿真节点
            sim_node,
            # 启动 RViz2 并自动加载配置
            rviz_node,
        ]
    )
