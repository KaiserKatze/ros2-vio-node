from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.logging import get_logger

import os
import pathlib


def setup_launch_entities(context, *args, **kwargs):
    logger = get_logger("euroc_vio")

    # 1. 获取参数
    target_str = LaunchConfiguration("target").perform(context)
    estimator_val = LaunchConfiguration("estimator").perform(context)

    home_path = pathlib.PosixPath(os.path.expanduser("~"))
    # 根据上一个 package 的录制路径定位 Bag 文件夹
    bag_path = home_path / "EuRoC_MAV_Datasets_As_ROS2_Bags" / target_str

    if not bag_path.exists():
        raise FileNotFoundError(f"未找到对应的 ROS2 Bag 路径: {bag_path}")

    # 获取 RViz 配置
    pkg_path = get_package_share_path("euroc_vio")
    rviz_config_path = pkg_path / "rviz" / "imu.rviz"

    # 2. 定义播放进程
    bag_play = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            str(bag_path.resolve()),
            "--read-ahead-queue-size",
            "5000",
        ],
        output="screen",
    )

    # 3. 定义 IMU 节点
    imu_node = Node(
        package="euroc_vio",
        executable="euroc_imu",
        name="trajectory_publisher",
        output="screen",
        parameters=[{"estimator": estimator_val}],
    )

    # 4. 定义 RViz 节点
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_config_path)],
    )

    return [
        imu_node,
        rviz_node,
        bag_play,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="target", default_value="V2_01_easy", description="数据集序列名称"
            ),
            DeclareLaunchArgument(
                name="estimator",
                default_value="rk4",
                description="估计算法 (rk4, mahony, madgwick)",
            ),
            OpaqueFunction(function=setup_launch_entities),
        ]
    )
