from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger
from launch.actions import Shutdown
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

import pathlib
import os

debug = False


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting trajectory analysis ...")

    path_home = pathlib.PosixPath(os.path.expanduser("~"))
    path_workdir = path_home / "vio_ws"
    path_workdir = pathlib.PosixPath("/mnt", "e", "Documents")

    mav0_path = path_workdir / "mav0"
    truth_path = mav0_path / "state_groundtruth_estimate0"
    path_truth_csv = str(truth_path / "data.csv")
    path_stereo_csv = str(path_workdir / "estimated_trajectory.csv")

    # 使用 GDB 查错
    prefix = ["xterm -fa 'Monospace' -fs 16 -e gdb -ex run --args"] if debug else []

    nodes = []

    # 纯视觉双目里程计 (展示由 VisualSlam.cpp 生产的轨迹数据)
    nodes.append(
        Node(
            package="euroc_vio",
            executable="SimpleDataLoader",
            name=f"loader_StereoEstimator",
            output="screen",
            parameters=[
                {
                    "csv_file": path_stereo_csv,
                    "topic_name": "/traj/stereo_est",
                    "skip_header": True,
                    "delim": ",",
                }
            ],
            prefix=prefix,
        )
    )

    # 启动真值的数据加载与发布器
    nodes.append(
        Node(
            package="euroc_vio",
            executable="SimpleDataLoader",
            name="loader_ground_truth",
            output="screen",
            parameters=[
                {
                    "csv_file": path_truth_csv,
                    "topic_name": "/traj/ground_truth",
                    "skip_header": True,
                    "delim": ",",
                }
            ],
        )
    )

    # 用 RViz 将轨迹可视化
    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            on_exit=Shutdown(),
        )
    )

    return LaunchDescription(nodes)
