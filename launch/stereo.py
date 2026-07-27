from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger
from launch.actions import ExecuteProcess
from launch.actions import Shutdown
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

import pathlib
import os

debug = False
use_evo = True
pypi_mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"


def generate_launch_description():
    logger = get_logger("euroc_vio")
    logger.info("Starting trajectory analysis ...")

    path_home = pathlib.PosixPath(os.path.expanduser("~"))
    path_workdir = path_home / "vio_ws"

    # mav0_path = pathlib.PosixPath("/mnt", "e", "Documents", "mav0")
    mav0_path = path_home / "EuRoC_MAV_Datasets" / "V2_01_easy" / "mav0"
    truth_path = mav0_path / "state_groundtruth_estimate0"
    path_truth_csv = truth_path / "data.csv"
    # path_stereo_raw = path_workdir / "estimated_trajectory.csv"
    path_stereo_raw = path_workdir / "trajectory_tum.txt"

    # 参考 EvoSim3.hpp, 借助 python 模块 evo 完成轨迹变换与误差评估:
    #   1. 真值轨迹 (EuRoC CSV) 转换为 TUM 格式, 作为参考轨迹;
    #   2. evo_traj --align 对估计轨迹执行 SE(3) Umeyama 对齐
    #      (不加 --correct_scale, 即 SE(3) 变换而非 EvoSim3 的 SIM(3));
    #   3. evo_ape --align 在 SE(3) 对齐的基础上计算绝对轨迹误差 (APE).
    # evo 的 --save_as_tum 会在当前工作目录按输入文件名生成 <文件名主干>.tum
    path_truth_tum = path_workdir / (path_truth_csv.stem + ".tum")
    path_aligned_tum = path_workdir / (path_stereo_raw.stem + ".tum")
    path_ape_results = path_workdir / "evo_ape_se3.zip"

    # 查找带有 evo 的 python 虚拟环境 (EvoSim3.hpp 约定其位于工作目录下)
    path_venv = path_workdir / ".venv"
    if use_evo and not path_venv.exists():
        # 在目录 path_workdir 中执行 `python -m venv .venv`
        # 接着执行 evo 安装流程 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple evo --upgrade --no-binary evo`
        logger.warning(
            f"未找到含 evo 的虚拟环境 ({path_venv!r}), "
            "正在创建虚拟环境和安装工具"
        )

        import subprocess
        try:
            python_exec = "$(which python || which python3)"
            subprocess.run(
                [python_exec, "-m", "venv", path_venv.name],
                cwd=str(path_workdir),
                check=True,
            )
            # 安装 evo
            subprocess.run(
                [
                    python_exec,
                    "-m",
                    "pip",
                    "install",
                    "-i",
                    pypi_mirror,
                    "--upgrade",
                    "evo",
                ],
                cwd=str(path_workdir),
                check=True,
            )
            logger.info("Virtual environment and evo installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup evo virtual environment: {e}")
            # 如果失败，设置 use_evo = False
            use_evo = False
            logger.warning("Skipping evo steps due to setup failure.")

    # 有 evo 时展示 SE(3) 对齐后的估计轨迹, 否则退回原始估计轨迹
    path_stereo_csv = str(path_aligned_tum if use_evo else path_stereo_raw)
    path_truth_csv = str(path_truth_csv)

    # 使用 GDB 查错
    prefix = ["xterm -fa 'Monospace' -fs 16 -e gdb -ex run --args"] if debug else []

    nodes = []

    # 纯视觉双目里程计 (展示由 VisualSlam.cpp 生产的轨迹数据)
    nodes.append(
        Node(
            package="euroc_vio",
            executable="SimpleDataLoader",
            name="loader_StereoEstimator",
            output="screen",
            parameters=[
                {
                    "csv_file": path_stereo_csv,
                    "topic_name": "/traj/stereo_est",
                    "skip_header": path_stereo_csv.endswith(".csv"),
                    "delim": "," if path_stereo_csv.endswith(".csv") else " ",
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

    if not use_evo:
        return LaunchDescription(nodes)

    # 参考 EvoSim3.hpp::TransformSim3 的调用方式 (source .venv && yes y | evo_*),
    # 区别在于此处只用 --align (SE(3)), 不用 --align --correct_scale (SIM(3))
    evo_command = " && ".join(
        [
            f'source "{path_venv}/bin/activate"',
            # 1. 真值轨迹: EuRoC CSV -> TUM (生成 data.tum)
            f'yes y | evo_traj euroc "{path_truth_csv}" --save_as_tum',
            # 2. 对估计轨迹执行 SE(3) 变换, 对齐到真值参考系
            #    (生成 trajectory_tum.tum, 供 RViz 与真值轨迹同框对比)
            f'yes y | evo_traj tum "{path_stereo_raw}"'
            f' --ref="{path_truth_tum}" --align --save_as_tum',
            # 3. 基于 SE(3) 对齐计算轨迹估计误差 APE
            #    (rmse/mean/median/std/min/max 打印到屏幕, 结果存档为 zip)
            f'yes y | evo_ape tum "{path_truth_tum}" "{path_stereo_raw}"'
            f' --align --save_results "{path_ape_results}"',
        ]
    )

    evo_se3 = ExecuteProcess(
        cmd=["bash", "-c", evo_command],
        name="evo_se3_ape",
        cwd=str(path_workdir),
        output="screen",
    )

    return LaunchDescription(
        [
            evo_se3,
            # evo 完成 SE(3) 变换与误差计算后, 再启动轨迹加载器与 RViz
            RegisterEventHandler(
                OnProcessExit(target_action=evo_se3, on_exit=nodes)
            ),
        ]
    )
