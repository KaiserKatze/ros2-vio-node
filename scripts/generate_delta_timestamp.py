#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import numpy as np
import pandas as pd


def generate_delta_timestamp_csv(path_imu_csv, path_cam_csv, path_output_csv):
    """读取 IMU 和相机的 CSV 文件中的时间戳，以 CAM 时间戳为基准，

    匹配最近的 IMU 时间戳，计算二者差值并导出为新的 CSV 文件。
    """
    print("=== 开始加载时间戳数据 ===")
    print(f"IMU 路径: {path_imu_csv}")
    print(f"CAM 路径: {path_cam_csv}\n")

    # 1. 检查输入文件是否存在
    if not path_imu_csv.exists():
        print(f"错误: 未找到 IMU 文件 {path_imu_csv}")
        return
    if not path_cam_csv.exists():
        print(f"错误: 未找到 CAM 文件 {path_cam_csv}")
        return

    # 2. 高效读取时间戳 (仅读取第一列)
    imu_df = pd.read_csv(path_imu_csv, usecols=[0], header=None, names=["timestamp"])
    cam_df = pd.read_csv(path_cam_csv, usecols=[0], header=None, names=["timestamp"])

    # 自动过滤可能存在的表头 (将非数字剔除)
    imu_timestamps = (
        pd.to_numeric(imu_df["timestamp"], errors="coerce").dropna().to_numpy()
    )
    cam_timestamps = (
        pd.to_numeric(cam_df["timestamp"], errors="coerce").dropna().to_numpy()
    )

    if len(imu_timestamps) == 0 or len(cam_timestamps) == 0:
        print("错误: 无法从 CSV 文件中解析出有效的时间戳。")
        return

    # 3. 自动单位检测与对齐 (防止仿真数据集与真机数据集单位不一致)
    # 若最大时间戳大于 1e12 则判定为纳秒 (ns)，否则判定为秒 (s)
    if np.max(imu_timestamps) > 1e12:
        scale = 1e9
        unit_str = "纳秒 (ns) -> 已自动转换为秒 (s)"
    else:
        scale = 1.0
        unit_str = "秒 (s)"

    t_imu = imu_timestamps / scale
    t_cam = cam_timestamps / scale

    print(f"数据集时间戳单位检测: {unit_str}")
    print(f"成功加载 IMU 时间戳数量: {len(t_imu)}")
    print(f"成功加载 CAM 时间戳数量: {len(t_cam)}\n")

    print("=== 开始进行时间戳最近邻匹配 ===")
    # 使用 np.searchsorted 快速实现大规模二分查找最近邻
    indices = np.searchsorted(t_imu, t_cam)
    indices = np.clip(indices, 1, len(t_imu) - 1)

    # 比较求得真正最近的 IMU 索引 (左侧邻居或右侧邻居)
    left_diff = t_cam - t_imu[indices - 1]
    right_diff = t_imu[indices] - t_cam
    choose_left = left_diff < right_diff

    closest_imu_times = np.where(choose_left, t_imu[indices - 1], t_imu[indices])

    # 计算每一个相机帧与最近 IMU 帧的真实时间偏差 (CAM_TS - IMU_TS)
    delta_ts = t_cam - closest_imu_times

    # 4. 构建输出 DataFrame 并导出 CSV
    df_output = pd.DataFrame(
        {"CAM_TS": t_cam, "IMU_TS": closest_imu_times, "DELTA_TS": delta_ts}
    )

    # 导出为 CSV，保留小数点后 6 位（微秒级精度）
    df_output.to_csv(path_output_csv, index=False, float_format="%.6f")

    print(f"🎉 成功生成对齐时间戳文件！")
    print(f"输出路径: {path_output_csv.resolve()}")
    print(f"包含的数据列: CAM_TS, IMU_TS, DELTA_TS (单位均为秒)\n")


def run():
    path_home = pathlib.Path(os.path.expanduser("~"))
    path_workdir = path_home / "vio_ws"
    path_workdir = pathlib.PosixPath("/mnt", "e", "Documents")
    mav0_path = path_workdir / "mav0"
    path_imu_csv = mav0_path / "imu0" / "data.csv"
    path_cam_csv = mav0_path / "cam0" / "data.csv"
    # 输出文件命名为 delta_timestamp.csv，默认保存在当前终端执行脚本的目录下
    path_output_csv = pathlib.Path("delta_timestamp.csv")
    generate_delta_timestamp_csv(path_imu_csv, path_cam_csv, path_output_csv)


if __name__ == "__main__":
    run()
