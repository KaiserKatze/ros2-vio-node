#!/usr/bin/env python3
"""IMU 仿真噪声统计验证脚本。

执行 IMU 仿真程序 (VisualSim)，读取加噪后的 IMU 测量值与 Ground Truth，
计算误差信号 (Measurement - GroundTruth) 的均值、方差、标准差与 Allan 方差，
并与 sensor.yaml 中的连续时间噪声密度换算值对比，使用 matplotlib 绘图。

用法:
    python3 analyze_imu_noise.py [--exe /path/to/VisualSim] [--workdir DIR]
                                 [--skip-run] [--output-dir DIR]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRAVITY_NORM = 9.81  # 与 VisualSim.cpp 中 gravity_world_norm_ 保持一致


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("/home/ros/vio_ws/build/euroc_vio/VisualSim"),
        help="VisualSim 可执行文件路径",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/imu_noise_analysis"),
        help="仿真程序工作目录 (mav0 输出位置)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="跳过仿真执行，直接分析 workdir 下已有的 mav0 数据",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图像输出目录 (默认与 workdir 相同)",
    )
    return parser.parse_args()


def run_simulation(exe: Path, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 运行仿真: {exe} (cwd={workdir})")
    result = subprocess.run(
        [str(exe)],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        sys.exit(f"[ERROR] 仿真程序退出码 {result.returncode}")


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_sensor_yaml(path: Path) -> dict:
    """从 sensor.yaml 提取采样率与连续时间噪声密度。"""
    keys = (
        "rate_hz",
        "gyroscope_noise_density",
        "gyroscope_random_walk",
        "accelerometer_noise_density",
        "accelerometer_random_walk",
    )
    text = path.read_text()
    params = {}
    for key in keys:
        match = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
        if match is None:
            sys.exit(f"[ERROR] sensor.yaml 缺少字段: {key}")
        params[key] = float(match.group(1))
    return params


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) 四元数批量转旋转矩阵，形状 (N, 3, 3)。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def compute_error_signals(imu: np.ndarray, groundtruth: np.ndarray):
    """计算误差信号 e = Measurement - GroundTruth(传感器系) = Bias + WhiteNoise，
    以及扣除 Bias 后的白噪声残差。"""
    n = min(len(imu), len(groundtruth))
    imu, groundtruth = imu[:n], groundtruth[:n]
    if not np.array_equal(imu[:, 0], groundtruth[:, 0]):
        sys.exit("[ERROR] IMU 与 Ground Truth 时间戳不一致")

    gyro_meas, accel_meas = imu[:, 1:4], imu[:, 4:7]
    quat = groundtruth[:, 4:8]
    gyro_bias, accel_bias = groundtruth[:, 11:14], groundtruth[:, 14:17]
    accel_world, gyro_world = groundtruth[:, 17:20], groundtruth[:, 20:23]

    # 由世界系真值重建传感器系理想采样值 (与 VisualSim.cpp 中的转换一致)
    rotation = quaternion_to_rotation_matrix(quat)
    gravity_world = np.array([0.0, 0.0, -GRAVITY_NORM])
    gyro_truth = np.einsum("nji,nj->ni", rotation, gyro_world)
    accel_truth = np.einsum("nji,nj->ni", rotation, accel_world - gravity_world)

    time = imu[:, 0] * 1e-9
    gyro_error = gyro_meas - gyro_truth
    accel_error = accel_meas - accel_truth
    gyro_white = gyro_error - gyro_bias
    accel_white = accel_error - accel_bias
    return time, gyro_error, accel_error, gyro_white, accel_white, gyro_bias, accel_bias


def allan_deviation(data: np.ndarray, rate_hz: float):
    """重叠 Allan 偏差。data 形状 (N, 3)，返回 (taus, adev 形状 (M, 3))。"""
    n = len(data)
    max_cluster = n // 3
    cluster_sizes = np.unique(
        np.logspace(0, np.log10(max_cluster), num=100).astype(int)
    )
    theta = np.cumsum(data, axis=0) / rate_hz  # 积分角/速度
    taus, adevs = [], []
    for m in cluster_sizes:
        # 重叠 Allan 方差: σ²(τ) = <(θ_{k+2m} - 2θ_{k+m} + θ_k)²> / (2τ²)
        diff = theta[2 * m:] - 2 * theta[m:-m] + theta[: -2 * m]
        tau = m / rate_hz
        taus.append(tau)
        adevs.append(np.sqrt(np.mean(diff**2, axis=0) / (2 * tau**2)))
    return np.array(taus), np.array(adevs)


def print_statistics(label: str, white: np.ndarray, sigma_expected: float):
    mean, var, std = white.mean(axis=0), white.var(axis=0), white.std(axis=0)
    print(f"\n=== {label} 白噪声残差 (Measurement - GroundTruth - Bias) ===")
    print(f"  均值:   {mean}")
    print(f"  方差:   {var}")
    print(f"  标准差: {std}")
    print(f"  期望离散标准差 σ_d = σ_c·√f = {sigma_expected:.6e}")
    ratio = std / sigma_expected
    print(f"  实测/期望 比值: {ratio} (应接近 1)")


def plot_bias(time, gyro_bias, accel_bias, output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis_index, axis_name in enumerate("xyz"):
        axes[0].plot(time, gyro_bias[:, axis_index], label=f"b_g {axis_name}")
        axes[1].plot(time, accel_bias[:, axis_index], label=f"b_a {axis_name}")
    axes[0].set_ylabel("Gyro bias [rad/s]")
    axes[1].set_ylabel("Accel bias [m/s²]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.legend()
        ax.grid(True)
    fig.suptitle("Bias 随机游走轨迹")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_white_noise_histogram(gyro_white, accel_white, sigma_gyro, sigma_accel, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, white, sigma, title in (
        (axes[0], gyro_white, sigma_gyro, "Gyro white noise [rad/s]"),
        (axes[1], accel_white, sigma_accel, "Accel white noise [m/s²]"),
    ):
        ax.hist(white.ravel(), bins=100, density=True, alpha=0.6, label="residual")
        grid = np.linspace(-4 * sigma, 4 * sigma, 400)
        gaussian = np.exp(-0.5 * (grid / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(grid, gaussian, "r-", label=f"N(0, σ={sigma:.3e})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    fig.suptitle("白噪声残差直方图 vs 理论高斯分布")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_allan(taus, adev, noise_density, random_walk, title, output: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for axis_index, axis_name in enumerate("xyz"):
        ax.loglog(taus, adev[:, axis_index], label=f"axis {axis_name}")
    # 理论参考线: 白噪声 σ(τ) = N/√τ (斜率 -1/2)；随机游走 σ(τ) = K·√(τ/3) (斜率 +1/2)
    ax.loglog(taus, noise_density / np.sqrt(taus), "k--",
              label=f"white noise N/√τ (N={noise_density:.2e})")
    ax.loglog(taus, random_walk * np.sqrt(taus / 3.0), "k:",
              label=f"random walk K·√(τ/3) (K={random_walk:.2e})")
    ax.set_xlabel("τ [s]")
    ax.set_ylabel("Allan deviation")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_arguments()
    output_dir = args.output_dir or args.workdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_simulation(args.exe, args.workdir)

    mav0 = args.workdir / "mav0"
    imu = load_csv(mav0 / "imu0" / "data.csv")
    groundtruth = load_csv(mav0 / "state_groundtruth_estimate0" / "data.csv")
    params = load_sensor_yaml(mav0 / "imu0" / "sensor.yaml")

    rate_hz = params["rate_hz"]
    sigma_gyro = params["gyroscope_noise_density"] * np.sqrt(rate_hz)
    sigma_accel = params["accelerometer_noise_density"] * np.sqrt(rate_hz)

    (time, gyro_error, accel_error, gyro_white, accel_white,
     gyro_bias, accel_bias) = compute_error_signals(imu, groundtruth)

    print(f"[INFO] 样本数: {len(time)}, 采样率: {rate_hz:.1f} Hz, "
          f"时长: {time[-1] - time[0]:.1f} s")
    print_statistics("Gyroscope", gyro_white, sigma_gyro)
    print_statistics("Accelerometer", accel_white, sigma_accel)

    plot_bias(time, gyro_bias, accel_bias, output_dir / "bias_random_walk.png")
    plot_white_noise_histogram(gyro_white, accel_white, sigma_gyro, sigma_accel,
                               output_dir / "white_noise_histogram.png")

    # Allan 方差基于纯误差信号 (Bias + WhiteNoise)，不受载具运动影响
    taus, gyro_adev = allan_deviation(gyro_error, rate_hz)
    _, accel_adev = allan_deviation(accel_error, rate_hz)
    plot_allan(taus, gyro_adev,
               params["gyroscope_noise_density"],
               params["gyroscope_random_walk"],
               "Gyroscope Allan Deviation", output_dir / "allan_gyro.png")
    plot_allan(taus, accel_adev,
               params["accelerometer_noise_density"],
               params["accelerometer_random_walk"],
               "Accelerometer Allan Deviation", output_dir / "allan_accel.png")

    print(f"\n[INFO] 图像已输出至 {output_dir}:")
    for name in ("bias_random_walk.png", "white_noise_histogram.png",
                 "allan_gyro.png", "allan_accel.png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
