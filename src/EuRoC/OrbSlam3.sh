#!/usr/bin/bash

set -euxo pipefail

source ./.venv/bin/activate

#==========================================================
# 利用 ORB-SLAM3 估计 EuRoC MAV 数据集的无人机位姿
#==========================================================

# Variable

path_to_orbslam3=/home/ros/ros2-euroc-orbslam/externals/ORB_SLAM3
name_executable=Stereo-Inertial
path_to_sequence_folder_1=/home/ros/EuRoC_MAV_Datasets/V2_01_easy

# Computed

path_to_workdir=$path_to_orbslam3/Examples/$name_executable
path_to_executable=$(find $path_to_workdir -maxdepth 1 -iname *euroc)
path_to_vocabulary=$path_to_orbslam3/Vocabulary/ORBvoc.txt
path_to_settings=$path_to_workdir/EuRoC.yaml
path_to_times_file_1=$path_to_workdir/EuRoC_TimeStamps/$(echo "$path_to_sequence_folder_1" | awk -F'[/_]' '{print $(NF-2)$(NF-1)}').txt

if [ -f "CameraTrajectory.txt" ]; then
  echo "Trajectory already exists, skip ORB-SLAM3."
else
  echo "Start processing EuRoC MAV datasets with ORB-SLAM3 ..."
  time $path_to_executable $path_to_vocabulary $path_to_settings $path_to_sequence_folder_1 $path_to_times_file_1
  # 在当前目录下，会产生两个文件： CameraTrajectory.txt 和 KeyFrameTrajectory.txt
fi

#==========================================================
# 利用 evo 检验 ORB-SLAM3 估计位姿的准确性
#==========================================================

# Computed

path_to_dir_groundtruth=$path_to_sequence_folder_1/mav0/state_groundtruth_estimate0

# 1. 格式转换: EuRoC → TUM
echo "Ground truth info:"
ls -l "$path_to_dir_groundtruth"
head -n 2 "$path_to_dir_groundtruth/data.csv"

evo_traj euroc "$path_to_dir_groundtruth/data.csv" --save_as_tum
mv data.tum ground_truth.tum

echo "First lines of ground_truth.tum:"
head -n 2 ground_truth.tum

# 2. 单位转换: convert ns → seconds
echo "ORB-SLAM3 trajectory head:"
ls -l "$path_to_orbslam3"
head -n 2 "$path_to_orbslam3/CameraTrajectory.txt"

# Use printf to enforce 9 decimal places to prevent scientific notation truncation
awk '{ printf "%.9f %s %s %s %s %s %s %s\n", $1 / 1000000000.0, $2, $3, $4, $5, $6, $7, $8 }' "$path_to_orbslam3/CameraTrajectory.txt" > est_seconds.tum

# 3. 计算绝对位姿误差 (Absolute Pose Error, APE) 并绘制图像
# - -vas           → verbose (-v) + align (-a) + correct_scale (-s)
# - -r full        → full 6-DoF error (recommended for EuRoC)
# - --plot_mode xz → top-down view (common for MAV datasets)

# evo_ape tum ground_truth.tum est_seconds.tum \
#     -r full \
#     --plot \
#     --plot_mode xz \
#     -vas

# Optional: also show statistics without plot (quiet mode)
evo_ape tum ground_truth.tum est_seconds.tum -r full -as

# 4. 进行 Sim(3) 变换
yes "y" | evo_traj tum est_seconds.tum --ref=ground_truth.tum --align --correct_scale --save_as_tum
