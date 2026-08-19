# 使用方法
```bash
# 编译
rm -rf build install log
colcon build --packages-select euroc_vio \
  --parallel-workers $(nproc) \
  --cmake-args -G Ninja \
    -D OpenCV_DIR=/usr/local/lib/cmake/opencv4 \
    -D ENABLE_TIMER=1 \
    -D ENABLE_TIME_LOGGER=1 \
  --event-handlers console_direct+
# 激活 ROS2 运行环境
source ~/vio_ws/install/local_setup.sh

# 创建并激活 Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install optuna evo numpy pandas matplotlib pyyaml

# 生成仿真数据
ros2 run euroc_vio VisualSim
# 为 imu0/data.csv 注入高斯白噪声与随机游走并同步 sensor.yaml
ros2 run euroc_vio ImuNoiseModel ./mav0/
# 运行单目惯性里程计
ros2 launch euroc_vio mono.py
# 运行双目惯性里程计
ros2 run euroc_vio StereoSlam ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/
ros2 run euroc_vio StereoSlam --visualize ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/
ros2 run --prefix 'gdb -ex run --args' euroc_vio StereoSlam ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/
ros2 launch euroc_vio stereo.py
# 查看活跃话题列表及其消息类型
ros2 topic list -t
# 查看指定话题 (真值轨迹)
ros2 topic echo /traj/ground_truth/path nav_msgs/msg/Path
ros2 topic echo /traj/stereo_est/path nav_msgs/msg/Path
# 优化 ESKF 超参数
ros2 run euroc_vio ehat.py \
  --config $(ros2 pkg prefix euroc_vio)/lib/euroc_vio/ehat.yaml
# 优化 MSCKF 超参数
ros2 run euroc_vio mhat.py \
  --config $(ros2 pkg prefix euroc_vio)/lib/euroc_vio/mhat.yaml
# MSCKF 单次评估 (调优器每次迭代内部执行的等价命令; 13 个可调超参数见 SRS 附录 6.1):
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --init-gravity imu \
  --output /tmp/trial.tum --pointcloud /tmp/trial.ply \
  --pixel-noise-sigma=1.5 --min-parallax=0.01 --huber-threshold=0.01
# 运行 MSCKF
ros2 run euroc_vio msckf ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/ --init groundtruth --mono-csv=estimated_motion_cam0.csv
ros2 run euroc_vio msckf /tmp/imu_noise_analysis/mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=./mav0/estimated_motion_sim1.csv
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
ros2 run --prefix 'gdb -ex run --args' euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
```
