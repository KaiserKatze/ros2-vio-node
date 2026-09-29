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
python3 -m pip install optuna evo numpy pandas matplotlib pyyaml seaborn

# 生成仿真数据（默认相机间隔 0.05 秒，即 20 Hz）
ros2 run euroc_vio VisualSim
# 指定相机拍照间隔（秒）或帧率（Hz），两种参数只能选择一种
ros2 run euroc_vio VisualSim --camera-interval 0.1
ros2 run euroc_vio VisualSim --camera-fps 10
# 指定长方体房间的进深、开间、层高（单位：米），可与相机参数组合
ros2 run euroc_vio VisualSim \
  --room-depth 12 --room-width 8 --room-height 4 --camera-fps 10
# 查看命令行帮助
ros2 run euroc_vio VisualSim --help
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
  --config $(ros2 pkg prefix euroc_vio)/lib/euroc_vio/ehat.yaml \
  --estimated-motion-csv estimated_motion_sim_cam1.csv
# 优化 MSCKF 超参数
ros2 run euroc_vio mhat.py \
  --config $(ros2 pkg prefix euroc_vio)/lib/euroc_vio/mhat.yaml
# MSCKF 单次评估 (调优器每次迭代内部执行的等价命令; 13 个可调超参数见 SRS 附录 6.1):
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --init-gravity imu \
  --feature-type orb --output /tmp/trial.tum --pointcloud /tmp/trial.ply \
  --pixel-noise-sigma=1.5 --min-parallax=0.01 --huber-threshold=0.01
# 运行 MSCKF
ros2 run euroc_vio msckf ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/ --init groundtruth --mono-csv=estimated_motion_cam0.csv
ros2 run euroc_vio msckf /tmp/imu_noise_analysis/mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=./mav0/estimated_motion_sim1.csv
ros2 run euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
ros2 run --prefix 'gdb -ex run --args' euroc_vio msckf ./mav0/ --init groundtruth --mono-csv=/tmp/nowhere/nothing
```

相机间隔和帧率必须为有限正数；不允许同时指定或重复指定。
IMU 和真值采样率保持为相机帧率的 10 倍，例如相机 10 Hz 时为 100 Hz。
生成数据的时间戳及 `sensor.yaml` 中的采样率会随参数同步调整。

房间始终为长方体，`--room-depth`（X 方向进深）、`--room-width`（Y 方向开间）、
`--room-height`（Z 方向层高）可独立指定，默认分别为 10、10、3 米。
尺寸必须为有限正数，同一参数不可重复指定。房间中心、路标点和相机运动轨迹
会根据尺寸自动调整，实际尺寸记录在生成的 `mav0/README.txt` 中。
网格划分段数保持为进深 20、开间 20、层高 6，网格间距随尺寸变化。

`mono.py` 将轨迹缓存记录保存到 `~/vio_ws/.mono_trajectory_cache.json`。
每次启动都会比较相关文件的 LastModified（纳秒）、大小和 MD5，以及估计参数。
检查范围包括六个输入 CSV/YAML、`mono.py`、`VisualInertial` 可执行文件、
已安装头文件，以及源码目录中的 `VisualInertial.cpp`、相关头文件和构建配置。
源码目录优先由 `EUROC_VIO_SOURCE_DIR` 指定，否则从当前脚本位置、
`~/vio_ws/build/euroc_vio/CMakeCache.txt` 和 `~/vio_ws/src` 查找。
找不到源码时会给出警告，并仅检查可用的已安装代码与输入资源。

依赖和参数未变、所有启用估计器的轨迹文件也与缓存记录一致时，
直接启动轨迹加载器和 RViz。首次运行、缓存缺失/损坏、依赖变化或轨迹被修改/删除时，
重新执行估计；仅在进程成功退出、全部轨迹已更新且非空、运行期间依赖未变时保存缓存。
调试模式始终重新执行且不保存缓存。此机制不会调用 C++ 编译；源码更新后运行的仍是
当前已安装的可执行文件，需要新版算法时请自行构建。

缓存逻辑可在不安装 ROS2、不编译 C++ 的环境中验证：
`python -m unittest discover -s test -p test_mono_cache.py`。
