# 使用方法
```bash
# 编译
rm -rf build install log
colcon build --packages-select euroc_vio \
  --parallel-workers $(nproc) \
  --cmake-args \
    -D OpenCV_DIR=/usr/local/lib/cmake/opencv4 \
  --event-handlers console_direct+
source ~/vio_ws/install/local_setup.sh

# 生成仿真数据（默认相机间隔 0.05 秒，即 20 Hz）
ros2 run euroc_vio VisualSim
# 指定相机拍照间隔（秒）或帧率（Hz），两种参数只能选择一种
ros2 run euroc_vio VisualSim --camera-interval 0.1
ros2 run euroc_vio VisualSim --camera-fps 10
# 查看命令行帮助
ros2 run euroc_vio VisualSim --help
# 运行单目惯性里程计
ros2 launch euroc_vio mono.py
# 运行双目惯性里程计
ros2 run euroc_vio StereoSlam --visualize mav0
ros2 launch euroc_vio stereo.py
# 查看活跃话题列表及其消息类型
ros2 topic list -t
# 查看指定话题 (真值轨迹)
ros2 topic echo /ground_truth/path nav_msgs/msg/Path
# 为单目惯性里程计优化 ESKF 超参数
ros2 run euroc_vio opt.py --config config.yaml
```

相机间隔和帧率必须为有限正数；不允许同时指定或重复指定。
IMU 和真值采样率保持为相机帧率的 10 倍，例如相机 10 Hz 时为 100 Hz。
生成数据的时间戳及 `sensor.yaml` 中的采样率会随参数同步调整。
