# 使用方法
```bash
# 编译
rm -rf build install log
colcon build --packages-select euroc_vio \
  --parallel-workers $(nproc) \
  --cmake-args -G Ninja \
    -D OpenCV_DIR=/usr/local/lib/cmake/opencv4 \
  --event-handlers console_direct+
source ~/vio_ws/install/local_setup.sh

# 生成仿真数据
ros2 run euroc_vio VisualSim
# 运行单目惯性里程计
ros2 launch euroc_vio mono.py
# 运行双目惯性里程计
ros2 run euroc_vio StereoSlam --visualize ~/EuRoC_MAV_Datasets/V2_01_easy/mav0/
ros2 launch euroc_vio stereo.py
# 查看活跃话题列表及其消息类型
ros2 topic list -t
# 查看指定话题 (真值轨迹)
ros2 topic echo /ground_truth/path nav_msgs/msg/Path
# 为单目惯性里程计优化 ESKF 超参数
ros2 run euroc_vio opt.py --config config.yaml
```
