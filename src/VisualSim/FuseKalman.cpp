// 基于松耦合的线性卡尔曼滤波，融合单目视觉与 IMU 数据

#include <Eigen/Dense>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "LinearKalmanFilter.hpp"
#include "euroc_vio/main.h"

struct FuseKalman : public rclcpp::Node
{
  const std::string imu_topic_name_{"/imu0"};
  // 单目视觉估计的姿态 (位置向量和朝向四元数)
  const std::string cam_topic_name_{"/path_fast_est"};
  // 融合单目视觉和 IMU 数据以后输出的估计轨迹
  const std::string est_path_topic_name_{"/path_fuse_est"};
  // 融合单目视觉和 IMU 数据以后输出的估计姿态
  const std::string est_pose_topic_name_{"/pose_fuse_est"};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscriber_imu_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscriber_pose_cam_;

  nav_msgs::msg::Path path_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_est_;
  nav_msgs::msg::Path pose_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_est_;

  rclcpp::TimerBase::SharedPtr timer_;

  // 实例化的线性卡尔曼滤波对象
  LinearKalmanFilter filter_;

  FuseKalman() : Node("FuseKalman")
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);

    publisher_path_est_
        = create_publisher<nav_msgs::msg::Path>(est_path_topic_name_, qos);
    publisher_pose_est_
        = create_publisher<nav_msgs::msg::Path>(est_pose_topic_name_, qos);

    subscriber_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_name_, qos,
        std::bind(&FuseKalman::SubscriberImuCallback, this, _1));
    subscriber_pose_cam_ = create_subscription<nav_msgs::msg::Path>(
        cam_topic_name_, qos,
        std::bind(&FuseKalman::SubscriberCamCallback, this, _1));
    timer_ = create_wall_timer(std::chrono::seconds(1), // 1 Hz 发布
                               std::bind(&FuseKalman::PublishTrajectory, this));

    path_msg_.header.frame_id = DEFAULT_FRAME_ID;
    pose_msg_.header.frame_id = DEFAULT_FRAME_ID;
  }

  /**
   * @brief 将四元数转换为欧拉角 (Roll, Pitch, Yaw)
   * @param q 表示朝向的 Eigen 四元数
   * @return Eigen::Vector3d 对应的欧拉角 (roll, pitch, yaw) 向量，单位为弧度
   */
  Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond &q)
  {
    Eigen::Vector3d angles;
    // 计算 roll (绕 x 轴)
    double sinr_cosp{2 * (q.w() * q.x() + q.y() * q.z())};
    double cosr_cosp{1 - 2 * (q.x() * q.x() + q.y() * q.y())};
    angles.x() = std::atan2(sinr_cosp, cosr_cosp);

    // 计算 pitch (绕 y 轴)
    double sinp{2 * (q.w() * q.y() - q.z() * q.x())};
    if (std::abs(sinp) >= 1)
    {
      angles.y() = std::copysign(M_PI / 2, sinp); // 超出范围时修正为 90 度
    }
    else
    {
      angles.y() = std::asin(sinp);
    }

    // 计算 yaw (绕 z 轴)
    double siny_cosp{2 * (q.w() * q.z() + q.x() * q.y())};
    double cosy_cosp{1 - 2 * (q.y() * q.y() + q.z() * q.z())};
    angles.z() = std::atan2(siny_cosp, cosy_cosp);

    return angles;
  }

  /**
   * @brief 设置滤波器进入 IMU 测量模式
   * 动态调整 measurementMatrix 以映射到线加速度 (ax, ay, az) 与 角速度 (wx, wy, wz) 的对应系统状态，并设定适合的测量噪声
   */
  void SetImuMeasurementModel()
  {
    filter_.kf_.measurementMatrix.setTo(0);
    // 对应状态向量的线加速度
    filter_.kf_.measurementMatrix.at<double>(0, 6) = 1; // ax
    filter_.kf_.measurementMatrix.at<double>(1, 7) = 1; // ay
    filter_.kf_.measurementMatrix.at<double>(2, 8) = 1; // az
    // 对应状态向量的角速度
    filter_.kf_.measurementMatrix.at<double>(3, 12) = 1; // wx
    filter_.kf_.measurementMatrix.at<double>(4, 13) = 1; // wy
    filter_.kf_.measurementMatrix.at<double>(5, 14) = 1; // wz

    // 设置针对高频传感器的测量协方差噪声
    cv::setIdentity(filter_.kf_.measurementNoiseCov, cv::Scalar::all(1e-2));
  }

  /**
   * @brief 设置滤波器进入相机测量模式
   * 动态调整 measurementMatrix 以映射到空间位置 (x, y, z) 与 欧拉角朝向 (roll, pitch, yaw) 的对应系统状态，并设定适合的测量噪声
   */
  void SetCamMeasurementModel()
  {
    filter_.kf_.measurementMatrix.setTo(0);
    // 对应状态向量的全局位置
    filter_.kf_.measurementMatrix.at<double>(0, 0) = 1; // x
    filter_.kf_.measurementMatrix.at<double>(1, 1) = 1; // y
    filter_.kf_.measurementMatrix.at<double>(2, 2) = 1; // z
    // 对应状态向量的姿态角
    filter_.kf_.measurementMatrix.at<double>(3, 9)  = 1; // roll
    filter_.kf_.measurementMatrix.at<double>(4, 10) = 1; // pitch
    filter_.kf_.measurementMatrix.at<double>(5, 11) = 1; // yaw

    // 设置针对视觉估计的高精度传感器测量协方差噪声 (噪声相比 IMU 更小)
    cv::setIdentity(filter_.kf_.measurementNoiseCov, cv::Scalar::all(1e-4));
  }

  /**
   * @brief IMU 测量数据的 ROS 回调函数
   * @param msg 来自 /imu0 话题的传感器数据，包含当前机体线加速度与角速度
   */
  void SubscriberImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    // 1. 在 IMU 高频帧上驱动线性卡尔曼滤波器执行状态演进
    filter_.kf_.predict();

    // 提取当前模型预测出的欧拉角姿态，以便后续计算世界坐标转换
    double roll{filter_.kf_.statePre.at<double>(9)};
    double pitch{filter_.kf_.statePre.at<double>(10)};
    double yaw{filter_.kf_.statePre.at<double>(11)};

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q{yawAngle * pitchAngle * rollAngle};

    // 获取机体自身坐标系下的加速度和角速度输入
    Eigen::Vector3d a_body(msg->linear_acceleration.x,
                           msg->linear_acceleration.y,
                           msg->linear_acceleration.z);
    Eigen::Vector3d w_body(msg->angular_velocity.x, msg->angular_velocity.y,
                           msg->angular_velocity.z);

    // 将加速度从机体坐标系转换到世界全局坐标系，并剔除重力影响 (设 Z 轴垂直向上，重力常数 9.81 m/s^2)
    Eigen::Vector3d a_world{q * a_body};
    a_world(2) -= 9.81;

    // 采用非奇异变换矩阵近似将机体角速度转换至当前系统状态向量依赖的欧拉角速率
    double sr{std::sin(roll)};
    double cr{std::cos(roll)};
    double sp{std::sin(pitch)};
    double cp{std::cos(pitch)};
    Eigen::Matrix3d T;
    if (std::abs(cp) > 1e-4)
    {
      T << 1, sr * sp / cp, cr * sp / cp, 0, cr, -sr, 0, sr / cp, cr / cp;
    }
    else
    {
      T = Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d euler_rates{T * w_body};

    // 2. 将测量矩阵和噪声重配置到 IMU 维度
    SetImuMeasurementModel();
    cv::Mat measurement{cv::Mat::zeros(6, 1, CV_64F)};
    measurement.at<double>(0) = a_world.x();
    measurement.at<double>(1) = a_world.y();
    measurement.at<double>(2) = a_world.z();
    measurement.at<double>(3) = euler_rates.x();
    measurement.at<double>(4) = euler_rates.y();
    measurement.at<double>(5) = euler_rates.z();

    // 修正卡尔曼滤波后验状态
    filter_.kf_.correct(measurement);

    // 3. 构建发布需要的最新的平移坐标和旋转朝向
    double est_x{filter_.kf_.statePost.at<double>(0)};
    double est_y{filter_.kf_.statePost.at<double>(1)};
    double est_z{filter_.kf_.statePost.at<double>(2)};
    double est_roll{filter_.kf_.statePost.at<double>(9)};
    double est_pitch{filter_.kf_.statePost.at<double>(10)};
    double est_yaw{filter_.kf_.statePost.at<double>(11)};

    Eigen::AngleAxisd est_rollAngle(est_roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd est_pitchAngle(est_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd est_yawAngle(est_yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond est_q{est_yawAngle * est_pitchAngle * est_rollAngle};
    Eigen::Vector3d est_p(est_x, est_y, est_z);

    rclcpp::Time now_time{msg->header.stamp};
    CreateFusedPose(now_time.nanoseconds(), est_p, est_q);
  }

  /**
   * @brief 相机视觉位姿数据的 ROS 回调函数
   * @param msg 来自单目相机的连续定位路径，仅取最后一次预测
   */
  void SubscriberCamCallback(const nav_msgs::msg::Path::ConstSharedPtr msg)
  {
    if (msg->poses.empty())
    {
      return;
    }

    // 取出最新的视觉估计位姿基准
    const auto &latest_pose{msg->poses.back().pose}; // geometry_msgs::msg::Pose

    double x{latest_pose.position.x};
    double y{latest_pose.position.y};
    double z{latest_pose.position.z};

    Eigen::Quaterniond q(latest_pose.orientation.w, latest_pose.orientation.x,
                         latest_pose.orientation.y, latest_pose.orientation.z);
    Eigen::Vector3d euler{QuaternionToEuler(q)};

    // 强制将当前的 KF 后验修正为先验，因为此帧相机测量极大概率在相同的时间步长下和 IMU 更新混合
    // 该策略允许多个异步校正 (correct) 基于同一先验不发生覆盖冲突
    filter_.kf_.statePre    = filter_.kf_.statePost.clone();
    filter_.kf_.errorCovPre = filter_.kf_.errorCovPost.clone();

    // 将测量模型适配至低频相机格式
    SetCamMeasurementModel();
    cv::Mat measurement{cv::Mat::zeros(6, 1, CV_64F)};
    measurement.at<double>(0) = x;
    measurement.at<double>(1) = y;
    measurement.at<double>(2) = z;
    measurement.at<double>(3) = euler.x();
    measurement.at<double>(4) = euler.y();
    measurement.at<double>(5) = euler.z();

    // 叠加相机精准位姿进行二次 correct
    filter_.kf_.correct(measurement);
  }

  // IMU 采样率是 200 Hz
  // 单目相机采样率 20 Hz，单目视觉估计姿态的输出数据率大约也是 20 Hz
  // 那么应该怎么融合不同频率的两种传感器的数据呢？
  // (注释补充)：在上面的实现中采取的是时间步长与最高频(200Hz IMU)对齐的策略，将 IMU 积分驱动 prediction。
  // 在获取 IMU 数据时进行 IMU 特定的正确测量 (Correct) 更新。
  // 在异步获取到低频视觉位姿时 (20Hz Camera)，临时修改卡尔曼测量映射阵 (measurementMatrix)，
  // 直接以带有极小不确定性的全局姿态叠加 Correct 修正，使得 200Hz 高频输出能无缝利用 20Hz 精准低频特征。

  void PublishTrajectory()
  {
    const auto timestamp{now()};
    path_msg_.header.stamp = timestamp;
    publisher_path_est_->publish(path_msg_);

    static size_t index{0};
    if (path_msg_.poses.empty())
    {
      return;
    }
    const auto &pose{path_msg_.poses[index]};
    pose_msg_.header.stamp = pose.header.stamp;
    pose_msg_.poses.clear();
    pose_msg_.poses.push_back(pose);
    index = (index + 1) % path_msg_.poses.size();
    publisher_pose_est_->publish(pose_msg_);
  }

  geometry_msgs::msg::PoseStamped
  CreateFusedPose(std::int64_t timestamp, const Eigen::Vector3d &position,
                  const Eigen::Quaterniond &attitude)
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id    = DEFAULT_FRAME_ID;
    msg.header.stamp       = rclcpp::Time{timestamp};
    msg.pose.position.x    = position.x();
    msg.pose.position.y    = position.y();
    msg.pose.position.z    = position.z();
    msg.pose.orientation.w = attitude.w();
    msg.pose.orientation.x = attitude.x();
    msg.pose.orientation.y = attitude.y();
    msg.pose.orientation.z = attitude.z();
    path_msg_.poses.push_back(msg);
    return msg;
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FuseKalman>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
