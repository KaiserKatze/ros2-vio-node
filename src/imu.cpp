// clang-format on

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/ahrs.hpp"

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

/*****************************
 * 常数与类型定义
 *****************************/

using MsgImu         = sensor_msgs::msg::Imu;
using MsgGroundTruth = geometry_msgs::msg::TransformStamped;
using MsgPath        = nav_msgs::msg::Path;

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
struct state_type : public std::array<double, 10>
{
  state_type()
      : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
  {
  }

  void SetQuaternion(double qw, double qx, double qy, double qz)
  {
    (*this)[6] = qw;
    (*this)[7] = qx;
    (*this)[8] = qy;
    (*this)[9] = qz;
  }

  void SetQuaternion(const cv::Vec4f &quat)
  {
    (*this)[6] = quat[0];
    (*this)[7] = quat[1];
    (*this)[8] = quat[2];
    (*this)[9] = quat[3];
  }
};

/*****************************
 * RC 低通滤波器类
 *****************************/
class RCFilter
{
private:
  double alpha_;
  bool initialized_;
  cv::Vec3d prev_val_;

public:
  RCFilter(double alpha = 0.1)
      : alpha_(alpha), initialized_(false), prev_val_(0, 0, 0)
  {
  }

  cv::Vec3d filter(const cv::Vec3d &val)
  {
    if (!initialized_)
    {
      prev_val_    = val;
      initialized_ = true;
      return val;
    }
    // 一阶低通滤波公式: Y_n = alpha * X_n + (1 - alpha) * Y_{n-1}
    prev_val_ = alpha_ * val + (1.0 - alpha_) * prev_val_;
    return prev_val_;
  }
};

/*****************************
 * IMU 运动学 ODE (常微分方程) 系统
 * 用于给 boost::numeric::odeint 进行 RK4 积分
 *****************************/
struct ImuKinematicsODE
{
  cv::Vec3d a;          // 经过滤波后的加速度 (机体坐标系)
  cv::Vec3d w;          // 经过滤波后的角速度 (机体坐标系)
  const double g{9.81}; // 重力加速度常数 (默认 Z 轴朝上)

  ImuKinematicsODE(const cv::Vec3d &accel, const cv::Vec3d &gyro,
                   const cv::Vec3d &ba, const cv::Vec3d &bw)
      : a(accel - ba), w(gyro - bw)
  {
  }

  void operator()(const state_type &x, state_type &dxdt,
                  const double /* t */) const
  {
    // 提取当前姿态四元数
    const double qw{x[6]};
    const double qx{x[7]};
    const double qy{x[8]};
    const double qz{x[9]};

    // 四元数转旋转矩阵 R (用于将机体加速度转换到世界坐标系)
    const double R00{1.0 - 2.0 * qy * qy - 2.0 * qz * qz};
    const double R01{2.0 * (qx * qy - qw * qz)};
    const double R02{2.0 * (qx * qz + qw * qy)};
    const double R10{2.0 * (qx * qy + qw * qz)};
    const double R11{1.0 - 2.0 * qx * qx - 2.0 * qz * qz};
    const double R12{2.0 * (qy * qz - qw * qx)};
    const double R20{2.0 * (qx * qz - qw * qy)};
    const double R21{2.0 * (qy * qz + qw * qx)};
    const double R22{1.0 - 2.0 * qx * qx - 2.0 * qy * qy};

    // 位置导数 = 速度
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];

    // EuRoC MAV 数据集中前几个 IMU 数据：
    // a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
    // 9.1283567083333317,0.10623870833333333,-2.6069344583333329
    // 9.422556208333333,1.1604535833333331,-2.623278875

    // 可以看出 X 轴的加速度约为 9.8 m/s^2，Y 轴和 Z 轴的加速度较小，说明重力主要作用在 X 轴上
    // X 轴的加速度值是正数，而 IMU 测量的是比力，恰好验证了数据说明书所说的“IMU 的 X 轴朝上”
    // 数据说明书还说 Z 轴是朝着 MAV 正前方（即双目相机的视线方向），Y 轴则是朝向 MAV 的右侧（右手坐标系）

    // 速度导数 = 加速度
    dxdt[3] = R00 * a[0] + R01 * a[1] + R02 * a[2] - g;
    dxdt[4] = R10 * a[0] + R11 * a[1] + R12 * a[2];
    dxdt[5] = R20 * a[0] + R21 * a[1] + R22 * a[2];

    // 姿态导数
    dxdt[6] = 0.5 * (-qx * w[0] - qy * w[1] - qz * w[2]);
    dxdt[7] = 0.5 * (qw * w[0] + qy * w[2] - qz * w[1]);
    dxdt[8] = 0.5 * (qw * w[1] - qx * w[2] + qz * w[0]);
    dxdt[9] = 0.5 * (qw * w[2] + qx * w[1] - qy * w[0]);
  }
};

enum class EstimatorType
{
  RK4,
  MAHONY,
  MADGWICK
};

/*****************************
 * IMU 数据处理单元
 *****************************/
class ImuWorker
{
private:
  bool use_filter_;
  RCFilter accel_filter_;
  RCFilter gyro_filter_;

  EstimatorType estimator_type_;

  bool first_msg_   = true;
  double last_time_ = 0.0;

  // 状态向量初始化 (初始原点, 速度为 0, 姿态为单位四元数)
  state_type state_;
  double ode_time{0.0};

  runge_kutta4<state_type> rk4_;
  std::shared_ptr<AbstractAHRS> ahrs_;

  // 零偏
  cv::Vec3d gyro_bias_{0, 0, 0};
  cv::Vec3d accel_bias_{0, 0, 0};

  // 初始重力对齐缓存
  std::vector<cv::Vec3d> init_acc_buffer_;

public:
  ImuWorker(bool use_filter         = true,
            EstimatorType estimator = EstimatorType::RK4, double init_px = 0.0,
            double init_py = 0.0, double init_pz = 0.0)
      : use_filter_(use_filter), accel_filter_(0.15), gyro_filter_(0.15),
        estimator_type_(estimator)
  {
    // 初始化状态向量
    state_[0] = init_px;
    state_[1] = init_py;
    state_[2] = init_pz;
    // 其他状态保持默认

    if (estimator == EstimatorType::RK4)
    {
      ahrs_ = nullptr; // RK4 不需要 AHRS
    }
    else if (estimator == EstimatorType::MAHONY)
    {
      ahrs_ = std::make_shared<MahonyAHRS>();
    }
    else if (estimator == EstimatorType::MADGWICK)
    {
      ahrs_ = std::make_shared<MadgwickAHRS>();
    }
    else
    {
      throw std::invalid_argument{"Unsupported estimator type"};
    }
  }

  void RK4Update(const cv::Vec3d &acc, const cv::Vec3d &gyro, double dt)
  {
    ImuKinematicsODE ode{acc, gyro, accel_bias_, gyro_bias_};
    rk4_.do_step(ode, state_, ode_time, dt);
    ode_time += dt;
  }

  void MahonyUpdate(const cv::Vec3d &acc, const cv::Vec3d &gyro, double dt)
  {
    // 利用 Mahony 算法或 Madgwick 算法，计算姿态四元数
    cv::Vec3f vecAccel{static_cast<cv::Vec3f>(acc)};
    cv::Vec3f vecGyro{static_cast<cv::Vec3f>(gyro)};
    ahrs_->Update(vecGyro, vecAccel, dt);
    // 更新姿态四元数
    this->state_.SetQuaternion(ahrs_->GetQuaternion());
    // 用龙格库塔法计算速度、位置
    RK4Update(vecAccel, vecGyro, dt);
  }

  void HeavyLift(const cv::Vec3d &filt_accel, const cv::Vec3d &filt_gyro,
                 double dt)
  {
    switch (estimator_type_)
    {
    case EstimatorType::RK4:
      RK4Update(filt_accel, filt_gyro, dt);
      break;
    case EstimatorType::MAHONY:
    case EstimatorType::MADGWICK:
      MahonyUpdate(filt_accel, filt_gyro, dt);
      break;
    default:
      break;
    }
  }

  std::optional<geometry_msgs::msg::PoseStamped>
  Work(const MsgImu::ConstSharedPtr &imu_msg, rclcpp::Logger logger)
  {
    const double current_time{imu_msg->header.stamp.sec
                              + imu_msg->header.stamp.nanosec * 1e-9};

    const double gx{imu_msg->angular_velocity.x};
    const double gy{imu_msg->angular_velocity.y};
    const double gz{imu_msg->angular_velocity.z};
    const double ax{imu_msg->linear_acceleration.x};
    const double ay{imu_msg->linear_acceleration.y};
    const double az{imu_msg->linear_acceleration.z};

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp       = imu_msg->header.stamp;
    pose_msg.header.frame_id    = "world";
    pose_msg.pose.position.x    = 0.0;
    pose_msg.pose.position.y    = 0.0;
    pose_msg.pose.position.z    = 0.0;
    pose_msg.pose.orientation.w = 1.0;
    pose_msg.pose.orientation.x = 0.0;
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = 0.0;

    // 读取原始数据
    cv::Vec3d raw_accel(ax, ay, az);
    cv::Vec3d raw_gyro(gx, gy, gz);

    // 1. RC 低通滤波
    const cv::Vec3d filt_accel{use_filter_ ? accel_filter_.filter(raw_accel)
                                           : raw_accel};
    const cv::Vec3d filt_gyro{use_filter_ ? gyro_filter_.filter(raw_gyro)
                                          : raw_gyro};

    // 初始重力对齐
    if (first_msg_)
    {
      init_acc_buffer_.push_back(raw_accel);

      // 等积累了足够的初始加速度数据后，计算初始姿态
      if (init_acc_buffer_.size() < 5)
      {
        return pose_msg;
      }

      // 计算平均加速度向量，作为重力方向的估计
      cv::Vec3d g_body(0.0, 0.0, 0.0);
      for (const cv::Vec3d &a : init_acc_buffer_)
      {
        g_body += a;
      }
      g_body /= static_cast<double>(init_acc_buffer_.size());
      g_body /= cv::norm(g_body);

      // 自适应判断重力方向（假设初始时刻 MAV 是静止的，且加速度主要来自重力）
      const double acc_x{std::abs(g_body[0])};
      const double acc_y{std::abs(g_body[1])};
      const double acc_z{std::abs(g_body[2])};
      if (!(acc_x > acc_y && acc_x > acc_z))
      {
        throw std::runtime_error{"Assertion failed: X-axis should have the "
                                 "largest acceleration component at rest"};
      }
      cv::Vec3d g_world(-1.0, 0.0, 0.0);

      cv::Vec3d v{g_body.cross(g_world)};
      double c{g_body.dot(g_world)};
      double s{cv::norm(v)};

      cv::Mat K{(cv::Mat_<double>(3, 3) << 0, -v[2], v[1], v[2], 0, -v[0],
                 -v[1], v[0], 0)};

      cv::Mat R{cv::Mat::eye(3, 3, CV_64F) + K
                + K * K * ((1 - c) / (s * s + 1e-8))};

      // 转 quaternion（简化）
      double qw{std::sqrt(1 + R.at<double>(0, 0) + R.at<double>(1, 1)
                          + R.at<double>(2, 2))
                * 0.5};
      double qx{0.0}, qy{0.0}, qz{0.0};
      if (std::abs(qw) < 1e-8)
      {
        // 特殊情况：qw ≈ 0（180°旋转），使用备用公式
        qw = 0.0;
        qx = std::sqrt(std::max(0.0, (1 + R.at<double>(0, 0)
                                      - R.at<double>(1, 1) - R.at<double>(2, 2))
                                         / 4.0));
        qy = std::sqrt(std::max(0.0, (1 - R.at<double>(0, 0)
                                      + R.at<double>(1, 1) - R.at<double>(2, 2))
                                         / 4.0));
        qz = std::sqrt(std::max(0.0, (1 - R.at<double>(0, 0)
                                      - R.at<double>(1, 1) + R.at<double>(2, 2))
                                         / 4.0));

        // 符号调整（基于旋转矩阵的对称元素，确保一致性）
        if (R.at<double>(2, 1) < 0)
        {
          qx = -qx;
        }
        if (R.at<double>(0, 2) < 0)
        {
          qy = -qy;
        }
        if (R.at<double>(1, 0) < 0)
        {
          qz = -qz;
        }
      }
      else
      {
        // 正常情况
        qx = (R.at<double>(2, 1) - R.at<double>(1, 2)) / (4 * qw);
        qy = (R.at<double>(0, 2) - R.at<double>(2, 0)) / (4 * qw);
        qz = (R.at<double>(1, 0) - R.at<double>(0, 1)) / (4 * qw);
      }

      state_[6] = qw;
      state_[7] = qx;
      state_[8] = qy;
      state_[9] = qz;

      last_time_ = current_time;
      first_msg_ = false;

      return pose_msg;
    }

    const double dt{current_time - last_time_};
    last_time_ = current_time;

    // 防御性检查：跳过时间戳异常的数据
    if (dt <= 0 || dt > 0.1)
    {
      return {}; // 时间间隔异常，可能是数据乱序或丢包，跳过处理
    }

    // ⭐ 静止检测
    double acc_norm{cv::norm(filt_accel)};
    double gyro_norm{cv::norm(filt_gyro)};

    bool is_static = (std::abs(acc_norm - 9.81) < 0.1) && (gyro_norm < 0.02);

    // ⭐ ZUPT + bias 更新
    if (is_static)
    {
      state_[3] = 0;
      state_[4] = 0;
      state_[5] = 0;

      double alpha = 0.01;
      gyro_bias_   = (1 - alpha) * gyro_bias_ + alpha * filt_gyro;

      accel_bias_ = (1 - alpha) * accel_bias_
                    + alpha * (filt_accel - cv::Vec3d(0, 0, -9.81));
    }

    // 2. 利用 RK4 进行积分推算
    HeavyLift(filt_accel, filt_gyro, dt);

    double px{state_[0]}, py{state_[1]}, pz{state_[2]};
    double vx{state_[3]}, vy{state_[4]}, vz{state_[5]};
    double &qw{state_[6]};
    double &qx{state_[7]};
    double &qy{state_[8]};
    double &qz{state_[9]};

    // 3. 积分后必须对四元数进行归一化，因为 RK4 不保证单位模长约束
    const double q_norm{std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz)};
    qw /= q_norm;
    qx /= q_norm;
    qy /= q_norm;
    qz /= q_norm;

    // 输出当前航位推算的结果
    RCLCPP_INFO_THROTTLE(
        logger, *rclcpp::Clock::make_shared(), 500,
        "IMU State => Gyro:[%.18f, %.18f, %.18f], Accel:[%.18f, %.18f, %.18f], "
        "Pos:[%.9f, %.9f, %.9f] Vel:[%.9f, %.9f, %.9f], Att:[%.9f, %.9f, %.9f, "
        "%.9f]",
        gx, gy, gz, ax, ay, az, px, py, pz, vx, vy, vz, qw, qx, qy, qz);

    pose_msg.pose.position.x    = px;
    pose_msg.pose.position.y    = py;
    pose_msg.pose.position.z    = pz;
    pose_msg.pose.orientation.w = qw;
    pose_msg.pose.orientation.x = qx;
    pose_msg.pose.orientation.y = qy;
    pose_msg.pose.orientation.z = qz;

    return pose_msg;
  }
};

/*****************************
 * ROS2 Node
 *****************************/
class ImuNode : public rclcpp::Node
{
private:
  rclcpp::Subscription<MsgImu>::SharedPtr imu_sub_direct;
  rclcpp::Subscription<MsgGroundTruth>::SharedPtr gt_sub_direct;
  rclcpp::Publisher<MsgPath>::SharedPtr imu_pub;
  rclcpp::Publisher<MsgPath>::SharedPtr gt_pub;
  MsgPath path_msg_imu;
  MsgPath path_msg_groundtruth;

  ImuWorker imu_worker_;

  void ImuCallback(const MsgImu::ConstSharedPtr &imu_msg)
  {
    auto msg{imu_worker_.Work(imu_msg, this->get_logger())};
    // 发布处理后的 IMU 数据
    if (msg)
    {
      this->path_msg_imu.header.stamp = imu_msg->header.stamp;
      this->path_msg_imu.poses.push_back(*msg);
      imu_pub->publish(this->path_msg_imu);
    }
  }

  void GroundTruthCallback(const MsgGroundTruth::ConstSharedPtr &gt_msg)
  {
    this->path_msg_groundtruth.header.stamp = gt_msg->header.stamp;
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp       = gt_msg->header.stamp;
    msg.header.frame_id    = "world";
    msg.pose.position.x    = gt_msg->transform.translation.x;
    msg.pose.position.y    = gt_msg->transform.translation.y;
    msg.pose.position.z    = gt_msg->transform.translation.z;
    msg.pose.orientation.w = gt_msg->transform.rotation.w;
    msg.pose.orientation.x = gt_msg->transform.rotation.x;
    msg.pose.orientation.y = gt_msg->transform.rotation.y;
    msg.pose.orientation.z = gt_msg->transform.rotation.z;
    this->path_msg_groundtruth.poses.push_back(std::move(msg));
    gt_pub->publish(this->path_msg_groundtruth);
  }

public:
  ImuNode(const char *input_imu_topic, const char *input_groundtruth_topic,
          const char *output_imu_topic, const char *output_groundtruth_topic)
      : Node("VIO")
  {
    this->declare_parameter("estimator", "rk4");
    this->declare_parameter("use_filter", true);
    this->declare_parameter("initial_position_x", 0.0);
    this->declare_parameter("initial_position_y", 0.0);
    this->declare_parameter("initial_position_z", 0.0);
    std::string estimator_str{this->get_parameter("estimator").as_string()};
    bool use_filter{this->get_parameter("use_filter").as_bool()};
    double init_px{this->get_parameter("initial_position_x").as_double()};
    double init_py{this->get_parameter("initial_position_y").as_double()};
    double init_pz{this->get_parameter("initial_position_z").as_double()};

    EstimatorType estimator;
    if (estimator_str == "rk4")
    {
      estimator = EstimatorType::RK4;
    }
    else if (estimator_str == "mahony")
    {
      estimator = EstimatorType::MAHONY;
    }
    else if (estimator_str == "madgwick")
    {
      estimator = EstimatorType::MADGWICK;
    }
    else
    {
      throw std::invalid_argument("Invalid estimator type: " + estimator_str);
    }
    imu_worker_ = ImuWorker(use_filter, estimator, init_px, init_py, init_pz);

    // 设置路径消息的坐标系
    path_msg_imu.header.frame_id         = "world";
    path_msg_groundtruth.header.frame_id = "world";

    const rclcpp::QoS qos(10);

    using std::placeholders::_1;

    imu_sub_direct = this->create_subscription<MsgImu>(
        input_imu_topic, qos, std::bind(&ImuNode::ImuCallback, this, _1));

    gt_sub_direct = this->create_subscription<MsgGroundTruth>(
        input_groundtruth_topic, qos,
        std::bind(&ImuNode::GroundTruthCallback, this, _1));

    imu_pub = create_publisher<MsgPath>(output_imu_topic, qos);
    gt_pub  = create_publisher<MsgPath>(output_groundtruth_topic, qos);
  }
};

int main(int argc, char **argv)
{
  std::cout << "Node 'euroc_imu' started\n";
  rclcpp::init(argc, argv);

  auto node{std::make_shared<ImuNode>("/imu0", "/vicon/firefly_sbx/firefly_sbx",
                                      "/path_imu", "/path_groundtruth")};

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
