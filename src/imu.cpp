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

#include "euroc_vio/msg/imu.hpp"
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

/*****************************
 * 常数与类型定义
 *****************************/

using InputMsgImu  = sensor_msgs::msg::Imu;
using OutputMsgImu = euroc_vio::msg::Imu;

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
using state_type = std::array<double, 10>;

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

  ImuKinematicsODE(const cv::Vec3d &accel, const cv::Vec3d &gyro)
      : a(accel), w(gyro)
  {
  }

  void operator()(const state_type &x, state_type &dxdt,
                  const double /* t */) const
  {
    // 提取当前姿态四元数
    double qw{x[6]}, qx{x[7]}, qy{x[8]}, qz{x[9]};

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

    // 位置导数 = 速度 (dp/dt = v)
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];

    // 速度导数 = 加速度 (dv/dt = R * a_body - g)
    dxdt[3] = R00 * a[0] + R01 * a[1] + R02 * a[2];
    dxdt[4] = R10 * a[0] + R11 * a[1] + R12 * a[2];
    dxdt[5] = R20 * a[0] + R21 * a[1] + R22 * a[2] - g;

    // 姿态导数 (dq/dt = 0.5 * q * w)
    dxdt[6] = 0.5 * (-qx * w[0] - qy * w[1] - qz * w[2]);
    dxdt[7] = 0.5 * (qw * w[0] + qy * w[2] - qz * w[1]);
    dxdt[8] = 0.5 * (qw * w[1] - qx * w[2] + qz * w[0]);
    dxdt[9] = 0.5 * (qw * w[2] + qx * w[1] - qy * w[0]);
  }
};

/*****************************
 * IMU 数据处理单元
 *****************************/
class ImuWorker
{
private:
  RCFilter accel_filter_;
  RCFilter gyro_filter_;

  bool first_msg_   = true;
  double last_time_ = 0.0;

  // 状态向量初始化 (初始原点, 速度为 0, 姿态为单位四元数)
  state_type state_ = {0.0, 0.0, 0.0,       // px, py, pz
                       0.0, 0.0, 0.0,       // vx, vy, vz
                       1.0, 0.0, 0.0, 0.0}; // qw, qx, qy, qz

  runge_kutta4<state_type> rk4_;

public:
  ImuWorker() : accel_filter_(0.15), gyro_filter_(0.15) {} // 可调节 alpha 参数

  std::optional<OutputMsgImu> Work(const InputMsgImu::ConstSharedPtr &imu_msg,
                                   rclcpp::Logger logger)
  {
    const double current_time{imu_msg->header.stamp.sec
                              + imu_msg->header.stamp.nanosec * 1e-9};

    const double gx{imu_msg->angular_velocity.x};
    const double gy{imu_msg->angular_velocity.y};
    const double gz{imu_msg->angular_velocity.z};
    const double ax{imu_msg->linear_acceleration.x};
    const double ay{imu_msg->linear_acceleration.y};
    const double az{imu_msg->linear_acceleration.z};

    // 读取原始数据
    cv::Vec3d raw_accel(ax, ay, az);
    cv::Vec3d raw_gyro(gx, gy, gz);

    // 1. RC 低通滤波
    const cv::Vec3d filt_accel{accel_filter_.filter(raw_accel)};
    const cv::Vec3d filt_gyro{gyro_filter_.filter(raw_gyro)};

    if (first_msg_)
    {
      last_time_ = current_time;
      first_msg_ = false;
      // TODO: 可以在这里添加基于重力的初始姿态对齐逻辑
      return {}; // 首条消息仅用于初始化，不进行积分推算
    }

    const double dt{current_time - last_time_};
    last_time_ = current_time;

    // 防御性检查：跳过时间戳异常的数据
    if (dt <= 0 || dt > 0.1)
    {
      return {}; // 时间间隔异常，可能是数据乱序或丢包，跳过处理
    }

    // 2. 利用 RK4 进行积分推算
    ImuKinematicsODE ode{filt_accel, filt_gyro};
    rk4_.do_step(ode, state_, 0.0, dt);

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

    OutputMsgImu msg;
    msg.header.stamp          = imu_msg->header.stamp;
    msg.linear_acceleration.x = ax;
    msg.linear_acceleration.y = ay;
    msg.linear_acceleration.z = az;
    msg.linear_velocity.x     = vx;
    msg.linear_velocity.y     = vy;
    msg.linear_velocity.z     = vz;
    msg.position.x            = px;
    msg.position.y            = py;
    msg.position.z            = pz;
    msg.angular_velocity.x    = gx;
    msg.angular_velocity.y    = gy;
    msg.angular_velocity.z    = gz;
    msg.orientation.w         = qw;
    msg.orientation.x         = qx;
    msg.orientation.y         = qy;
    msg.orientation.z         = qz;
    return msg;
  }
};

/*****************************
 * ROS2 Node
 *****************************/
class ImuNode : public rclcpp::Node
{
private:
  // IMU 直接采用标准的 Subscription 来保证高频接收
  rclcpp::Subscription<InputMsgImu>::SharedPtr imu_sub_direct;
  rclcpp::Publisher<OutputMsgImu>::SharedPtr imu_pub;

  ImuWorker imu_worker_;

  void ImuCallback(const InputMsgImu::ConstSharedPtr &imu_msg)
  {
    // 调用 IMU 处理单元，传入消息与 Logger
    const std::optional<OutputMsgImu> msg{
        imu_worker_.Work(imu_msg, this->get_logger())};
    // 发布处理后的 IMU 数据
    if (msg)
    {
      imu_pub->publish(*msg);
    }
  }

public:
  ImuNode(const char *input_imu_topic, const char *output_topic) : Node("VIO")
  {
    rclcpp::QoS qos(10);

    using std::placeholders::_1;

    // 独立且直接地订阅 IMU 话题，防止降频和丢包
    imu_sub_direct = this->create_subscription<InputMsgImu>(
        input_imu_topic, qos, std::bind(&ImuNode::ImuCallback, this, _1));

    imu_pub = create_publisher<OutputMsgImu>(output_topic, qos);
  }
};

int main(int argc, char **argv)
{
  std::cout << "Node 'euroc_imu' started\n";
  rclcpp::init(argc, argv);

  auto node{std::make_shared<ImuNode>("/imu0", "/imu9")};

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
