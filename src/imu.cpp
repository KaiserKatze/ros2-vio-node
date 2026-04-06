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
#include <type_traits>
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

/*****************************
 * 常数与类型定义
 *****************************/

static constexpr double g_norm{9.81}; // 重力加速度常数
static constexpr char DEFAULT_FRAME_ID[]{"world"};

using MsgImu         = sensor_msgs::msg::Imu;
using MsgGroundTruth = geometry_msgs::msg::TransformStamped;
using MsgPose        = geometry_msgs::msg::PoseStamped;
using MsgPath        = nav_msgs::msg::Path;

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
struct ImuState : public std::array<double, 10>
{
  ImuState()
      : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
  {
  }

  double &GetPositionX()
  {
    return (*this)[0];
  }

  double &GetPositionY()
  {
    return (*this)[1];
  }

  double &GetPositionZ()
  {
    return (*this)[2];
  }

  double &GetVelocityX()
  {
    return (*this)[3];
  }

  double &GetVelocityY()
  {
    return (*this)[4];
  }

  double &GetVelocityZ()
  {
    return (*this)[5];
  }

  double &GetQuaternionW()
  {
    return (*this)[6];
  }

  double &GetQuaternionX()
  {
    return (*this)[7];
  }

  double &GetQuaternionY()
  {
    return (*this)[8];
  }

  double &GetQuaternionZ()
  {
    return (*this)[9];
  }

  double GetPositionX() const
  {
    return (*this)[0];
  }

  double GetPositionY() const
  {
    return (*this)[1];
  }

  double GetPositionZ() const
  {
    return (*this)[2];
  }

  cv::Vec3d GetPosition() const
  {
    return cv::Vec3d(GetPositionX(), GetPositionY(), GetPositionZ());
  }

  double GetVelocityX() const
  {
    return (*this)[3];
  }

  double GetVelocityY() const
  {
    return (*this)[4];
  }

  double GetVelocityZ() const
  {
    return (*this)[5];
  }

  cv::Vec3d GetVelocity() const
  {
    return cv::Vec3d(GetVelocityX(), GetVelocityY(), GetVelocityZ());
  }

  double GetQuaternionW() const
  {
    return (*this)[6];
  }

  double GetQuaternionX() const
  {
    return (*this)[7];
  }

  double GetQuaternionY() const
  {
    return (*this)[8];
  }

  double GetQuaternionZ() const
  {
    return (*this)[9];
  }

  void SetPosition(double px, double py, double pz)
  {
    (*this)[0] = px;
    (*this)[1] = py;
    (*this)[2] = pz;
  }

  void SetPosition(const cv::Vec3d &pos)
  {
    (*this)[0] = pos[0];
    (*this)[1] = pos[1];
    (*this)[2] = pos[2];
  }

  void SetVelocity(double vx, double vy, double vz)
  {
    (*this)[3] = vx;
    (*this)[4] = vy;
    (*this)[5] = vz;
  }

  void SetVelocity(const cv::Vec3d &vel)
  {
    (*this)[3] = vel[0];
    (*this)[4] = vel[1];
    (*this)[5] = vel[2];
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

  void NormalizeQuaternion()
  {
    double &qw{GetQuaternionW()};
    double &qx{GetQuaternionX()};
    double &qy{GetQuaternionY()};
    double &qz{GetQuaternionZ()};
    const double q_norm{std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz)};
    qw /= q_norm;
    qx /= q_norm;
    qy /= q_norm;
    qz /= q_norm;
  }
};

struct ImuDerivative : public std::array<double, 10>
{
  ImuDerivative()
      : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  {
  }

  void SetVelocity(const cv::Vec3d &velocity)
  {
    (*this)[0] = velocity[0];
    (*this)[1] = velocity[1];
    (*this)[2] = velocity[2];
  }

  void SetAcceleration(const cv::Vec3d &acceleration)
  {
    (*this)[3] = acceleration[0];
    (*this)[4] = acceleration[1];
    (*this)[5] = acceleration[2];
  }

  void SetQuaternionDerivative(const cv::Vec4d &quat_derivative)
  {
    (*this)[6] = quat_derivative[0];
    (*this)[7] = quat_derivative[1];
    (*this)[8] = quat_derivative[2];
    (*this)[9] = quat_derivative[3];
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

  cv::Vec3d Filter(const cv::Vec3d &val)
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

  void Reset()
  {
    initialized_ = false;
    prev_val_    = cv::Vec3d(0, 0, 0);
  }
};

/*****************************
 * IMU 运动学 ODE (常微分方程) 系统
 * 用于给 boost::numeric::odeint 进行 RK4 积分
 *****************************/
struct ImuKinematicsODE
{
  cv::Vec3d a0, a1; // 上一帧和当前帧的加速度 (机体坐标系)
  cv::Vec3d w0, w1; // 上一帧和当前帧的角速度 (机体坐标系)
  double t0, t1;    // 对应的时间戳

  ImuKinematicsODE(const cv::Vec3d &accel0, const cv::Vec3d &accel1,
                   const cv::Vec3d &gyro0, const cv::Vec3d &gyro1, double time0,
                   double time1)
      : a0{accel0}, a1{accel1}, w0{gyro0}, w1{gyro1}, t0{time0}, t1{time1}
  {
  }

  void operator()(const ImuState &x, ImuDerivative &dxdt, const double t) const
  {
    // 采用一阶线性插值作为保持器
    double alpha{0.0};
    if (t1 > t0)
    {
      alpha = std::clamp((t - t0) / (t1 - t0), 0.0, 1.0);
    }
    const cv::Vec3d a{a0 + (a1 - a0) * alpha};
    const cv::Vec3d w{w0 + (w1 - w0) * alpha};

    // 提取当前姿态四元数
    const double qw{x.GetQuaternionW()};
    const double qx{x.GetQuaternionX()};
    const double qy{x.GetQuaternionY()};
    const double qz{x.GetQuaternionZ()};

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
    dxdt.SetVelocity(x.GetVelocity());

    // EuRoC MAV 数据集中前几个 IMU 数据：
    // a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
    // 9.1283567083333317,0.10623870833333333,-2.6069344583333329
    // 9.422556208333333,1.1604535833333331,-2.623278875

    // 可以看出 X 轴的加速度约为 9.8 m/s^2，Y 轴和 Z 轴的加速度较小，说明重力主要作用在 X 轴上
    // X 轴的加速度值是正数，而 IMU 测量的是比力，恰好验证了数据说明书所说的“IMU 的 X 轴朝上”
    // 数据说明书还说 Z 轴是朝着 MAV 正前方（即双目相机的视线方向），Y 轴则是朝向 MAV 的右侧（右手坐标系）

    // 速度导数 = 加速度
    cv::Vec3d accInWorldFrame{R00 * a[0] + R01 * a[1] + R02 * a[2] - g_norm,
                              R10 * a[0] + R11 * a[1] + R12 * a[2],
                              R20 * a[0] + R21 * a[1] + R22 * a[2]};
    dxdt.SetAcceleration(accInWorldFrame);

    // 姿态导数
    const cv::Vec4d attDerivative{
        0.5 * (-qx * w[0] - qy * w[1] - qz * w[2]),
        0.5 * (qw * w[0] + qy * w[2] - qz * w[1]),
        0.5 * (qw * w[1] - qx * w[2] + qz * w[0]),
        0.5 * (qw * w[2] + qx * w[1] - qy * w[0]),
    };
    dxdt.SetQuaternionDerivative(attDerivative);
  }
};

enum class EstimatorType
{
  RK4,
  MAHONY,
  MADGWICK
};

template <typename T>
concept PoseHandler = requires(T t, const MsgPose &msg) {
  { t.HandlePose(msg) } -> std::same_as<void>;
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

  bool is_gravity_aligned_{false};
  double last_time_{0.0};
  MsgImu prev_msg_imu_;

  // 状态向量初始化 (初始原点, 速度为 0, 姿态为单位四元数)
  ImuState state_;
  double ode_time{0.0};

  boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4_;
  std::shared_ptr<AbstractAHRS> ahrs_;

  // 零偏
  cv::Vec3d gyro_bias_{0, 0, 0};
  cv::Vec3d accel_bias_{0, 0, 0};

  MsgPose pose_msg_;

  // 初始重力对齐缓存
  std::vector<MsgImu> init_imu_msg_buffer_;

  cv::Vec3d CalculateAverageGravity() const
  {
    cv::Vec3d g_body{0.0, 0.0, 0.0};
    for (const MsgImu &imu_msg : init_imu_msg_buffer_)
    {
      const double ax{imu_msg.linear_acceleration.x};
      const double ay{imu_msg.linear_acceleration.y};
      const double az{imu_msg.linear_acceleration.z};
      const cv::Vec3d raw_accel{ax, ay, az};
      g_body += raw_accel;
    }
    g_body /= static_cast<double>(init_imu_msg_buffer_.size());
    g_body /= cv::norm(g_body);
    return g_body;
  }

  void EstimateInitialAttitude()
  {
    // 计算平均加速度向量，作为重力方向的估计
    const cv::Vec3d g_body{CalculateAverageGravity()};
    const cv::Vec3d g_world{-1.0, 0.0, 0.0};
    const cv::Vec3d v{g_body.cross(g_world)};
    const double c{g_body.dot(g_world)};
    const double s{cv::norm(v)};

    cv::Mat K{(cv::Mat_<double>(3, 3) << 0, -v[2], v[1], v[2], 0, -v[0], -v[1],
               v[0], 0)};

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
      qx = std::sqrt(std::max(0.0, (1 + R.at<double>(0, 0) - R.at<double>(1, 1)
                                    - R.at<double>(2, 2))
                                       / 4.0));
      qy = std::sqrt(std::max(0.0, (1 - R.at<double>(0, 0) + R.at<double>(1, 1)
                                    - R.at<double>(2, 2))
                                       / 4.0));
      qz = std::sqrt(std::max(0.0, (1 - R.at<double>(0, 0) - R.at<double>(1, 1)
                                    + R.at<double>(2, 2))
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

    state_.SetQuaternion(qw, qx, qy, qz);
  }

  void ComposePoseMessage(MsgPose &msg)
  {
    // 积分后必须对四元数进行归一化，因为 RK4 不保证单位模长约束
    state_.NormalizeQuaternion();
    msg.pose.position.x    = state_.GetPositionX();
    msg.pose.position.y    = state_.GetPositionY();
    msg.pose.position.z    = state_.GetPositionZ();
    msg.pose.orientation.w = state_.GetQuaternionW();
    msg.pose.orientation.x = state_.GetQuaternionX();
    msg.pose.orientation.y = state_.GetQuaternionY();
    msg.pose.orientation.z = state_.GetQuaternionZ();
  }

  void HandleImu(const MsgImu &imu_msg, PoseHandler auto *node_ptr)
  {
    const rclcpp::Time msg_stamp{imu_msg.header.stamp};
    const double current_time{msg_stamp.seconds()};
    const double gx{imu_msg.angular_velocity.x};
    const double gy{imu_msg.angular_velocity.y};
    const double gz{imu_msg.angular_velocity.z};
    const double ax{imu_msg.linear_acceleration.x};
    const double ay{imu_msg.linear_acceleration.y};
    const double az{imu_msg.linear_acceleration.z};
    const cv::Vec3d accel0{
        prev_msg_imu_.linear_acceleration.x,
        prev_msg_imu_.linear_acceleration.y,
        prev_msg_imu_.linear_acceleration.z,
    };
    const cv::Vec3d accel1{ax, ay, az};
    const cv::Vec3d gyro0{
        prev_msg_imu_.angular_velocity.x,
        prev_msg_imu_.angular_velocity.y,
        prev_msg_imu_.angular_velocity.z,
    };
    const cv::Vec3d gyro1{gx, gy, gz};

    // // 读取原始数据
    // const cv::Vec3d raw_accel{ax, ay, az};
    // const cv::Vec3d raw_gyro{gx, gy, gz};

    // // RC 低通滤波
    // const cv::Vec3d filt_accel{use_filter_ ? accel_filter_.Filter(raw_accel)
    //                                        : raw_accel};
    // const cv::Vec3d filt_gyro{use_filter_ ? gyro_filter_.Filter(raw_gyro)
    //                                       : raw_gyro};

    ZeroVelocityUpdate(accel1, gyro1);

    // 利用 RK4 进行积分推算
    Integrate(accel0, accel1, gyro0, gyro1, last_time_, current_time);
    pose_msg_.header.stamp = msg_stamp;
    ComposePoseMessage(pose_msg_);
    node_ptr->HandlePose(pose_msg_);
  }

  void RecoverPoseMessageFromCachedImuMessage(PoseHandler auto *node_ptr)
  {
    // 恢复初始状态
    accel_filter_.Reset();
    gyro_filter_.Reset();
    bool first_msg{true};
    for (const MsgImu &imu_msg : init_imu_msg_buffer_)
    {
      const rclcpp::Time msg_stamp{imu_msg.header.stamp};
      const double current_time{msg_stamp.seconds()};
      if (!first_msg)
      {
        HandleImu(imu_msg, node_ptr);
        first_msg = false;
      }
      prev_msg_imu_ = imu_msg;
      last_time_    = current_time;
    }
  }

  void ZeroVelocityUpdate(const cv::Vec3d &filt_accel,
                          const cv::Vec3d &filt_gyro)
  {
    // 静止检测
    const double acc_norm{cv::norm(filt_accel)};
    const double gyro_norm{cv::norm(filt_gyro)};
    const bool is_static{(std::abs(acc_norm - 9.81) < 0.1)
                         && (gyro_norm < 0.02)};
    // ZUPT + bias 更新
    if (is_static)
    {
      state_.SetVelocity(0, 0, 0);
      static constexpr double alpha{0.01};
      gyro_bias_  = (1 - alpha) * gyro_bias_ + alpha * filt_gyro;
      accel_bias_ = (1 - alpha) * accel_bias_
                    + alpha * (filt_accel - cv::Vec3d(0, 0, -9.81));
    }
  }

public:
  ImuWorker(bool use_filter         = true,
            EstimatorType estimator = EstimatorType::RK4, double init_px = 0.0,
            double init_py = 0.0, double init_pz = 0.0)
      : use_filter_(use_filter), accel_filter_(0.15), gyro_filter_(0.15),
        estimator_type_(estimator)
  {
    // 初始化状态向量
    state_.SetPosition(init_px, init_py, init_pz);
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

    pose_msg_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void RK4Update(const cv::Vec3d &accel0, const cv::Vec3d &accel1,
                 const cv::Vec3d &gyro0, const cv::Vec3d &gyro1, double time0,
                 double time1)
  {
    const double dt{time1 - time0};
    ImuKinematicsODE ode{accel0, accel1, gyro0, gyro1, time0, time1};
    rk4_.do_step(ode, state_, ode_time, dt);
    ode_time += dt;
  }

  void MahonyUpdate(const cv::Vec3d &accel0, const cv::Vec3d &accel1,
                    const cv::Vec3d &gyro0, const cv::Vec3d &gyro1,
                    double time0, double time1)
  {
    // 利用 Mahony 算法或 Madgwick 算法，计算姿态四元数
    cv::Vec3f accel2{static_cast<cv::Vec3f>(accel1)};
    cv::Vec3f gyro2{static_cast<cv::Vec3f>(gyro1)};
    ahrs_->Update(gyro2, accel2, time1 - time0);
    // 更新姿态四元数
    this->state_.SetQuaternion(ahrs_->GetQuaternion());
    // 用龙格库塔法计算速度、位置
    RK4Update(accel0, accel2, gyro0, gyro2, time0, time1);
  }

  void Integrate(const cv::Vec3d &accel0, const cv::Vec3d &accel1,
                 const cv::Vec3d &gyro0, const cv::Vec3d &gyro1, double time0,
                 double time1)
  {
    switch (estimator_type_)
    {
    case EstimatorType::RK4:
      RK4Update(accel0, accel1, gyro0, gyro1, time0, time1);
      break;
    case EstimatorType::MAHONY:
    case EstimatorType::MADGWICK:
      MahonyUpdate(accel0, accel1, gyro0, gyro1, time0, time1);
      break;
    default:
      break;
    }
  }

  void Work(const MsgImu::ConstSharedPtr &imu_msg, PoseHandler auto *node_ptr)
  {
    const rclcpp::Time msg_stamp{imu_msg->header.stamp};
    const double current_time{msg_stamp.seconds()};

    // 初始重力对齐
    if (!is_gravity_aligned_)
    {
      init_imu_msg_buffer_.push_back(*imu_msg);

      // 等积累了足够的初始加速度数据后，计算初始姿态
      if (init_imu_msg_buffer_.size() < 10)
      {
        return;
      }

      // 主要目的：估计无人机的初始姿态
      EstimateInitialAttitude();

      is_gravity_aligned_ = true;

      RecoverPoseMessageFromCachedImuMessage(node_ptr);
    }
    else
    {
      HandleImu(*imu_msg, node_ptr);
    }
    // 缓存上一条 IMU 消息
    prev_msg_imu_ = *imu_msg;
    last_time_    = current_time;
  }
};

class ImuPathPublisher
{
public:
  ImuPathPublisher(rclcpp::Node *node_ptr, const char *input_imu_topic,
                   const char *output_imu_topic, ImuWorker &&worker)
      : node_ptr_{node_ptr}, imu_worker_{std::move(worker)}
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);
    subscriber_ = node_ptr_->create_subscription<MsgImu>(
        input_imu_topic, qos,
        std::bind(&ImuPathPublisher::SubscriberCallback, this, _1));
    publisher_ = node_ptr_->create_publisher<MsgPath>(output_imu_topic, qos);
    // 设置路径消息的坐标系
    this->msg_path_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void HandlePose(const MsgPose &msg)
  {
    this->msg_path_.header.stamp = msg.header.stamp;
    this->msg_path_.poses.push_back(msg);
    this->publisher_->publish(this->msg_path_);
  }

private:
  void PrintAverageSampleRate(const MsgImu::ConstSharedPtr &msg)
  {
    static size_t msg_counter{0};
    static double first_timestamp{0.0};
    const double msg_timestamp{
        static_cast<rclcpp::Time>(msg->header.stamp).seconds()};
    if (msg_counter == 0)
    {
      first_timestamp = msg_timestamp;
    }
    else
    {
      const double elapsed_time{msg_timestamp - first_timestamp};
      const double average_sample_rate{msg_counter / elapsed_time};
      RCLCPP_INFO(node_ptr_->get_logger(), "Average Sample Rate (IMU): %.1f Hz",
                  average_sample_rate);
    }
    ++msg_counter;
  }

  void SubscriberCallback(const MsgImu::ConstSharedPtr &msg)
  {
    PrintAverageSampleRate(msg);
    this->imu_worker_.Work(msg, this);
  }

  rclcpp::Node *node_ptr_;
  rclcpp::Subscription<MsgImu>::SharedPtr subscriber_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_;
  MsgPath msg_path_;
  ImuWorker imu_worker_;
};

class GroundTruthPublisher
{
public:
  GroundTruthPublisher(rclcpp::Node *node_ptr,
                       const char *input_groundtruth_topic,
                       const char *output_groundtruth_topic)
      : node_ptr_{node_ptr}
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);
    subscriber_ = node_ptr_->create_subscription<MsgGroundTruth>(
        input_groundtruth_topic, qos,
        std::bind(&GroundTruthPublisher::SubscriberCallback, this, _1));
    publisher_
        = node_ptr_->create_publisher<MsgPath>(output_groundtruth_topic, qos);
    msg_pose_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
  }

private:
  void SubscriberCallback(const MsgGroundTruth::ConstSharedPtr &msg)
  {
    const auto stamp{msg->header.stamp};
    msg_path_.header.stamp       = stamp;
    msg_pose_.header.stamp       = stamp;
    msg_pose_.pose.position.x    = msg->transform.translation.x;
    msg_pose_.pose.position.y    = msg->transform.translation.y;
    msg_pose_.pose.position.z    = msg->transform.translation.z;
    msg_pose_.pose.orientation.w = msg->transform.rotation.w;
    msg_pose_.pose.orientation.x = msg->transform.rotation.x;
    msg_pose_.pose.orientation.y = msg->transform.rotation.y;
    msg_pose_.pose.orientation.z = msg->transform.rotation.z;
    msg_path_.poses.push_back(msg_pose_);
    publisher_->publish(msg_path_);
  }

  rclcpp::Node *node_ptr_;
  rclcpp::Subscription<MsgGroundTruth>::SharedPtr subscriber_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_;
  MsgPath msg_path_;
  MsgPose msg_pose_;
};

/*****************************
 * ROS2 Node
 *****************************/
class ImuNode : public rclcpp::Node
{
public:
  ImuNode(const char *input_imu_topic, const char *input_groundtruth_topic,
          const char *output_imu_topic, const char *output_groundtruth_topic)
      : Node("IMU")
  {
    this->declare_parameter("estimator", "rk4");
    this->declare_parameter("use_filter", true);
    this->declare_parameter("initial_position_x", 0.0);
    this->declare_parameter("initial_position_y", 0.0);
    this->declare_parameter("initial_position_z", 0.0);
    const std::string estimator_str{
        this->get_parameter("estimator").as_string()};
    const bool use_filter{this->get_parameter("use_filter").as_bool()};
    const double init_px{this->get_parameter("initial_position_x").as_double()};
    const double init_py{this->get_parameter("initial_position_y").as_double()};
    const double init_pz{this->get_parameter("initial_position_z").as_double()};

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
    pub_imu = std::make_unique<ImuPathPublisher>(
        this, input_imu_topic, output_imu_topic,
        ImuWorker{use_filter, estimator, init_px, init_py, init_pz});
    pub_gt = std::make_unique<GroundTruthPublisher>(
        this, input_groundtruth_topic, output_groundtruth_topic);
  }

private:
  std::unique_ptr<GroundTruthPublisher> pub_gt;
  std::unique_ptr<ImuPathPublisher> pub_imu;
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
