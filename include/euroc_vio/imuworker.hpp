#ifndef IMUWORKER_HPP
#define IMUWORKER_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
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
#include "euroc_vio/imustate.hpp"
#include "euroc_vio/main.h"
#include "euroc_vio/rcfilter.hpp"

using MsgImu         = sensor_msgs::msg::Imu;
using MsgGroundTruth = geometry_msgs::msg::TransformStamped;
using MsgPose        = geometry_msgs::msg::PoseStamped;
using MsgPath        = nav_msgs::msg::Path;

static constexpr double g_norm{9.81}; // 重力加速度常数

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
  static constexpr size_t init_imu_msg_buffer_capacity_{10};
  std::vector<MsgImu> init_imu_msg_buffer_;

  cv::Vec3d CalculateAverageGravity() const
  {
    if (init_imu_msg_buffer_.empty())
    {
      throw std::runtime_error{"init_imu_msg_buffer_ is empty"};
    }
    cv::Vec3d g_body{std::transform_reduce(
        init_imu_msg_buffer_.cbegin(), init_imu_msg_buffer_.cend(),
        cv::Vec3d::zeros(), std::plus<>(),
        [](const MsgImu &imu_msg) -> cv::Vec3d
        {
          return cv::Vec3d{imu_msg.linear_acceleration.x,
                           imu_msg.linear_acceleration.y,
                           imu_msg.linear_acceleration.z};
        })};
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
    init_imu_msg_buffer_.reserve(init_imu_msg_buffer_capacity_);
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
      if (init_imu_msg_buffer_.size() < init_imu_msg_buffer_capacity_)
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

#endif /* IMUWORKER_HPP */
