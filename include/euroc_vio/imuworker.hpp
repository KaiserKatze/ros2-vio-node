#ifndef IMUWORKER_HPP
#define IMUWORKER_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/ahrs.hpp"
#include "euroc_vio/imustate.hpp"
#include "euroc_vio/main.h"
#include "euroc_vio/zupt.hpp"

using MsgImu         = sensor_msgs::msg::Imu;
using MsgGroundTruth = geometry_msgs::msg::PoseStamped;
using MsgPose        = geometry_msgs::msg::PoseStamped;
using MsgPath        = nav_msgs::msg::Path;
using Vec3           = Eigen::Vector3d;
using Vec4           = Eigen::Vector4d;
static_assert(!std::is_same_v<geometry_msgs::msg::TransformStamped,
                              geometry_msgs::msg::PoseStamped>,
              "MsgGroundTruth cannot be both TransformStamped and PoseStamped");
static_assert(
    std::is_same_v<MsgGroundTruth, geometry_msgs::msg::TransformStamped>
        || std::is_same_v<MsgGroundTruth, geometry_msgs::msg::PoseStamped>,
    "MsgGroundTruth must be either PoseStamped or TransformStamped");

static constexpr double g_norm{9.81}; // 重力加速度常数

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
  EstimatorType estimator_type_;

  bool is_first_frame_{true};
  double last_time_{0.0};
  Vec3 acc_prev_{Vec3::Zero()};
  Vec3 gyro_prev_{Vec3::Zero()};

  // 状态向量初始化 (初始原点, 速度为 0, 姿态为单位四元数)
  ImuState state_;

  boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4_;
  std::shared_ptr<AbstractAHRS<double>> ahrs_;

  // 零偏
  Vec3 gyro_bias_{Vec3::Zero()};
  Vec3 accel_bias_{Vec3::Zero()};

  MsgPose pose_msg_;

#if USE_ZUPT
  ZUPT<> zupt_{};
#endif

#if USE_ZUPT
  void EstimateOrientation()
  {
    try
    {
      const Eigen::Quaterniond q{this->zupt_.EstimateOrientation()};
      this->state_.SetQuaternion(q);
      RCLCPP_INFO(rclcpp::get_logger(NODE_NAME),
                  "Estimated orientation (quaternion): [w: %.3f, x: %.3f, y: "
                  "%.3f, z: %.3f]",
                  q.w(), q.x(), q.y(), q.z());
    }
    catch (const std::exception &e)
    {
      RCLCPP_WARN(rclcpp::get_logger(NODE_NAME),
                  "Failed to estimate orientation: %s", e.what());
    }
  }
#endif

  void ComposePoseMessage(MsgPose &msg)
  {
    msg.pose.position.x = state_.GetPositionX();
    msg.pose.position.y = state_.GetPositionY();
    msg.pose.position.z = state_.GetPositionZ();

    msg.pose.orientation.w = state_.GetQuaternionW();
    msg.pose.orientation.x = state_.GetQuaternionX();
    msg.pose.orientation.y = state_.GetQuaternionY();
    msg.pose.orientation.z = state_.GetQuaternionZ();
  }

#if USE_ZUPT
  void ZeroVelocityUpdate(const Vec3 &accel, const Vec3 &gyro)
  {
    Vector6d data;
    data << gyro, accel;

    const bool is_static{zupt_.Update(data)};
    static bool is_prev_static{true};

    if (!is_static)
    {
      if (is_prev_static)
      {
        RCLCPP_WARN(rclcpp::get_logger(NODE_NAME), "MAV is not static.");
      }
      is_prev_static = is_static;
      return;
    }
    is_prev_static = is_static;

    RCLCPP_WARN(rclcpp::get_logger(NODE_NAME), "MAV is static.");

    this->state_.SetVelocity(0.0, 0.0, 0.0);
    this->EstimateOrientation();

    static constexpr double alpha{0.01};

    gyro_bias_ = (1 - alpha) * gyro_bias_ + alpha * gyro;

    // 加速度计的零偏不能固定减去 [0,0,-9.81]。
    // 我们需要将期望的静止比力(世界系的 [9.81, 0, 0]) 投影到当前的载具参考系下，
    // 再与测量值作差来计算偏差。

    // 提取当前姿态四元数
    Eigen::Quaterniond q{state_.GetQuaternion()};

    // 四元数转旋转矩阵 R_wv (载具系到世界系的变换)
    Eigen::Matrix3d R{q.toRotationMatrix()};

    // 矩阵 R_wv 的转置，即世界系到载具参考系的旋转 R_vw 第一列
    Vec3 expected_g{R.col(0)};
    expected_g *= g_norm;
    accel_bias_ = (1 - alpha) * accel_bias_ + alpha * (accel - expected_g);
  }
#endif

public:
  ImuWorker(EstimatorType estimator = EstimatorType::RK4) :
    estimator_type_(estimator)
  {
    double init_px{0.0};
    double init_py{0.0};
    double init_pz{0.0};
    // 初始化状态向量
    state_.SetPosition(init_px, init_py, init_pz);
    // 其他状态保持默认

    if (estimator == EstimatorType::RK4)
    {
      ahrs_ = nullptr; // RK4 不需要 AHRS
    }
    else if (estimator == EstimatorType::MAHONY)
    {
      ahrs_ = std::make_shared<MahonyAHRS<double>>();
    }
    else if (estimator == EstimatorType::MADGWICK)
    {
      ahrs_ = std::make_shared<MadgwickAHRS<double>>();
    }
    else
    {
      throw std::invalid_argument{"Unsupported estimator type"};
    }

    pose_msg_.header.frame_id = DEFAULT_FRAME_ID;
  }

  // 已知量
  // 1. 传感器参考系下测得的加速度和角速度
  // 上一帧和当前帧的【已扣除零偏】加速度 (载具参考系)
  // 上一帧和当前帧的【已扣除零偏】角速度 (载具参考系)
  // 上一帧和当前帧对应的时间戳
  // 2. 在惯性参考系下的重力加速度向量
  //      EuRoC MAV 数据集中前几个 IMU 数据：
  //        a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
  //        9.1283567083333317,0.10623870833333333,-2.6069344583333329
  //        9.422556208333333,1.1604535833333331,-2.623278875
  //      可以看出 X 轴的加速度约为 9.8 m/s^2，Y 轴和 Z 轴的加速度较小，说明重力主要作用在 X 轴上
  //      X 轴的加速度值是正数，而 IMU 测量的是比力，恰好验证了数据说明书所说的“IMU 的 X 轴朝上”
  //      数据说明书还说 Z 轴是朝着 MAV 正前方（即双目相机的视线方向），Y 轴则是朝向 MAV 的右侧（右手坐标系）
  //      由于重力向量指向 X 轴负方向，所以真正的重力加速度大约是 [-9.81, 0, 0]
  // 3. 从载具参考系到传感器参考系的旋转矩阵 = 单位矩阵
  // Eigen::Quaterniond C_sv;
  // 4. 传感器参考系的原点在载具参考系下的平移向量 = 零向量
  // Vec3 r_sv_v;
  void RK4Update(const Vec3 &accel0, const Vec3 &accel1, const Vec3 &gyro0,
                 const Vec3 &gyro1, double time0, double time1)
  {
    const double dt{time1 - time0};
    const Eigen::Vector3d gravity_world{0.0, 0.0, -g_norm};

    // 更新速度和位置

    // 提取当前姿态四元数 (载具参考系到惯性参考系的旋转 C_iv = C_is)
    // 无人机的朝向 = 惯性参考系到载具参考系的旋转 C_vi 的逆 = 载具参考系到惯性参考系的旋转 C_iv
    Eigen::Quaterniond q{this->state_.GetQuaternion()};
    // 传感器参考系下的平均线加速度
    Eigen::Vector3d acc_sensor = (accel0 + accel1) * 0.5;
    acc_sensor
        = Eigen::Vector3d{acc_sensor.z(), -acc_sensor.y(), acc_sensor.x()};
    // 惯性参考系下的平均线加速度
    //    EuRoC MAV 数据集中 IMU 传感器参考系与载具参考系重合
    //    即 C_sv = 1, C_si = C_vi，r_sv_v = 0
    //    因此加速度转换公式简化为：
    //       acc_sensor = C_si * (acc_world - gravity_world)
    //       C_si_inverse * acc_sensor = acc_world - gravity_world
    //       acc_world = C_si_inverse * acc_sensor + gravity_world
    //       acc_world = C_is * acc_sensor + gravity_world
    Eigen::Vector3d acc_world = q * acc_sensor + gravity_world;
    // 将 IMU 参考系的 X 轴映射为 Z 轴
    // 将 IMU 参考系的 Y 轴映射为 -Y 轴
    // 将 IMU 参考系的 Z 轴映射为 X 轴
    // \delta s = \overline{v} \cdot \delta t
    // \delta v = \overline{a} \cdot \delta t
    // v(t + \delta t) = v(t) + \overline{a} \cdot \delta t
    // s(t + \delta t) = s(t) + \overline{v} \cdot \delta t
    // \overline{v} = 0.5 \cdot (v(t) + v(t + \delta t))
    // s(t + \delta t) = s(t) + v(t) \cdot \delta t + \frac12 \cdot \overline{a} \cdot (\delta t)^2
    Eigen::Vector3d velocity{this->state_.GetVelocity()};
    Eigen::Vector3d position{this->state_.GetPosition()};
    Eigen::Vector3d delta_velocity{acc_world * dt};

    RCLCPP_INFO(rclcpp::get_logger("ImuWorker"),
                "\n\tPrevious Attitude: [w: %.3f, x: %.3f, y: %.3f, z: %.3f]; "
                "\n\tPrevious Velocity: [x: %.3f, y: %.3f, y: %.3f]; "
                "\n\tPrevious Position: [x: %.3f, y: %.3f, y: %.3f]",
                q.w(), q.x(), q.y(), q.z(), velocity.x(), velocity.y(),
                velocity.z(), position.x(), position.y(), position.z());

    position += velocity * dt + 0.5 * delta_velocity * dt;
    this->state_.SetPosition(position);
    velocity += delta_velocity;
    this->state_.SetVelocity(velocity);

    // 更新姿态

    // 载具参考系下的平均角速度 = 传感器参考系下的平均角速度
    Eigen::Vector3d gyro_avg = (gyro0 + gyro1) * 0.5;
    // 角增量
    Eigen::Vector3d phi   = gyro_avg * dt;
    const double phi_norm = phi.norm();
    // 四元数增量
    Eigen::Quaterniond delta_q;
    if (phi_norm < 1e-6)
    {
      // 陀螺仪积分 \dot{q} = 0.5 * q * w
      // q(t + \delta t) = q(t) + \dot{q}(t) * \delta t
      //                 = q(t) + 0.5 * q(t) * w * \delta t
      //                 = q(t) * 1 + q(t) * (0.5 * w * \delta t)
      //                 = q(t) * (1.0, 0.5 * w * \delta t)
      delta_q = Eigen::Quaterniond(1.0, 0.5 * phi.x(), 0.5 * phi.y(),
                                   0.5 * phi.z());
    }
    else
    {
      delta_q
          = Eigen::Quaterniond(Eigen::AngleAxisd(phi_norm, phi.normalized()));
    }
    q = delta_q * q;

    this->state_.SetQuaternion(q);
    // 积分后必须对四元数进行归一化，因为 RK4 不保证单位模长约束
    this->state_.NormalizeQuaternion();

    RCLCPP_INFO(rclcpp::get_logger("ImuWorker"),
                "\n\tAverage Accel: [x: %.3f, y: %.3f, z: %.3f]; "
                "\n\tRotated Accel: [x: %.3f, y: %.3f, z: %.3f]; "
                "\n\tAverage Gyro: [x: %.3f, y: %.3f, z: %.3f]; "
                "\n\tδ Velocity: [x: %.3f, y: %.3f, z: %.3f]",
                acc_sensor.x(), acc_sensor.y(), acc_sensor.z(), acc_world.x(),
                acc_world.y(), acc_world.z(), gyro_avg.x(), gyro_avg.y(),
                gyro_avg.z(), delta_velocity.x(), delta_velocity.y(),
                delta_velocity.z());

    RCLCPP_INFO(rclcpp::get_logger("ImuWorker"),
                "\n\tCurrent Attitude: [w: %.3f, x: %.3f, y: %.3f, z: %.3f]; "
                "\n\tCurrent Velocity: [x: %.3f, y: %.3f, y: %.3f]; "
                "\n\tCurrent Position: [x: %.3f, y: %.3f, y: %.3f]",
                q.w(), q.x(), q.y(), q.z(), velocity.x(), velocity.y(),
                velocity.z(), position.x(), position.y(), position.z());
  }

  /**
   * @brief 利用 Mahony 算法或 Madgwick 算法，计算姿态四元数
   */
  void MahonyUpdate(const Vec3 &accel0, const Vec3 &accel1, const Vec3 &gyro0,
                    const Vec3 &gyro1, double time0, double time1)
  {
    Vec3 accel{accel1};
    Vec3 gyro{gyro1};

    // 步进更新 AHRS 算法以获得精确的有重力修正的姿态
    this->ahrs_->Update(gyro, accel, time1 - time0);

    // 切忌在 RK4 积分之前覆盖状态变量 state_

    // 如果先覆盖，RK4 会再次利用 gyro 数据对姿态求导并积分，导致同一帧的角速度被积分两次（或者与 Mahony 抗衡）。
    // 正确的做法是：利用现有的姿态去推算速度和位置（RK4），推算完成后，抛弃纯运动学算出的漂移姿态，用 Mahony 的姿态覆盖它。

    // 正常运行 RK4 计算速度和位置 (RK4 内部会自己处理自己的四元数状态)
    this->RK4Update(accel0, accel1, gyro0, gyro1, time0, time1);

    // 将 RK4 推算出的纯积分姿态替换为由 Mahony 滤波器校准过的姿态
    this->state_.SetQuaternion(this->ahrs_->GetQuaternion());
  }

  void Integrate(const Vec3 &accel0, const Vec3 &accel1, const Vec3 &gyro0,
                 const Vec3 &gyro1, double time0, double time1)
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
    const double current_time{msg_stamp.nanoseconds() * 1e-9};

    Vec3 accel{imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
               imu_msg->linear_acceleration.z};

    Vec3 gyro{imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
              imu_msg->angular_velocity.z};

#if USE_ZUPT
    ZeroVelocityUpdate(accel, gyro);
#endif

    // 如果是第一帧，只记录初始状态，不进行积分计算
    if (is_first_frame_)
    {
      acc_prev_       = accel;
      gyro_prev_      = gyro;
      last_time_      = current_time;
      is_first_frame_ = false;
      return;
    }

    Integrate(acc_prev_ - accel_bias_, accel - accel_bias_,
              gyro_prev_ - gyro_bias_, gyro - gyro_bias_, last_time_,
              current_time);

    pose_msg_.header.stamp = msg_stamp;
    ComposePoseMessage(pose_msg_);
    node_ptr->HandlePose(pose_msg_);

    // 缓存上一条 IMU 消息
    acc_prev_  = accel;
    gyro_prev_ = gyro;
    last_time_ = current_time;
  }
};

#endif /* IMUWORKER_HPP */
