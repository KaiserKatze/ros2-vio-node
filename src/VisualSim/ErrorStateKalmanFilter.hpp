#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <meta>
#include <print>
#include <stdexcept>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "euroc_vio/Interpolation.hpp"

/**
 * @brief 基于松耦合的误差状态卡尔曼滤波，为短航程无人机提供姿态解算功能
 */
template <typename value_type = double>
class ErrorStateKalmanFilter
{
#pragma region 公有类型

public:
  using Vector3 = Eigen::Vector<value_type, 3>;
  using Matrix3 = Eigen::Matrix<value_type, 3, 3>;

  // 朝向的四元数形式 (仅用于存储，不用于运算)
  using Quaternion = Eigen::Quaternion<value_type>;
  // 朝向的李群形式
  using Attitude = Sophus::SO3<value_type>;

  // 名义状态变量 (记作 $x$)
  struct NominalStateVariable
  {
    // 位置 $r^{iv}_i$
    Vector3 position_{Vector3::Zero()};
    // 线速度 $\dot{r}^{iv}_i$
    Vector3 linear_velocity_{Vector3::Zero()};
    // 朝向 $C_{iv}$
    Quaternion attitude_{Quaternion::Identity()};
    // 加速度计零偏
    Vector3 accelerometer_bias_{Vector3::Zero()};
    // 陀螺仪零偏
    Vector3 gyroscope_bias_{Vector3::Zero()};
  };

  // 误差状态变量 (指真实状态与估计状态之差，即 $\delta x = x_true - x_est$)
  struct ErrorStateVariable
  {
    // 位置误差
    Vector3 position_error_{Vector3::Zero()};
    // 线速度误差
    Vector3 linear_velocity_error_{Vector3::Zero()};
    // 旋转误差的轴角表示
    Vector3 rotation_error_{Vector3::Zero()};
    // 加速度计零偏误差
    Vector3 accelerometer_bias_error_{Vector3::Zero()};
    // 陀螺仪零偏误差
    Vector3 gyroscope_bias_error_{Vector3::Zero()};
  };

  // 目前不考虑将重力加速度加入状态空间!

  struct DatumImuImpl
  {
    // 时间戳 (单位: 纳秒)
    std::int64_t timestamp_;
    // 角速度向量 (单位: rad s^-1)
    Vector3 angular_velocity_;
    // 线加速度向量 (单位: m s^-2)
    Vector3 linear_acceleration_;
  };

  struct DatumFastImpl
  {
    // 时间戳 (单位: 纳秒)
    std::int64_t timestamp_;
    // 角位移向量 (单位: rad)
    Vector3 angular_displacement_;
    // 单位化平移向量 (无单位)
    Vector3 normalized_translation_;
  };

#pragma endregion

#pragma region 私有类型

private:
  // 具体取值由 DatumFastImpl 中的成员变量的种类及数量决定
  static constexpr int dimMonocularData{2 * 3};

  // 具体取值由 ErrorStateVariable 中的成员变量的种类及数量决定
  static constexpr int dimErrorState{5 * 3};

  // 实际参与运算的名义状态向量或误差状态向量 (每当需要从 ESKF 中读取数据时，用 Eigen::Map 进行转换)
  using VariableImpl = Eigen::Vector<value_type, dimErrorState>;
  // 误差状态的协方差矩阵
  using CovarianceErrorState
      = Eigen::Matrix<value_type, dimErrorState, dimErrorState>;
  // 观测雅可比矩阵
  using JacobiMeasurement
      = Eigen::Matrix<value_type, dimMonocularData, dimErrorState>;

#pragma endregion

public:
  /**
   * @brief 设置初始朝向
   * @note 调用者必须在启动 ESKF 以前，首先对静止状态下的无人机校准重力方向，估计初始朝向。
   * @note 特别是 EuRoC MAV 数据集，在它提供的数据中，
   *       静止状态下的无人机的 IMU 传感器参考系的坐标轴与世界参考系的坐标轴不是平行的。
   *       IMU 参考系的 X 轴正方向与世界参考系的 Z 轴正方向近似同向；
   *       IMU 参考系的 Y 轴正方向与世界参考系的 Y 轴负方向近似同向；
   *       IMU 参考系的 Z 轴正方向与世界参考系的 X 轴正方向近似同向。
   *       从 ESKF 的角度来看，无法保证重力加速度方向是沿 IMU 参考系的 Z 轴负方向的。
   */
  void AttitudeUpdate(const Attitude &attitude)
  {
    // TODO
  }

  /**
   * @brief 每当收到新的 IMU 数据时调用
   * @param imu_data IMU 数据提供的角速度向量和线加速度向量
   * @note 调用者必须保证 IMU 数据是在“体坐标系”下的表示
   */
  void ImuUpdate(const DatumImuImpl *imu_data)
  {
    // TODO
  }

  /**
   * @brief 每当收到新的单目视觉数据时调用
   * @param monocular_data 单目视觉数据提供的角位移向量和单位化平移向量
   * @note 调用者必须保证单目视觉数据是在“体坐标系”下的表示
   */
  void MonocularUpdate(const DatumFastImpl *monocular_data)
  {
    // TODO
  }

  /**
   * @brief 获取当前名义状态
   */
  NominalStateVariable GetNominalState() const
  {
    // TODO
  }

#pragma region 私有成员函数

private:
  /**
   * @brief 将误差状态注入名义状态
   */
  void InjectError()
  {
    // TODO
    // 1. 更新位置: r = r + dr
    // 2. 更新姿态: q = q * exp(d_theta)
    // 3. 更新零偏: b = b + db
    // 4. 重置 error_state_ 为零
  }

#pragma endregion

#pragma region 私有成员变量

private:
  // 重力加速度大小
  value_type gravity_world_norm{9.81};
  // 世界坐标系下的重力加速度
  Vector3 gravity_world{-gravity_world_norm * Vector3::UnitZ()};
  // 对获取的第一帧 IMU 数据作特殊处理
  bool is_initialized_{false};
  // 名义状态 (计算时使用 VariableImpl 实例，读取时使用 NominalStateVariable 实例)
  VariableImpl nominal_state_{};
  // 误差状态
  VariableImpl error_state_{};
  // 加速度计噪声
  Vector3 accelerometer_noise_{Vector3::Zero()};
  // 陀螺仪噪声
  Vector3 gyroscope_noise_{Vector3::Zero()};
  // 加速度计零偏噪声
  Vector3 accelerometer_bias_noise_{Vector3::Zero()};
  // 陀螺仪零偏噪声
  Vector3 gyroscope_bias_noise_{Vector3::Zero()};
  // 误差状态协方差矩阵
  CovarianceErrorState error_state_covariance_{
      CovarianceErrorState::Identity()
  };
  // 上一帧 IMU 时间戳
  std::int64_t last_imu_time_;

#pragma endregion
};
