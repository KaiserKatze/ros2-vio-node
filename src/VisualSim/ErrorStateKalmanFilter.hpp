#pragma once

#include <cmath>
#include <cstdio>
#include <format>
#include <meta>
#include <print>
#include <stdexcept>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "euroc_vio/Interpolation.hpp"

template <typename value_type = double>
struct ErrorStateKalmanFilter
{
  using Vector3    = Eigen::Vector<value_type, 3>;
  using Matrix3    = Eigen::Vector<value_type, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;

  // 名义状态变量
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

  // 误差状态变量
  // 误差状态是指真实状态与估计状态之差，即 δx = x_true - x_est
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

  /**
   * @brief 每当收到新的 IMU 数据时调用
   * @note 相当于 Measurement Update
   */
  void ImuUpdate(const Vector3 &angular_velocity,
                 const Vector3 &linear_acceleration)
  {
  }

  /**
   * @brief 每当收到新的单目视觉数据时调用
   * @note 相当于 Observation Update
   */
  void MonocularUpdate(const Vector3 &angular_displacement,
                       const Vector3 &normalized_translation)
  {
  }
};
