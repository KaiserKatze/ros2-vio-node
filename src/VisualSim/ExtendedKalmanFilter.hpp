#pragma once

#include <cmath>
#include <cstdio>
#include <format>
#include <meta>
#include <print>
#include <stdexcept>

#include <Eigen/Dense>

#include "euroc_vio/Interpolation.hpp"

template <typename value_type>
struct ExtendedKalmanFilter
{
  using Vector3    = Eigen::Vector<value_type, 3>;
  using Matrix3    = Eigen::Vector<value_type, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;

  // 状态变量
  struct StateVariable
  {
    Vector3 position_{Vector3::Zero()};
    Vector3 linear_velocity_{Vector3::Zero()};
    Vector3 linear_acceleration_{Vector3::Zero()};
    Quaternion attitude_{Quaternion::Identity()};
    Quaternion attitude_derivative_{Quaternion::Zero()};
  };

  struct ObservableVariableIMU
  {
    Vector3 angular_velocity_;
    Vector3 linear_acceleration_;
  };

  struct ObservableVariableCamera
  {
    Vector3 angular_velocity_;
    Vector3 delta_position_;
  };
};
