#pragma once

#include <cstdint>

#include <Eigen/Core>

template <typename value_type = double>
struct StereoObservation
{
  using Vector2 = Eigen::Vector<value_type, 2>;

  // 路标点 ID
  std::uint32_t feature_id_{0};
  // 左目图像中角点坐标
  Vector2 pt_left_{Vector2::Zero()};
  // 右目图像中角点坐标
  Vector2 pt_right_{Vector2::Zero()};
  // 左目图像中角点响应值 (FAST score)
  value_type response_left_{0.0};
  // 右目图像中角点响应值 (FAST score)
  value_type response_right_{0.0};
};
