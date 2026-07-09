#pragma once

#include <cstdint>

#include <Eigen/Core>

template <typename value_type = double>
struct StereoObservation
{
  using Vector2 = Eigen::Vector<value_type, 2>;
  using Vector3 = Eigen::Vector<value_type, 3>;

  // 路标点 ID
  std::uint32_t feature_id_{0};
  // 左目图像中角点坐标
  Vector2 pt_left_{Vector2::Zero()};
  // 右目图像中角点坐标
  Vector2 pt_right_{Vector2::Zero()};
  // 通过三角化得到的路标点在左目相机坐标系下的坐标
  Vector3 landmark_{Vector3::Zero()};
};
