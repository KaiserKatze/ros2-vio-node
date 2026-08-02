#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include <sophus/se3.hpp>

namespace FastVIO::VisualSim
{

/**
 * @brief 相机
 */
template <typename value_type>
struct Camera
{
  using Pose      = Sophus::SE3<value_type>;
  using Intrinsic = Eigen::Matrix<value_type, 3, 3>;

  // 相机内参
  Intrinsic intrinsic_{Intrinsic::Identity()};
  // 相机外参: 载具坐标系 → 传感器坐标系的变换 (T_SB), p_S = T_SB · p_B
  Pose sensor_from_body_{};
  int width_{752};
  int height_{480};

  Camera()
  {
    intrinsic_(0, 0) = static_cast<value_type>(450);
    intrinsic_(1, 1) = static_cast<value_type>(450);
    intrinsic_(0, 2) = static_cast<value_type>(width_ * 0.5);
    intrinsic_(1, 2) = static_cast<value_type>(height_ * 0.5);
  }

  using Point2 = Eigen::Vector<value_type, 2>;
  using Point3 = Eigen::Vector<value_type, 3>;

  /**
   * @param object_point 单个三维路标点在世界坐标系下的坐标
   * @param body_pose 载具在世界坐标系下的位姿 (T_WB)
   */
  Point3 ProjectPoint(const Point3 &object_point,
                      const Pose &body_pose = Pose{}) const
  {
    return intrinsic_
           * (sensor_from_body_ * body_pose.inverse() * object_point);
  }

  /**
   * @param object_matrix 任意个三维路标点在世界坐标系下的坐标 (组成的 3xN 矩阵)
   * @param body_pose 载具在世界坐标系下的位姿 (T_WB)
   */
  std::pair<std::vector<std::size_t>, std::vector<Point2>>
  Project(const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &object_matrix,
          const Pose &body_pose = Pose{}) const
  {
    // 将三维点的非齐次坐标转换为齐次坐标
    auto n_points{object_matrix.cols()};
    Eigen::Matrix<value_type, 4, Eigen::Dynamic> object_matrix_homo(4,
                                                                    n_points);
    object_matrix_homo(Eigen::seq(0, 2), Eigen::all) = object_matrix;
    object_matrix_homo.row(3).setOnes();

    // 组装相机外参矩阵 (世界系 → 传感器系)
    const Pose sensor_from_world{sensor_from_body_ * body_pose.inverse()};
    const auto extrinsic_matrix{sensor_from_world.matrix3x4()};
    // 投影得到像素坐标系下的齐次坐标
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_matrix_homo{
        // (3x3) * (3x4) * (4xN)
        intrinsic_ * extrinsic_matrix * object_matrix_homo,
    };

    // 检查三维点是否处于相机视域内
    std::vector<std::size_t> visible_indices;
    std::vector<Point2> visible_pixel_points;
    visible_indices.reserve(n_points);
    visible_pixel_points.reserve(n_points);

    for (decltype(n_points) i = 0; i < n_points; ++i)
    {
      const Point3 pixel_point_homo{pixel_matrix_homo.col(i)};
      const value_type w{pixel_point_homo(2)};
      // 深度测试: 点必须在相机的前方
      if (w <= std::numeric_limits<value_type>::epsilon())
      {
        continue;
      }

      const value_type u{pixel_point_homo(0) / w};
      const value_type v{pixel_point_homo(1) / w};
      // 边界测试: 投影点在成像范围内
      if (0.0 < u && u < static_cast<value_type>(width_) && 0.0 < v
          && v < static_cast<value_type>(height_))
      {
        visible_indices.push_back(i); // 直接推入索引！
        visible_pixel_points.emplace_back(u, v);
      }
    }
    return {visible_indices, visible_pixel_points};
  }
};

} // namespace FastVIO::VisualSim
