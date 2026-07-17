module;

#include <limits>
#include <vector>

#include <Eigen/Dense>

export module FastVIO.VisualSim:Camera;

// import std;

namespace FastVIO::VisualSim
{

/**
 * @brief 相机
 */
template <typename value_type>
struct Camera
{
  Eigen::Matrix<value_type, 3, 3> intrinsic_{
      Eigen::Matrix<value_type, 3, 3>::Identity(),
  };
  Eigen::Matrix<value_type, 3, 3> rotation_{
      Eigen::Matrix<value_type, 3, 3>::Identity(),
  };
  Eigen::Vector<value_type, 3> translation_{
      Eigen::Vector<value_type, 3>::Zero(),
  };
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

  Point3 ProjectPoint(const Point3 &object_point,
                      const Eigen::Matrix<value_type, 3, 3> &parent_rotation
                      = Eigen::Matrix<value_type, 3, 3>::Identity(),
                      const Eigen::Vector<value_type, 3> &parent_translation
                      = Eigen::Vector<value_type, 3>::Zero()) const
  {
    const Point3 object_point_parent{
        parent_rotation * object_point + parent_translation,
    };
    const Point3 point_normalized{
        rotation_ * object_point_parent + translation_,
    };
    const Point3 pixel_point{
        intrinsic_ * point_normalized,
    };
    return pixel_point;
  }

  std::pair<std::vector<size_t>, std::vector<Point2>>
  Project(const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &object_matrix,
          const Eigen::Matrix<value_type, 3, 3> &parent_rotation
          = Eigen::Matrix<value_type, 3, 3>::Identity(),
          const Eigen::Vector<value_type, 3> &parent_translation
          = Eigen::Vector<value_type, 3>::Zero()) const
  {
    // 将三维点的非齐次坐标转换为齐次坐标
    auto n_points{object_matrix.cols()};
    Eigen::Matrix<value_type, 4, Eigen::Dynamic> object_matrix_homo(4,
                                                                    n_points);
    object_matrix_homo(Eigen::seq(0, 2), Eigen::all) = object_matrix;
    object_matrix_homo.row(3).setOnes();

    // 组装相机外参矩阵
    Eigen::Matrix<value_type, 3, 4> extrinsic_matrix;
    extrinsic_matrix.template block<3, 3>(0, 0) //
        = rotation_ * parent_rotation;
    extrinsic_matrix.template block<3, 1>(0, 3) //
        = rotation_ * parent_translation + translation_;
    // 投影得到像素坐标系下的齐次坐标
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_matrix_homo{
        // (3x3) * (3x4) * (4,N)
        intrinsic_ * extrinsic_matrix * object_matrix_homo,
    };

    // 检查三维点是否处于相机视域内
    std::vector<size_t> visible_indices;
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
