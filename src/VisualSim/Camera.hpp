#pragma once

// #include <iomanip>
// #include <iostream>
#include <limits>

#include <Eigen/Dense>

/**
 * @brief 相机
 */
template <typename value_type> struct Camera
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
  int width{752};
  int height{480};

  using Point2 = Eigen::Vector<value_type, 2>;
  using Point3 = Eigen::Vector<value_type, 3>;
  using Point4 = Eigen::Vector<value_type, 4>;

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
    // std::cerr << "\t[DEBUG]\n"
    //           << std::fixed << std::setprecision(1) << "\t\t点 ["
    //           << object_point.x() << ", " << object_point.y() << ", "
    //           << object_point.z() << "]\n"
    //           << "\t\t在体坐标系下的坐标 = [" << object_point_parent.x() << ", "
    //           << object_point_parent.y() << ", " << object_point_parent.z()
    //           << "]\n"
    //           << "\t\t在归一化图像坐标系下的坐标 = [" << point_normalized.x()
    //           << ", " << point_normalized.y() << ", " << point_normalized.z()
    //           << "]\n"
    //           << "\t\t在图像坐标系下的坐标 = [" << pixel_point.x() << ", "
    //           << pixel_point.y() << ", " << pixel_point.z() << "]\n"
    //           << "\n";
    return pixel_point;
  }

  std::pair<std::vector<Point3>, std::vector<Point2>>
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

    // 投影到相机坐标系，得到齐次坐标
    Eigen::Matrix<value_type, 3, 4> extrinsic_matrix;
    extrinsic_matrix.template block<3, 3>(0, 0) //
        = rotation_ * parent_rotation;
    extrinsic_matrix.template block<3, 1>(0, 3) //
        = rotation_ * parent_translation + translation_;
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_matrix_homo{
        // (3x3) * (3x4) * (4,N)
        intrinsic_ * extrinsic_matrix * object_matrix_homo,
    };

    // 检查三维点是否处于相机视域内
    std::vector<Point3> visible_object_points;
    std::vector<Point2> visible_pixel_points;
    visible_object_points.reserve(n_points);
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
      if (0.0 < u && u < static_cast<value_type>(width) && 0.0 < v
          && v < static_cast<value_type>(height))
      {
        visible_object_points.push_back(object_matrix.col(i));
        visible_pixel_points.emplace_back(u, v);
      }
    }

    return {visible_object_points, visible_pixel_points};
  }
};
