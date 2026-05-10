#pragma once

#include <iostream>
#include <map>
#include <ranges>
#include <tuple>

#include "Camera.hpp"

/**
 * @brief 双目相机
 */
template <typename value_type> struct StereoRig
{
  Camera<value_type> camera_left_;
  Camera<value_type> camera_right_;

  using Point2 = Eigen::Vector<value_type, 2>;
  using Point3 = Eigen::Vector<value_type, 3>;
  using Point4 = Eigen::Vector<value_type, 4>;

  std::tuple<std::vector<Point3>, std::vector<Point2>, std::vector<Point2>>
  Project(const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &object_matrix,
          const Eigen::Matrix<value_type, 3, 3> &parent_rotation
          = Eigen::Matrix<value_type, 3, 3>::Identity(),
          const Eigen::Vector<value_type, 3> &parent_translation
          = Eigen::Vector<value_type, 3>::Zero()) const
  {
    auto &&[object_points_left, pixel_points_left] = camera_left_.Project(
        object_matrix, parent_rotation, parent_translation);
    auto &&[object_points_right, pixel_points_right] = camera_right_.Project(
        object_matrix, parent_rotation, parent_translation);

    {
      std::cerr << "\t当前场景中，左目可见路标点有 "
                << object_points_left.size() << " 个, 右目可见路标点有 "
                << object_points_right.size() << " 个.\n";
      // std::cerr << "\t左目可见路标点 = [\n";
      // for (const Point3 &object_point : object_points_left)
      // {
      //   std::cerr << "\t\t[" << object_point.x() << ", " << object_point.y()
      //             << ", " << object_point.z() << "];\n";
      // }
      // std::cerr << "\t]\n";
      // std::cerr << "\t右目可见路标点 = [\n";
      // for (const Point3 &object_point : object_points_right)
      // {
      //   std::cerr << "\t\t[" << object_point.x() << ", " << object_point.y()
      //             << ", " << object_point.z() << "];\n";
      // }
      // std::cerr << "\t]\n";
    }

    // 左目、右目视图中可见三维点可能不一样，需要取交集
    std::vector<Point3> common_object_points;
    std::vector<Point2> common_image_left;
    std::vector<Point2> common_image_right;

    // 使用 Map 存储右目可见点，Key 是三维点的坐标（利用 Eigen 的比较逻辑）
    // 如果点数极多，可以考虑基于索引匹配，但这里基于坐标比对最通用
    const auto comp = [](const Point3 &p1, const Point3 &p2)
    { return std::tie(p1(0), p1(1), p1(2)) < std::tie(p2(0), p2(1), p2(2)); };

    const auto lookup_map
        = std::views::zip(object_points_left, pixel_points_left)
          | std::ranges::to<std::map<Point3, Point2, decltype(comp)>>(comp);

    // 遍历左目点，如果在右目也存在，则存入结果

    for (const auto &[object_point_right, pixel_point_right] :
         std::views::zip(object_points_right, pixel_points_right))
    {
      const auto it{lookup_map.find(object_point_right)};
      if (it != lookup_map.end())
      {
        common_object_points.push_back(object_point_right);
        common_image_left.push_back(it->second);
        common_image_right.push_back(pixel_point_right);
      }
    }

    return {common_object_points, common_image_left, common_image_right};
  }
};
