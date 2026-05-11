#pragma once

#include <iostream>
#include <tuple>
#include <vector>

#include "Camera.hpp"

/**
 * @brief 双目相机
 */
template <typename value_type> struct StereoRig
{
  Camera<value_type> camera_left_;
  Camera<value_type> camera_right_;

  using Point2 = Eigen::Vector<value_type, 2>;

  std::tuple<std::vector<size_t>, std::vector<Point2>, std::vector<Point2>>
  Project(const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &object_matrix,
          const Eigen::Matrix<value_type, 3, 3> &parent_rotation
          = Eigen::Matrix<value_type, 3, 3>::Identity(),
          const Eigen::Vector<value_type, 3> &parent_translation
          = Eigen::Vector<value_type, 3>::Zero()) const
  {
    auto &&[indices_left, pixels_left] = camera_left_.Project(
        object_matrix, parent_rotation, parent_translation);
    auto &&[indices_right, pixels_right] = camera_right_.Project(
        object_matrix, parent_rotation, parent_translation);

    {
      std::cerr << "\t当前场景中，"
                   "左目可见路标点有 "
                << indices_left.size()
                << " 个, "
                   "右目可见路标点有 "
                << indices_right.size() << " 个.\n";
    }

    // 左目、右目视图中可见三维点可能不一样，需要取交集
    std::vector<size_t> common_indices;
    std::vector<Point2> common_image_left;
    std::vector<Point2> common_image_right;

    // 因为 Camera 推入的索引天然是升序的，所以采用 O(N) 的双指针法取交集即可
    size_t i{0}, j{0};
    while (i < indices_left.size() && j < indices_right.size())
    {
      if (indices_left[i] == indices_right[j])
      {
        common_indices.push_back(indices_left[i]);
        common_image_left.push_back(pixels_left[i]);
        common_image_right.push_back(pixels_right[j]);
        ++i;
        ++j;
      }
      else if (indices_left[i] < indices_right[j])
      {
        ++i;
      }
      else
      {
        ++j;
      }
    }

    return {common_indices, common_image_left, common_image_right};
  }
};
