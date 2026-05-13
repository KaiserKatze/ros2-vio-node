#pragma once

#include <iostream>
#include <vector>

#include "Camera.hpp"

template <typename value_type> struct Frame
{
  using Point2 = Eigen::Vector<value_type, 2>;

  std::vector<size_t> indices_left_;
  std::vector<Point2> pixels_left_;
  std::vector<size_t> indices_right_;
  std::vector<Point2> pixels_right_;
};

template <typename value_type> struct AlignedFrame
{
  using Point2 = Eigen::Vector<value_type, 2>;

  std::vector<size_t> indices_common_;
  std::vector<Point2> pixels_left_;
  std::vector<Point2> pixels_right_;

  AlignedFrame()                                = default;
  ~AlignedFrame()                               = default;
  AlignedFrame(const AlignedFrame &)            = delete;
  AlignedFrame &operator=(const AlignedFrame &) = delete;
  AlignedFrame(AlignedFrame &&other)            = default;
  AlignedFrame &operator=(AlignedFrame &&)      = default;
  AlignedFrame(const Frame<value_type> &);
};

/**
 * @brief 双目相机
 */
template <typename value_type> struct StereoRig
{
  Camera<value_type> camera_left_;
  Camera<value_type> camera_right_;

  using Point2 = Eigen::Vector<value_type, 2>;

  Frame<value_type>
  Project(const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &object_matrix,
          const Eigen::Matrix<value_type, 3, 3> &parent_rotation
          = Eigen::Matrix<value_type, 3, 3>::Identity(),
          const Eigen::Vector<value_type, 3> &parent_translation
          = Eigen::Vector<value_type, 3>::Zero()) const
  {
    Frame<value_type> frame;

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

    frame.indices_left_  = std::move(indices_left);
    frame.indices_right_ = std::move(indices_right);
    frame.pixels_left_   = std::move(pixels_left);
    frame.pixels_right_  = std::move(pixels_right);

    return frame;
  }

  static AlignedFrame<value_type> AlignViews(const Frame<value_type> &frame)
  {
    const auto &indices_left{frame.indices_left_};
    const auto &indices_right{frame.indices_right_};
    const auto &pixels_left{frame.pixels_left_};
    const auto &pixels_right{frame.pixels_right_};

    const size_t n_points{std::max(indices_left.size(), indices_right.size())};

    // 左目、右目视图中可见三维点可能不一样，需要取交集
    AlignedFrame<value_type> aligned_frame;
    std::vector<size_t> &common_indices{aligned_frame.indices_common_};
    std::vector<Point2> &common_image_left{aligned_frame.pixels_left_};
    std::vector<Point2> &common_image_right{aligned_frame.pixels_right_};

    common_indices.reserve(n_points);
    common_image_left.reserve(n_points);
    common_image_right.reserve(n_points);

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
    return aligned_frame;
  }

  static void AlignFrames(AlignedFrame<value_type> &frame1,
                          AlignedFrame<value_type> &frame2)
  {
    const std::vector<size_t> &indices1{frame1.indices_common_};
    const std::vector<size_t> &indices2{frame2.indices_common_};
    std::vector<size_t> common_indices;
    std::vector<Point2> common_frame1_image_left;
    std::vector<Point2> common_frame1_image_right;
    std::vector<Point2> common_frame2_image_left;
    std::vector<Point2> common_frame2_image_right;

    size_t i{0}, j{0};
    while (i < indices1.size() && j < indices2.size())
    {
      if (indices1[i] == indices2[j])
      {
        common_indices.push_back(indices1[i]);
        common_frame1_image_left.push_back(frame1.pixels_left_[i]);
        common_frame1_image_right.push_back(frame1.pixels_right_[i]);
        common_frame2_image_left.push_back(frame2.pixels_left_[j]);
        common_frame2_image_right.push_back(frame2.pixels_right_[j]);
        ++i;
        ++j;
      }
      else if (indices1[i] < indices2[j])
      {
        ++i;
      }
      else
      {
        ++j;
      }
    }

    frame1.indices_common_ = common_indices;
    frame2.indices_common_ = common_indices;
    frame1.pixels_left_    = std::move(common_frame1_image_left);
    frame1.pixels_right_   = std::move(common_frame1_image_right);
    frame2.pixels_left_    = std::move(common_frame2_image_left);
    frame2.pixels_right_   = std::move(common_frame2_image_right);
  }
};

template <typename value_type>
AlignedFrame<value_type>::AlignedFrame(const Frame<value_type> &frame) :
  AlignedFrame<value_type>(StereoRig<value_type>::AlignViews(frame))
{
}
