#pragma once

#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/types.hpp>

template <typename PointType = cv::Point2f> struct EKF
{
  /**
   * @brief 利用视觉里程计前端提取的特征点，在运动控制未知的前提下，实施扩展卡尔曼滤波
   */
  void Update(
      const std::vector<PointType> &corners_prev_left,
      const std::vector<PointType> &corners_prev_right,
      const std::vector<PointType> &corners_next_left,
      const std::vector<PointType> &corners_next_right,
      std::vector<Eigen::Vector<typename PointType::value_type, 4>> &landmarks)
  {
    // TODO

    (void) corners_prev_left;
    (void) corners_prev_right;
    (void) corners_next_left;
    (void) corners_next_right;
    (void) landmarks;
  }
};
