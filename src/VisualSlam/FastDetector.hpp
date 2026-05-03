#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CornerDetection.hpp"

namespace CornerDetection
{

template <typename PointType = cv::Point2f>
struct FastDetector : public AbstractDetector
{
private:
  const cv::TermCriteria criteria_{
      (cv::TermCriteria::COUNT) | (cv::TermCriteria::EPS),
      // Maximum number of iterations
      20,
      // Minimum change per iteration
      0.3,
  };

  static constexpr int fastThreshold{20};
  static constexpr bool fastNonmaxSuppression{true};
  static constexpr cv::FastFeatureDetector::DetectorType fastType{
      cv::FastFeatureDetector::TYPE_9_16};

  // 已包含 FAST 探测器的创建
  cv::Ptr<cv::FastFeatureDetector> fastFeatureDetector{
      cv::FastFeatureDetector::create(fastThreshold, fastNonmaxSuppression,
                                      fastType)};

public:
  bool FindCorners(const cv::Mat &gray_prev_left,
                   const cv::Mat &gray_prev_right,
                   const cv::Mat &gray_next_left,
                   const cv::Mat &gray_next_right,
                   std::vector<PointType> &corners_prev_left,
                   std::vector<PointType> &corners_prev_right,
                   std::vector<PointType> &corners_next_left,
                   std::vector<PointType> &corners_next_right) const
  {
    //===================================
    // 从上一帧左目到上一帧右目
    {
      // 使用 FAST 检测器获取 cv::KeyPoint
      std::vector<cv::KeyPoint> keypoints_prev_left_ext;
      fastFeatureDetector->detect(gray_prev_left, keypoints_prev_left_ext);

      // 按照响应值（response）降序排列，保留最显著的前 maxCorners 个角点，提取最优角点
      std::ranges::sort(keypoints_prev_left_ext, std::greater<>{},
                        &cv::KeyPoint::response);
      if (keypoints_prev_left_ext.size() > maxCorners)
      {
        keypoints_prev_left_ext.resize(maxCorners);
      }

      // 将 KeyPoint 映射回 PointType
      std::vector<PointType> corners_prev_left_ext
          = keypoints_prev_left_ext
            | std::views::transform([](const cv::KeyPoint &kp)
                                    { return PointType(kp.pt); })
            | std::ranges::to<std::vector>();

      std::vector<PointType> corners_prev_right_ext;

      if (corners_prev_left.size() + corners_prev_left_ext.size() < minCorners)
      {
        std::cerr << "\t未通过第 0 轮筛选!\n";
        return false;
      }

      std::vector<unsigned char> features_found_pl_pr;
      // https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
      cv::calcOpticalFlowPyrLK(gray_prev_left, gray_prev_right,
                               corners_prev_left_ext, corners_prev_right_ext,
                               features_found_pl_pr, cv::noArray(), winSize,
                               maxLevel, criteria_);

      // 压缩数据
      std::cerr << "\t筛选前，角点个数 = " << corners_prev_left_ext.size()
                << "\n";

      auto zipped_view
          = std::views::zip(corners_prev_left_ext, corners_prev_right_ext,
                            features_found_pl_pr)
            | std::views::filter(
                [atol = atol_parallax](const auto &tuple)
                {
                  const auto &[p_prev_left, p_prev_right, found] = tuple;
                  // 1. 必须是追踪成功的点
                  // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                  // 3. 极线过滤：纵坐标之差必须小于阈值
                  return found && (p_prev_left.x > p_prev_right.x)
                         && (std::abs(p_prev_left.y - p_prev_right.y) < atol);
                });

      // 因为 view 是延迟计算的，所以必须先创建副本，绝对不能使用 std::vector::assign 方法进行赋值
      std::vector<PointType> new_corners_prev_left_ext
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<0>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_prev_right_ext
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<1>(tuple); })
            | std::ranges::to<std::vector>();

      corners_prev_left_ext  = std::move(new_corners_prev_left_ext);
      corners_prev_right_ext = std::move(new_corners_prev_right_ext);

      std::cerr << "\t筛选后，角点个数 = " << corners_prev_left_ext.size()
                << "\n";

      if (corners_prev_left.size() + corners_prev_left_ext.size() < minCorners)
      {
        std::cerr << "\t未通过第 1 轮筛选!\n";
        return false;
      }

      if (corners_prev_left.empty() || corners_prev_right.empty())
      {
        // 缺少先验信息，需要初始化角点数据
        corners_prev_left  = std::move(corners_prev_left_ext);
        corners_prev_right = std::move(corners_prev_right_ext);
      }
      else
      {
        // 已有先验信息
        // 【修改】：使用 C++23/C++26 引入的 std::ranges::append_range，更现代且不易出错
        corners_prev_left.reserve(corners_prev_left.size()
                                  + corners_prev_left_ext.size());
        corners_prev_right.reserve(corners_prev_right.size()
                                   + corners_prev_right_ext.size());
        corners_prev_left.append_range(std::move(corners_prev_left_ext));
        corners_prev_right.append_range(std::move(corners_prev_right_ext));
      }
    }

    //===================================
    // 从上一帧右目到下一帧右目
    {
      std::vector<unsigned char> features_found_pr_nr;
      cv::calcOpticalFlowPyrLK(gray_prev_right, gray_next_right,
                               corners_prev_right, corners_next_right,
                               features_found_pr_nr, cv::noArray(), winSize,
                               maxLevel, criteria_);

      // 压缩数据
      std::cerr << "\t筛选前，角点个数 = " << corners_prev_left.size() << "\n";

      auto zipped_view
          = std::views::zip(corners_prev_left, corners_prev_right,
                            corners_next_right, features_found_pr_nr)
            | std::views::filter(
                [](const auto &tuple)
                {
                  // 1. 必须是追踪成功的点
                  return std::get<3>(tuple);
                });

      // 因为 view 是延迟计算的，所以必须先创建副本，绝对不能使用 std::vector::assign 方法进行赋值
      std::vector<PointType> new_corners_prev_left
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<0>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_prev_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<1>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_next_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<2>(tuple); })
            | std::ranges::to<std::vector>();

      corners_prev_left  = std::move(new_corners_prev_left);
      corners_prev_right = std::move(new_corners_prev_right);
      corners_next_right = std::move(new_corners_next_right);

      std::cerr << "\t筛选后，角点个数 = " << corners_prev_left.size() << "\n";

      if (corners_prev_left.size() < minCorners)
      {
        std::cerr << "\t未通过第 2 轮筛选!\n";
        return false;
      }
    }

    //===================================
    // 从下一帧右目到下一帧左目
    {
      std::vector<unsigned char> features_found_nr_nl;
      cv::calcOpticalFlowPyrLK(gray_next_right, gray_next_left,
                               corners_next_right, corners_next_left,
                               features_found_nr_nl, cv::noArray(), winSize,
                               maxLevel, criteria_);

      // 压缩数据
      std::cerr << "\t筛选前，角点个数 = " << corners_prev_left.size() << "\n";

      auto zipped_view
          = std::views::zip(corners_prev_left, corners_prev_right,
                            corners_next_left, corners_next_right,
                            features_found_nr_nl)
            | std::views::filter(
                [atol = atol_parallax](const auto &tuple)
                {
                  const auto &[p_prev_left, p_prev_right, p_next_left,
                               p_next_right, found] = tuple;
                  // 1. 必须是追踪成功的点
                  // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                  // 3. 极线过滤：纵坐标之差必须小于阈值
                  return found && (p_next_left.x > p_next_right.x)
                         && (std::abs(p_next_left.y - p_next_right.y) < atol);
                });

      std::vector<PointType> new_corners_prev_left
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<0>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_prev_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<1>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_next_left
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<2>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_next_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<3>(tuple); })
            | std::ranges::to<std::vector>();

      corners_prev_left  = std::move(new_corners_prev_left);
      corners_prev_right = std::move(new_corners_prev_right);
      corners_next_left  = std::move(new_corners_next_left);
      corners_next_right = std::move(new_corners_next_right);

      std::cerr << "\t筛选后，角点个数 = " << corners_prev_left.size() << "\n";

      if (corners_prev_left.size() < minCorners)
      {
        std::cerr << "\t未通过第 3 轮筛选!\n";
        return false;
      }
    }

    //===================================
    // 从下一帧左目到上一帧左目
    {
      std::vector<PointType> corners_prev_left_loopback;
      std::vector<unsigned char> features_found_nl_pl;
      cv::calcOpticalFlowPyrLK(gray_next_left, gray_prev_left,
                               corners_next_left, corners_prev_left_loopback,
                               features_found_nl_pl, cv::noArray(), winSize,
                               maxLevel, criteria_);
      // 压缩数据
      std::cerr << "\t筛选前，角点个数 = " << corners_prev_left.size() << "\n";

      auto zipped_view
          = std::views::zip(corners_prev_left, corners_prev_right,
                            corners_next_left, corners_next_right,
                            corners_prev_left_loopback, features_found_nl_pl)
            | std::views::filter(
                [atol = atol_coincidence](const auto &tuple)
                {
                  const auto &[p_prev_left, p_prev_right, p_next_left,
                               p_next_right, p_loop_back, found] = tuple;
                  // 1. 必须是追踪成功的点
                  // 2. 重合过滤：横、纵坐标之差必须小于阈值
                  return found
                         && (std::abs(p_prev_left.x - p_loop_back.x) < atol)
                         && (std::abs(p_prev_left.y - p_loop_back.y) < atol);
                });

      std::vector<PointType> new_corners_prev_left
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<0>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_prev_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<1>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_next_left
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<2>(tuple); })
            | std::ranges::to<std::vector>();
      std::vector<PointType> new_corners_next_right
          = zipped_view
            | std::views::transform([](const auto &tuple)
                                    { return std::get<3>(tuple); })
            | std::ranges::to<std::vector>();

      corners_prev_left  = std::move(new_corners_prev_left);
      corners_prev_right = std::move(new_corners_prev_right);
      corners_next_left  = std::move(new_corners_next_left);
      corners_next_right = std::move(new_corners_next_right);

      std::cerr << "\t筛选后，角点个数 = " << corners_prev_left.size() << "\n";

      if (corners_prev_left.size() < minCorners)
      {
        std::cerr << "\t未通过第 4 轮筛选!\n";
        return false;
      }
    }

    std::cerr << "\t通过所有筛选!\n";
    return true;
  }
};

} // namespace CornerDetection
