#pragma once

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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

  cv::Ptr<cv::FastFeatureDetector> fastFeatureDetector{
      cv::FastFeatureDetector::create(fastThreshold, fastNonmaxSuppression,
                                      fastType)};

public:
  const cv::Size subpix_win_size{5, 5};
  const cv::Size subpix_zero_zone{-1, -1};
  const cv::TermCriteria subpix_criteria{
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
      40,
      0.01,
  };

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
      // 1. 计算需要补充的角点数量
      int num_needed{static_cast<int>(maxCorners)
                     - static_cast<int>(corners_prev_left.size())};

      if (num_needed > 0)
      {
        // 2. 提取新的角点
        std::vector<cv::KeyPoint> keypoints_prev_left_ext;

        if (gray_prev_left.empty() || gray_prev_right.empty())
        {
          fastFeatureDetector->detect(gray_prev_left, keypoints_prev_left_ext,
                                      cv::noArray());
        }
        else
        {
          // 创建空间掩膜 (Mask)，避免重复提取已经在追踪的角点
          // 初始为全白 (255)，在已有角点处画黑色实心圆 (0)
          cv::Mat mask{gray_prev_left.size(), CV_8U, cv::Scalar{255}};
          for (const auto &pt : corners_prev_left)
          {
            // 这里的半径相当于非极大值抑制的物理范围，可根据图像分辨率微调
            cv::circle(mask, pt, 5, cv::Scalar{0}, -1);
          }
          fastFeatureDetector->detect(gray_prev_left, keypoints_prev_left_ext,
                                      mask);
        }

        if (corners_prev_left.size() + keypoints_prev_left_ext.size()
            < minCorners)
        {
          return false;
        }

        if (!keypoints_prev_left_ext.empty())
        {
          // 按照响应值（response）降序排列，保留最显著的前 maxCorners 个角点，提取最优角点
          std::ranges::sort(keypoints_prev_left_ext, std::greater<>{},
                            &cv::KeyPoint::response);

          // // 限制送入光流的新点数量 (考虑光流筛选会有折损，预留 2 倍冗余)
          // size_t extract_limit = static_cast<size_t>(num_needed * 2);
          // if (keypoints_prev_left_ext.size() > extract_limit)
          // {
          //   keypoints_prev_left_ext.resize(extract_limit);
          // }

          // 将 KeyPoint 映射回 PointType
          std::vector<PointType> corners_prev_left_ext
              = keypoints_prev_left_ext
                | std::views::transform([](const cv::KeyPoint &kp)
                                        { return PointType(kp.pt); })
                | std::ranges::to<std::vector>();
          cv::cornerSubPix(gray_prev_left, corners_prev_left_ext,
                           subpix_win_size, subpix_zero_zone, subpix_criteria);

          std::vector<PointType> corners_prev_right_ext;
          std::vector<unsigned char> features_found_pl_pr;

          // https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
          cv::calcOpticalFlowPyrLK(gray_prev_left, gray_prev_right,
                                   corners_prev_left_ext,
                                   corners_prev_right_ext, features_found_pl_pr,
                                   cv::noArray(), winSize, maxLevel, criteria_);
          cv::cornerSubPix(gray_prev_right, corners_prev_right_ext,
                           subpix_win_size, subpix_zero_zone, subpix_criteria);

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
                             && (std::abs(p_prev_left.y - p_prev_right.y)
                                 < atol);
                    });

          // 因为 view 是延迟计算的，所以必须先创建副本
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

          // 合并到主追踪序列
          corners_prev_left.reserve(corners_prev_left.size()
                                    + new_corners_prev_left_ext.size());
          corners_prev_right.reserve(corners_prev_right.size()
                                     + new_corners_prev_right_ext.size());
          corners_prev_left.append_range(std::move(new_corners_prev_left_ext));
          corners_prev_right.append_range(
              std::move(new_corners_prev_right_ext));
        }
      }

      // 如果补充完之后，连最低要求都没达到，说明图像严重退化
      if (corners_prev_left.size() < minCorners)
      {
        return false;
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
      cv::cornerSubPix(gray_next_right, corners_next_right, subpix_win_size,
                       subpix_zero_zone, subpix_criteria);

      auto zipped_view
          = std::views::zip(corners_prev_left, corners_prev_right,
                            corners_next_right, features_found_pr_nr)
            | std::views::filter(
                [](const auto &tuple)
                {
                  // 1. 必须是追踪成功的点
                  return std::get<3>(tuple);
                });

      // 因为 view 是延迟计算的，所以必须先创建副本
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

      if (corners_prev_left.size() < minCorners)
      {
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
      cv::cornerSubPix(gray_next_left, corners_next_left, subpix_win_size,
                       subpix_zero_zone, subpix_criteria);

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

      if (corners_prev_left.size() < minCorners)
      {
        return false;
      }
    }

    //===================================
    // 从下一帧左目到上一帧左目 (闭环)
    {
      std::vector<PointType> corners_prev_left_loopback;
      std::vector<unsigned char> features_found_nl_pl;
      cv::calcOpticalFlowPyrLK(gray_next_left, gray_prev_left,
                               corners_next_left, corners_prev_left_loopback,
                               features_found_nl_pl, cv::noArray(), winSize,
                               maxLevel, criteria_);
      cv::cornerSubPix(gray_prev_left, corners_prev_left_loopback,
                       subpix_win_size, subpix_zero_zone, subpix_criteria);

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

      if (corners_prev_left.size() < minCorners)
      {
        return false;
      }
    }

    // std::cerr << "\t通过所有筛选! 最终存活角点 = " << corners_prev_left.size()
    //           << "\n";
    return true;
  }
};

} // namespace CornerDetection
