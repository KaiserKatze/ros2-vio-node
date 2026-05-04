#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "CornerDetection.hpp"

namespace CornerDetection
{

template <typename PointType = cv::Point2f>
struct OrbFlannDetector : public AbstractDetector
{
private:
  // ORB 特征提取器
  // 初始容量设为 5000，可根据实际分辨率调整
  cv::Ptr<cv::ORB> orb_detector_{cv::ORB::create(5000)};

  // FLANN 匹配器，配置 LSH 索引以适配 ORB 的二进制描述子 (CV_8U)
  cv::Ptr<cv::DescriptorMatcher> flann_matcher_{
      cv::makePtr<cv::FlannBasedMatcher>(
          cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))};

  // ORB 匹配时的最大汉明距离阈值
  static constexpr float max_hamming_distance_{50.0f};

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
    // 准备各帧的关键点与描述子容器
    std::vector<cv::KeyPoint> keypoints_prev_left, keypoints_prev_right,
        keypoints_next_right, keypoints_next_left;
    cv::Mat descriptors_prev_left, descriptors_prev_right,
        descriptors_next_right, descriptors_next_left;

    //===================================
    // 第 0 步：初始化与补充上一帧左目特征点
    //===================================
    {
      // 1. 将现有的 PointType 转换回 cv::KeyPoint 以便计算描述子
      for (const auto &pt : corners_prev_left)
      {
        keypoints_prev_left.emplace_back(pt, 20.f);
      }

      // 2. 计算需要补充的角点数量
      int num_needed{static_cast<int>(maxCorners)
                     - static_cast<int>(corners_prev_left.size())};

      if (num_needed > 0)
      {
        std::vector<cv::KeyPoint> new_kp;

        if (gray_prev_left.empty() || gray_prev_right.empty())
        {
          orb_detector_->detect(gray_prev_left, new_kp, cv::noArray());
        }
        else
        {
          // 创建空间掩膜 (Mask)，避免重复提取已经在追踪的角点
          cv::Mat mask{gray_prev_left.size(), CV_8U, cv::Scalar{255}};
          for (const auto &pt : corners_prev_left)
          {
            cv::circle(mask, pt, 5, cv::Scalar{0}, -1);
          }
          orb_detector_->detect(gray_prev_left, new_kp, mask);
        }

        if (!new_kp.empty())
        {
          // 按响应值降序排序，保留最显著的角点
          std::ranges::sort(new_kp, std::greater<>{}, &cv::KeyPoint::response);

          if (new_kp.size() > static_cast<size_t>(num_needed))
          {
            new_kp.resize(num_needed);
          }
          // C++23/26: 现代合并容器方式
          keypoints_prev_left.append_range(std::move(new_kp));
        }
      }

      if (keypoints_prev_left.size() < minCorners)
      {
        return false;
      }

      // 计算左目关键点的描述子
      orb_detector_->compute(gray_prev_left, keypoints_prev_left,
                             descriptors_prev_left);
    }

    // 备份初始的左目关键点与描述子，用于最终的闭环校核 (Loopback)
    const std::vector<cv::KeyPoint> orig_kp_pl = keypoints_prev_left;
    const cv::Mat orig_desc_pl                 = descriptors_prev_left.clone();

    if (descriptors_prev_left.empty())
    {
      return false;
    }

    //===================================
    // 第 1 轮筛选：从上一帧左目到上一帧右目
    //===================================
    {
      orb_detector_->detectAndCompute(gray_prev_right, cv::noArray(),
                                      keypoints_prev_right,
                                      descriptors_prev_right);
      if (descriptors_prev_right.empty())
      {
        return false;
      }

      std::vector<cv::DMatch> matches_01;
      flann_matcher_->match(descriptors_prev_left, descriptors_prev_right,
                            matches_01);

      // 筛选逻辑：汉明距离 + 视差过滤 + 极线约束
      auto valid_matches
          = matches_01
            | std::views::filter(
                [&](const cv::DMatch &m)
                {
                  const auto &p_l = keypoints_prev_left[m.queryIdx].pt;
                  const auto &p_r = keypoints_prev_right[m.trainIdx].pt;
                  return (m.distance < max_hamming_distance_) && (p_l.x > p_r.x)
                         && (std::abs(p_l.y - p_r.y) < atol_parallax);
                })
            | std::ranges::to<std::vector>();

      if (valid_matches.size() < minCorners)
      {
        return false;
      }

      // 更新追踪链：提取配对成功的关键点和描述子作为下一次匹配的 Query
      std::vector<cv::KeyPoint> next_kp_pl, next_kp_pr;
      cv::Mat next_desc_pl, next_desc_pr;
      for (const auto &m : valid_matches)
      {
        next_kp_pl.push_back(keypoints_prev_left[m.queryIdx]);
        next_desc_pl.push_back(descriptors_prev_left.row(m.queryIdx));

        next_kp_pr.push_back(keypoints_prev_right[m.trainIdx]);
        next_desc_pr.push_back(descriptors_prev_right.row(m.trainIdx));
      }

      keypoints_prev_left    = std::move(next_kp_pl);
      descriptors_prev_left  = std::move(next_desc_pl);
      keypoints_prev_right   = std::move(next_kp_pr);
      descriptors_prev_right = std::move(next_desc_pr);
    }

    //===================================
    // 第 2 轮筛选：从上一帧右目到下一帧右目
    //===================================
    {
      orb_detector_->detectAndCompute(gray_next_right, cv::noArray(),
                                      keypoints_next_right,
                                      descriptors_next_right);
      if (descriptors_next_right.empty())
      {
        return false;
      }

      std::vector<cv::DMatch> matches_12;
      flann_matcher_->match(descriptors_prev_right, descriptors_next_right,
                            matches_12);

      // 筛选逻辑：仅汉明距离 (时间追踪通常不需要极线和视差约束)
      auto valid_matches
          = matches_12
            | std::views::filter([](const cv::DMatch &m)
                                 { return m.distance < max_hamming_distance_; })
            | std::ranges::to<std::vector>();

      if (valid_matches.size() < minCorners)
      {
        return false;
      }

      std::vector<cv::KeyPoint> next_kp_pl, next_kp_pr, next_kp_nr;
      cv::Mat next_desc_pl, next_desc_pr, next_desc_nr;
      for (const auto &m : valid_matches)
      {
        next_kp_pl.push_back(keypoints_prev_left[m.queryIdx]);
        next_desc_pl.push_back(descriptors_prev_left.row(m.queryIdx));

        next_kp_pr.push_back(keypoints_prev_right[m.queryIdx]);
        next_desc_pr.push_back(descriptors_prev_right.row(m.queryIdx));

        next_kp_nr.push_back(keypoints_next_right[m.trainIdx]);
        next_desc_nr.push_back(descriptors_next_right.row(m.trainIdx));
      }

      keypoints_prev_left    = std::move(next_kp_pl);
      descriptors_prev_left  = std::move(next_desc_pl);
      keypoints_prev_right   = std::move(next_kp_pr);
      descriptors_prev_right = std::move(next_desc_pr);
      keypoints_next_right   = std::move(next_kp_nr);
      descriptors_next_right = std::move(next_desc_nr);
    }

    //===================================
    // 第 3 轮筛选：从下一帧右目到下一帧左目
    //===================================
    {
      orb_detector_->detectAndCompute(gray_next_left, cv::noArray(),
                                      keypoints_next_left,
                                      descriptors_next_left);
      if (descriptors_next_left.empty())
      {
        return false;
      }

      std::vector<cv::DMatch> matches_23;
      flann_matcher_->match(descriptors_next_right, descriptors_next_left,
                            matches_23);

      // 筛选逻辑：汉明距离 + 视差过滤 (右转左，注意坐标方向) + 极线约束
      auto valid_matches
          = matches_23
            | std::views::filter(
                [&](const cv::DMatch &m)
                {
                  const auto &p_nr = keypoints_next_right[m.queryIdx].pt;
                  const auto &p_nl = keypoints_next_left[m.trainIdx].pt;
                  // 视差过滤：左目横坐标必须大于右目横坐标
                  return (m.distance < max_hamming_distance_)
                         && (p_nl.x > p_nr.x)
                         && (std::abs(p_nl.y - p_nr.y) < atol_parallax);
                })
            | std::ranges::to<std::vector>();

      if (valid_matches.size() < minCorners)
      {
        return false;
      }

      std::vector<cv::KeyPoint> next_kp_pl, next_kp_pr, next_kp_nr, next_kp_nl;
      cv::Mat next_desc_nl;
      for (const auto &m : valid_matches)
      {
        next_kp_pl.push_back(keypoints_prev_left[m.queryIdx]);
        next_kp_pr.push_back(keypoints_prev_right[m.queryIdx]);
        next_kp_nr.push_back(keypoints_next_right[m.queryIdx]);

        next_kp_nl.push_back(keypoints_next_left[m.trainIdx]);
        next_desc_nl.push_back(descriptors_next_left.row(m.trainIdx));
      }

      keypoints_prev_left  = std::move(next_kp_pl);
      keypoints_prev_right = std::move(next_kp_pr);
      keypoints_next_right = std::move(next_kp_nr);
      keypoints_next_left  = std::move(next_kp_nl);
      descriptors_next_left
          = std::move(next_desc_nl); // 仅保留 desc_nl 用于最终闭环
    }

    //===================================
    // 第 4 轮筛选：从下一帧左目到上一帧左目 (闭环)
    //===================================
    {
      std::vector<cv::DMatch> matches_30;
      // 将最新的左目特征与第一步备份的初始左目特征进行匹配
      flann_matcher_->match(descriptors_next_left, orig_desc_pl, matches_30);

      // 筛选逻辑：汉明距离 + 重合度过滤
      auto valid_matches
          = matches_30
            | std::views::filter(
                [&](const cv::DMatch &m)
                {
                  // p_tracked 是通过前三轮匹配链条传承下来的上一帧左目坐标
                  const auto &p_tracked = keypoints_prev_left[m.queryIdx].pt;
                  // p_loop_back 是本次在原始特征全集中真实匹配到的上一帧左目坐标
                  const auto &p_loop_back = orig_kp_pl[m.trainIdx].pt;

                  return (m.distance < max_hamming_distance_)
                         && (std::abs(p_tracked.x - p_loop_back.x)
                             < atol_coincidence)
                         && (std::abs(p_tracked.y - p_loop_back.y)
                             < atol_coincidence);
                })
            | std::ranges::to<std::vector>();

      if (valid_matches.size() < minCorners)
      {
        return false;
      }

      // 最终输出：将经过所有轮次筛选和闭环验证的角点存入引用参数中
      std::vector<PointType> final_pl, final_pr, final_nr, final_nl;
      for (const auto &m : valid_matches)
      {
        final_pl.push_back(PointType(keypoints_prev_left[m.queryIdx].pt));
        final_pr.push_back(PointType(keypoints_prev_right[m.queryIdx].pt));
        final_nr.push_back(PointType(keypoints_next_right[m.queryIdx].pt));
        final_nl.push_back(PointType(keypoints_next_left[m.queryIdx].pt));
      }

      corners_prev_left  = std::move(final_pl);
      corners_prev_right = std::move(final_pr);
      corners_next_right = std::move(final_nr);
      corners_next_left  = std::move(final_nl);
    }

    std::cerr << "\t通过所有筛选! 最终存活角点 = " << corners_prev_left.size()
              << "\n";
    return true;
  }
};

} // namespace CornerDetection
