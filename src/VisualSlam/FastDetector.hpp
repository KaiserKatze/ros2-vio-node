#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "CornerDetection.hpp"

namespace FastVIO::CornerDetection
{

// https://docs.opencv.org/4.13.0/df/d74/classcv_1_1FastFeatureDetector.html
struct FastDetector : public AbstractDetector
{
  using PointType = cv::Point2f;

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
      cv::FastFeatureDetector::TYPE_9_16
  };

  // https://docs.opencv.org/4.13.0/dc/d84/group__core__basic.html#ga524e5e94ebf48db273a71ab275eaf5b5
  // https://docs.opencv.org/4.13.0/df/d74/classcv_1_1FastFeatureDetector.html#a3bbc39b65bdda963b129ed3841d2de07
  cv::Ptr<cv::FastFeatureDetector> fastFeatureDetector{
      cv::FastFeatureDetector::create(fastThreshold, fastNonmaxSuppression,
                                      fastType)
  };

public:
  const cv::Size subpix_win_size{5, 5};
  const cv::Size subpix_zero_zone{-1, -1};
  const cv::TermCriteria subpix_criteria{
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
      40,
      0.01,
  };

  using Points = std::vector<PointType>;

  bool FindCorners(const cv::Mat &gray_prev_left,
                   const cv::Mat &gray_prev_right,
                   const cv::Mat &gray_next_left,
                   const cv::Mat &gray_next_right,
                   std::vector<PointType> &corners_prev_left,
                   std::vector<PointType> &corners_prev_right,
                   std::vector<PointType> &corners_next_left,
                   std::vector<PointType> &corners_next_right,
                   std::vector<std::uint32_t> &feature_ids, bool use_hint) const
  {
    // 断言：确保 feature_ids 严格递增（随索引增大而增大）
    assert(std::is_sorted(feature_ids.begin(), feature_ids.end(),
                          std::less<std::uint32_t>()));
    assert(gray_prev_left.dims == gray_prev_right.dims
           && gray_prev_left.rows == gray_prev_right.rows
           && gray_prev_left.cols == gray_prev_right.cols
           && gray_prev_left.dims == gray_next_left.dims
           && gray_prev_left.rows == gray_next_left.rows
           && gray_prev_left.cols == gray_next_left.cols
           && gray_prev_left.dims == gray_next_right.dims
           && gray_prev_left.rows == gray_next_right.rows
           && gray_prev_left.cols == gray_next_right.cols);

    if (!TrackStereoPrevLeftToPrevRight(gray_prev_left, gray_prev_right,
                                        corners_prev_left, corners_prev_right,
                                        feature_ids))
    {
      return false;
    }

    if (!TrackTemporalRight(gray_prev_right, gray_next_right, corners_prev_left,
                            corners_prev_right, corners_next_left,
                            corners_next_right, feature_ids, use_hint))
    {
      return false;
    }

    if (!TrackStereoNextRightToNextLeft(gray_next_right, gray_next_left,
                                        corners_prev_left, corners_prev_right,
                                        corners_next_left, corners_next_right,
                                        feature_ids, use_hint))
    {
      return false;
    }

    if (!ValidateLoopback(gray_next_left, gray_prev_left, corners_prev_left,
                          corners_prev_right, corners_next_left,
                          corners_next_right, feature_ids))
    {
      return false;
    }

    return true;
  }

private:
  SubPixAdaptor CreateSubPixAdaptor(const cv::Mat &image) const noexcept
  {
    return {image, subpix_win_size, subpix_zero_zone, subpix_criteria};
  }

  void CalcOpticalFlowPyrLK(const cv::Mat &prevImg, const cv::Mat &nextImg,
                            const std::vector<PointType> &prevPts,
                            std::vector<PointType> &nextPts,
                            std::vector<unsigned char> &status, int flags) const
  {
    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status,
                             cv::noArray(), winSize, maxLevel, criteria_,
                             flags);
  }

  //===================================
  // 从上一帧左目到上一帧右目
  bool
  TrackStereoPrevLeftToPrevRight(const cv::Mat &gray_prev_left,
                                 const cv::Mat &gray_prev_right,
                                 std::vector<PointType> &corners_prev_left,
                                 std::vector<PointType> &corners_prev_right,
                                 std::vector<std::uint32_t> &feature_ids) const
  {
    assert(corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == feature_ids.size());

    // 1. 计算需要补充的角点数量
    const int num_needed{static_cast<int>(maxCorners)
                         - static_cast<int>(corners_prev_left.size())};

    // 不仅数量要够，还必须确保左右目特征点数组不为空且大小一致
    if (num_needed <= 0 && !corners_prev_right.empty()
        && corners_prev_left.size() == corners_prev_right.size())
    {
      return HaveEnoughCorners(corners_prev_left);
    }

    // 2. 提取新的角点
    // https://docs.opencv.org/4.13.0/d2/d29/classcv_1_1KeyPoint.html
    std::vector<cv::KeyPoint> keypoints_prev_left_ext;
    if (corners_prev_left.empty() || corners_prev_right.empty())
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

    // 新提取的角点需要赋予 feature_id
    ExtendFeatureIdList(feature_ids, keypoints_prev_left_ext);

    if (!HaveEnoughCorners(corners_prev_left, keypoints_prev_left_ext))
    {
      return false;
    }

    if (keypoints_prev_left_ext.empty())
    {
      return HaveEnoughCorners(corners_prev_left)
             && corners_prev_left.size() == corners_prev_right.size();
    }

    // 按照响应值（response）降序排列，保留最显著的前 maxCorners 个角点，提取最优角点
    std::ranges::sort(keypoints_prev_left_ext, std::greater<>{},
                      &cv::KeyPoint::response);
    if (keypoints_prev_left_ext.size() > maxCorners)
    {
      keypoints_prev_left_ext.resize(maxCorners);
    }

    // 将 KeyPoint 映射回 PointType
    Points corners_prev_left_ext
        = keypoints_prev_left_ext
          | std::views::transform([](const cv::KeyPoint &kp)
                                  { return PointType{kp.pt}; })
          | std::ranges::to<std::vector>()
          | CreateSubPixAdaptor(gray_prev_left);

    Points corners_prev_right_ext;
    std::vector<unsigned char> features_found_pl_pr;

    // https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    CalcOpticalFlowPyrLK(gray_prev_left, gray_prev_right, //
                         corners_prev_left_ext,
                         corners_prev_right_ext, //
                         features_found_pl_pr, 0);

    auto zipped_view
        = std::views::zip(features_found_pl_pr, corners_prev_left_ext,
                          corners_prev_right_ext, feature_ids)
          | std::views::filter(
              [atol = atol_parallax](const auto &tuple)
              {
                const auto &[found, p_prev_left, p_prev_right, feature_id]
                    = tuple;
                // 1. 必须是追踪成功的点
                // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                // 3. 极线过滤：纵坐标之差必须小于阈值
                return found && (p_prev_left.x > p_prev_right.x)
                       && (std::abs(p_prev_left.y - p_prev_right.y) < atol);
              }
          );

    // 因为 view 是延迟计算的，所以必须先创建副本
    Points new_corners_prev_left_ext
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<1>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_prev_right_ext
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<2>(tuple); })
          | std::ranges::to<std::vector>();
    auto new_feature_ids
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<3>(tuple); })
          | std::ranges::to<std::vector>();

    // 合并到主追踪序列
    corners_prev_left.reserve(corners_prev_left.size()
                              + new_corners_prev_left_ext.size());
    corners_prev_right.reserve(corners_prev_right.size()
                               + new_corners_prev_right_ext.size());
    corners_prev_left.append_range(std::move(new_corners_prev_left_ext));
    corners_prev_right.append_range(std::move(new_corners_prev_right_ext));
    feature_ids = std::move(new_feature_ids);

    return HaveEnoughCorners(corners_prev_left)
           && corners_prev_left.size() == corners_prev_right.size();
  }

  //===================================
  // 从上一帧右目到下一帧右目
  bool TrackTemporalRight(const cv::Mat &gray_prev_right,
                          const cv::Mat &gray_next_right,
                          std::vector<PointType> &corners_prev_left,
                          std::vector<PointType> &corners_prev_right,
                          std::vector<PointType> &corners_next_left,
                          std::vector<PointType> &corners_next_right,
                          std::vector<std::uint32_t> &feature_ids,
                          bool use_hint) const
  {
    // 由于 corners_prev_right 的长度可能比 corners_next_right 的长度大
    // 函数 TrackStereoPrevLeftToPrevRight 新增的角点是追加在 corners_prev_right 的末尾
    // 所以可以把 corners_prev_right 末尾元素添加到 corners_next_right 的末尾
    if (use_hint)
    {
      if (corners_prev_left.size() > corners_next_left.size())
      {
        corners_next_left.insert(corners_next_left.end(),
                                 corners_prev_left.begin()
                                     + corners_next_left.size(),
                                 corners_prev_left.end());
      }

      if (corners_prev_right.size() > corners_next_right.size())
      {
        corners_next_right.insert(corners_next_right.end(),
                                  corners_prev_right.begin()
                                      + corners_next_right.size(),
                                  corners_prev_right.end());
      }
    }

    assert(corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_left.size()
           && corners_prev_right.size() == corners_next_right.size()
           && corners_prev_left.size() == feature_ids.size());

    std::vector<unsigned char> features_found_pr_nr;
    const int lk_flags_prev_right_to_next_right{
        (use_hint && !corners_next_right.empty()
         && corners_prev_right.size() == corners_next_right.size())
            ? cv::OPTFLOW_USE_INITIAL_FLOW
            : 0
    };
    CalcOpticalFlowPyrLK(gray_prev_right, gray_next_right, corners_prev_right,
                         corners_next_right, features_found_pr_nr,
                         lk_flags_prev_right_to_next_right);

    auto zipped_view = std::views::zip(features_found_pr_nr, corners_prev_left,
                                       corners_prev_right, corners_next_right,
                                       corners_next_left, feature_ids)
                       | std::views::filter(
                           [](const auto &tuple)
                           {
                             // 1. 必须是追踪成功的点
                             return std::get<0>(tuple);
                           }
                       );

    // 因为 view 是延迟计算的，所以必须先创建副本
    Points new_corners_prev_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<1>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_prev_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<2>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<3>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<4>(tuple); })
          | std::ranges::to<std::vector>();
    auto new_feature_ids
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<5>(tuple); })
          | std::ranges::to<std::vector>();

    corners_prev_left  = std::move(new_corners_prev_left);
    corners_prev_right = std::move(new_corners_prev_right);
    corners_next_right = std::move(new_corners_next_right);
    corners_next_left  = std::move(new_corners_next_left);
    feature_ids        = std::move(new_feature_ids);

    return HaveEnoughCorners(corners_prev_left)
           && corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_right.size()
           && corners_prev_left.size() == corners_next_left.size();
  }

  //===================================
  // 从下一帧右目到下一帧左目
  bool
  TrackStereoNextRightToNextLeft(const cv::Mat &gray_next_right,
                                 const cv::Mat &gray_next_left,
                                 std::vector<PointType> &corners_prev_left,
                                 std::vector<PointType> &corners_prev_right,
                                 std::vector<PointType> &corners_next_left,
                                 std::vector<PointType> &corners_next_right,
                                 std::vector<std::uint32_t> &feature_ids,
                                 bool use_hint) const
  {
    assert(corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_left.size()
           && corners_prev_left.size() == corners_next_right.size()
           && corners_prev_left.size() == feature_ids.size());

    std::vector<unsigned char> features_found_nr_nl;
    const int lk_flags_next_right_to_next_left{
        (use_hint && !corners_next_left.empty()
         && corners_next_right.size() == corners_next_left.size())
            ? cv::OPTFLOW_USE_INITIAL_FLOW
            : 0
    };
    CalcOpticalFlowPyrLK(gray_next_right, gray_next_left, corners_next_right,
                         corners_next_left, features_found_nr_nl,
                         lk_flags_next_right_to_next_left);

    auto zipped_view
        = std::views::zip(features_found_nr_nl, corners_prev_left,
                          corners_prev_right, corners_next_left,
                          corners_next_right, feature_ids)
          | std::views::filter(
              [atol = atol_parallax](const auto &tuple)
              {
                const auto &[found, p_prev_left, p_prev_right, p_next_left,
                             p_next_right, feature_id] = tuple;
                // 1. 必须是追踪成功的点
                // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                // 3. 极线过滤：纵坐标之差必须小于阈值
                return found && (p_next_left.x > p_next_right.x)
                       && (std::abs(p_next_left.y - p_next_right.y) < atol);
              }
          );

    Points new_corners_prev_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<1>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_prev_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<2>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<3>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<4>(tuple); })
          | std::ranges::to<std::vector>();
    auto new_feature_ids
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<5>(tuple); })
          | std::ranges::to<std::vector>();

    corners_prev_left  = std::move(new_corners_prev_left);
    corners_prev_right = std::move(new_corners_prev_right);
    corners_next_left  = std::move(new_corners_next_left);
    corners_next_right = std::move(new_corners_next_right);
    feature_ids        = std::move(new_feature_ids);

    return HaveEnoughCorners(corners_prev_left)
           && corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_left.size()
           && corners_prev_left.size() == corners_next_right.size();
  }

  //===================================
  // 从下一帧左目到上一帧左目 (闭环)
  bool ValidateLoopback(const cv::Mat &gray_next_left,
                        const cv::Mat &gray_prev_left,
                        std::vector<PointType> &corners_prev_left,
                        std::vector<PointType> &corners_prev_right,
                        std::vector<PointType> &corners_next_left,
                        std::vector<PointType> &corners_next_right,
                        std::vector<std::uint32_t> &feature_ids) const
  {
    assert(corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_left.size()
           && corners_prev_left.size() == corners_next_right.size()
           && corners_prev_left.size() == feature_ids.size());

    Points corners_prev_left_loopback;
    std::vector<unsigned char> features_found_nl_pl;
    CalcOpticalFlowPyrLK(gray_next_left, gray_prev_left, corners_next_left,
                         corners_prev_left_loopback, features_found_nl_pl, 0);

    auto zipped_view
        = std::views::zip(features_found_nl_pl, corners_prev_left,
                          corners_prev_right, corners_next_left,
                          corners_next_right, corners_prev_left_loopback,
                          feature_ids)
          | std::views::filter(
              [atol = atol_coincidence](const auto &tuple)
              {
                const auto &[found, p_prev_left, p_prev_right, p_next_left,
                             p_next_right, p_loop_back, feature_id] = tuple;
                // 1. 必须是追踪成功的点
                // 2. 重合过滤：横、纵坐标之差必须小于阈值
                return found && (std::abs(p_prev_left.x - p_loop_back.x) < atol)
                       && (std::abs(p_prev_left.y - p_loop_back.y) < atol);
              }
          );

    Points new_corners_prev_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<1>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_prev_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<2>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_left
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<3>(tuple); })
          | std::ranges::to<std::vector>();
    Points new_corners_next_right
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<4>(tuple); })
          | std::ranges::to<std::vector>();
    auto new_feature_ids
        = zipped_view
          | std::views::transform([](const auto &tuple)
                                  { return std::get<6>(tuple); })
          | std::ranges::to<std::vector>();

    corners_prev_left  = std::move(new_corners_prev_left);
    corners_prev_right = std::move(new_corners_prev_right);
    corners_next_left  = std::move(new_corners_next_left);
    corners_next_right = std::move(new_corners_next_right);
    feature_ids        = std::move(new_feature_ids);

    return HaveEnoughCorners(corners_prev_left)
           && corners_prev_left.size() == corners_prev_right.size()
           && corners_prev_left.size() == corners_next_left.size()
           && corners_prev_left.size() == corners_next_right.size()
           && corners_prev_left.size() == corners_prev_left_loopback.size();
  }
};

} // namespace FastVIO::CornerDetection
