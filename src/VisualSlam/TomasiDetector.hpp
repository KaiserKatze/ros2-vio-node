#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

#include "CornerDetection.hpp"

namespace CornerDetection
{

template <typename PointType = cv::Point2f>
struct TomasiDetector : public AbstractDetector
{
private:
  // Quality level (percent of maximum)
  static constexpr double qualityLevel{0.01};
  // Min distance between corners
  static constexpr double minDistance{
      // 由于 EuRoC MAV 数据集的 V2_01_easy 子集中
      // 房间的进深、开间大约是 5 米
      // 所以最远处（例如墙上的）路标点的视差约为 10 个像素
      // 而距离更近的路标点的视差更大
      // 所以这里取值应该是 10.0
      10.0,
  };
  // Maximum pyramid level to construct
  static constexpr double blockSize{3.0};
  // true: Harris, false: Shi-Tomasi
  static constexpr bool useHarrisDetector{false};
  static constexpr double freeParamHarisDetector{0.04};
  const cv::TermCriteria criteria_{
      (cv::TermCriteria::COUNT) | (cv::TermCriteria::EPS),
      // Maximum number of iterations
      20,
      // Minimum change per iteration
      0.3,
  };

public:
  bool FindCorners(const cv::Mat &gray_left, const cv::Mat &gray_right,
                   std::vector<PointType> &corners_left,
                   std::vector<PointType> &corners_right) const
  {
    // https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    cv::goodFeaturesToTrack(gray_left, corners_left,
                            static_cast<int>(maxCorners), qualityLevel,
                            minDistance, cv::noArray(), blockSize,
                            useHarrisDetector, freeParamHarisDetector);

    if (corners_left.size() < minCorners)
    {
      return false;
    }

    std::vector<unsigned char> features_found;

    // https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    cv::calcOpticalFlowPyrLK(gray_left, gray_right, corners_left, corners_right,
                             features_found, cv::noArray(), winSize, maxLevel,
                             criteria_);

    // 压缩数据
    {
      std::cerr << "\t筛选前，角点个数 = " << corners_left.size() << "\n";

      // 将左目点、右目点、状态位三者打包
      auto zipped_view
          = std::views::zip(corners_left, corners_right, features_found)
            | std::views::filter(
                [atol = atol_parallax](const auto &tuple)
                {
                  const auto &[p_left, p_right, found] = tuple;
                  // 1. 必须是追踪成功的点
                  // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                  // 3. 极线过滤：纵坐标之差必须小于阈值
                  return found && (p_left.x > p_right.x)
                         && (std::abs(p_left.y - p_right.y) < atol);
                });

      // 因为 view 是延迟计算的，所以必须先创建副本，绝对不能使用 std::vector::assign 方法进行赋值
      std::vector<PointType> new_corners_left;
      std::vector<PointType> new_corners_right;
      for (const auto &[point_left, point_right, found] : zipped_view)
      {
        new_corners_left.push_back(point_left);
        new_corners_right.push_back(point_right);
      }

      corners_left  = std::move(new_corners_left);
      corners_right = std::move(new_corners_right);

      std::cerr << "\t筛选后，角点个数 = " << corners_left.size() << "\n";
    }

    if (corners_right.size() < minCorners)
    {
      return false;
    }

    return true;
  }

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
      std::vector<PointType> corners_prev_left_ext, corners_prev_right_ext;

      cv::goodFeaturesToTrack(gray_prev_left, corners_prev_left_ext,
                              static_cast<int>(maxCorners), qualityLevel,
                              minDistance, cv::noArray(), blockSize,
                              useHarrisDetector, freeParamHarisDetector);

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
        corners_prev_left.reserve(
            std::max(corners_prev_left.capacity(),
                     corners_prev_left.size() + corners_prev_left_ext.size()));
        corners_prev_right.reserve(std::max(
            corners_prev_right.capacity(),
            corners_prev_right.size() + corners_prev_right_ext.size()));
        corners_prev_left.insert(
            corners_prev_left.end(),
            std::make_move_iterator(corners_prev_left_ext.begin()),
            std::make_move_iterator(corners_prev_left_ext.end()));
        corners_prev_right.insert(
            corners_prev_right.end(),
            std::make_move_iterator(corners_prev_right_ext.begin()),
            std::make_move_iterator(corners_prev_right_ext.end()));
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
