#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
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

template <typename T>
concept OpenCV_Points
    = std::is_same_v<std::remove_cvref_t<T>, std::vector<cv::Point2f>>;

namespace FastVIO::CornerDetection
{

struct AbstractDetector
{
  static constexpr std::size_t minCorners{10};
  static constexpr std::size_t maxCorners{200};
  static_assert(minCorners <= maxCorners);
  static_assert(maxCorners
                < static_cast<std::size_t>(std::numeric_limits<int>::max()));
  static constexpr double atol_parallax{1.5};
  static constexpr double atol_coincidence{1.0};
  const cv::Size winSize{15, 15};
  static constexpr int maxLevel{2};

  template <typename PointType>
  static bool HaveEnoughCorners(const std::vector<PointType> &corners) noexcept
  {
    return corners.size() >= minCorners;
  }

  template <typename... PointTypes>
  static bool
  HaveEnoughCorners(const std::vector<PointTypes> &...corners) noexcept
  {
    // 各个角点集合的大小之和
    return (corners.size() + ...) >= minCorners;
  }
};

// 自定义适配器，用于包装 cv::cornerSubPix
struct SubPixAdaptor
{
  const cv::Mat &image_;
  cv::Size win_size_;
  cv::Size zero_zone_;
  cv::TermCriteria criteria_;

  // 重载 | 运算符
  friend decltype(auto) operator|(OpenCV_Points auto &&vec,
                                  const SubPixAdaptor &adaptor)
  {
    if (!vec.empty() && !adaptor.image_.empty())
    {
      // 计算安全边距（必须保证窗口能在图像内部完整展开）
      const float x_min{static_cast<float>(adaptor.win_size_.width + 1)};
      const float y_min{static_cast<float>(adaptor.win_size_.height + 1)};
      const float x_max{static_cast<float>(adaptor.image_.cols) - x_min};
      const float y_max{static_cast<float>(adaptor.image_.rows) - y_min};

      std::vector<cv::Point2f> valid_pts;
      std::vector<std::size_t> valid_indices;
      std::size_t len{vec.size()};
      valid_pts.reserve(len);
      valid_indices.reserve(len);

      // 仅对位于合法区域内的点做亚像素精化
      for (std::size_t i = 0; i < len; ++i)
      {
        const cv::Point2f &pt{vec[i]};
        const float x{static_cast<float>(pt.x)};
        const float y{static_cast<float>(pt.y)};
        if (x >= x_min && x <= x_max && y >= y_min && y <= y_max)
        {
          valid_pts.emplace_back(x, y);
          valid_indices.push_back(i);
        }
      }

      if (!valid_pts.empty())
      {
        cv::cornerSubPix(adaptor.image_, valid_pts, adaptor.win_size_,
                         adaptor.zero_zone_, adaptor.criteria_);

        for (std::size_t i = 0; i < valid_pts.size(); ++i)
        {
          const std::size_t orig_idx = valid_indices[i];
          vec[orig_idx]              = valid_pts[i];
        }
      }
    }

    return std::forward<decltype(vec)>(vec);
  }
};

} // namespace FastVIO::CornerDetection
