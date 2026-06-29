#pragma once

#include <cstdio>
#include <cstdlib>
#include <type_traits>
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

// 1. 定义基础 trait：默认不是 vector
template <typename T>
struct is_vector : std::false_type
{
};
// 2. 偏特化：匹配 std::vector 模板实例
// 注意：vector 有两个模板参数，一个是类型 T，一个是分配器 Allocator
template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type
{
};
// 3. 封装为 Concept
template <typename T>
concept StdVectorType = is_vector<std::remove_cvref_t<T>>::value;

namespace CornerDetection
{

struct AbstractDetector
{
  static constexpr size_t minCorners{10};
  static constexpr size_t maxCorners{200};
  static_assert(minCorners <= maxCorners,
                "minCorners must be less than or equal to maxCorners");
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
  static bool HaveEnoughCorners(const std::vector<PointTypes> &...corners)
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
  friend auto operator|(StdVectorType auto &&vec, const SubPixAdaptor &adaptor)
  {
    if (!vec.empty())
    {
      using VecType = std::remove_cvref_t<decltype(vec)>;
      using PointT  = typename VecType::value_type;
      if constexpr (std::is_same_v<PointT, cv::Point2f>)
      {
        cv::cornerSubPix(adaptor.image_, vec, adaptor.win_size_,
                         adaptor.zero_zone_, adaptor.criteria_);
      }
      else
      {
        std::vector<cv::Point2f> vec_f
            = vec
              | std::views::transform(
                  [](const auto &pt)
                  {
                    return cv::Point2f(static_cast<float>(pt.x),
                                       static_cast<float>(pt.y));
                  }
              )
              | std::ranges::to<std::vector>();

        cv::cornerSubPix(adaptor.image_, vec_f, adaptor.win_size_,
                         adaptor.zero_zone_, adaptor.criteria_);

        for (size_t i = 0; i < vec.size(); ++i)
        {
          vec[i].x = vec_f[i].x;
          vec[i].y = vec_f[i].y;
        }
      }
    }
    return vec;
  }
};

// 辅助工厂函数，类似于 std::views::transform
inline auto RefineSubPix(const cv::Mat &image, cv::Size win_size = {5, 5},
                         cv::Size zero_zone = {-1, -1},
                         cv::TermCriteria criteria
                         = {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40,
                            0.01})
{
  return SubPixAdaptor{image, win_size, zero_zone, criteria};
}

} // namespace CornerDetection
