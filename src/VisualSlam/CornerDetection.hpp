#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
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

namespace CornerDetection
{

struct AbstractDetector
{
  static constexpr size_t minCorners{10};
  static constexpr size_t maxCorners{200};
  static constexpr double atol_parallax{2.0};
  static constexpr double atol_coincidence{1.5};
  const cv::Size winSize{15, 15};
  static constexpr int maxLevel{2};
};

} // namespace CornerDetection
