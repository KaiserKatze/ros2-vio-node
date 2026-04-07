#ifndef RCFILTER_HPP
#define RCFILTER_HPP

#include <opencv2/core.hpp>

/*****************************
 * RC 低通滤波器类
 *****************************/
class RCFilter
{
private:
  double alpha_;
  bool initialized_;
  cv::Vec3d prev_val_;

public:
  RCFilter(double alpha = 0.1)
      : alpha_(alpha), initialized_(false), prev_val_(0, 0, 0)
  {
  }

  cv::Vec3d Filter(const cv::Vec3d &val)
  {
    if (!initialized_)
    {
      prev_val_    = val;
      initialized_ = true;
      return val;
    }
    // 一阶低通滤波公式: Y_n = alpha * X_n + (1 - alpha) * Y_{n-1}
    prev_val_ = alpha_ * val + (1.0 - alpha_) * prev_val_;
    return prev_val_;
  }

  void Reset()
  {
    initialized_ = false;
    prev_val_    = cv::Vec3d(0, 0, 0);
  }
};

#endif /* RCFILTER_HPP */
