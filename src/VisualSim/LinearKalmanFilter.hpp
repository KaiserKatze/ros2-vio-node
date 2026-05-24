#pragma once

#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

// @see: https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html
// @see: https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html

struct LinearKalmanFilter
{
  cv::KalmanFilter kf_;

  // 状态数 (位置,线速度,线加速度,朝向角,角速度,角加速度)
  const int n_states_{18};
  // 测量数 (位置,朝向角)
  const int n_measurements_{6};
  // 控制数
  const int n_inputs_{0};
  // 测量的时间步长
  const double dt_{1.0 / 200.0};
  // 卡尔曼滤波要求的内点个数阈值 (所谓内点是指重投影误差足够小的样本)
  const int min_inliers_kalman_{30};

  LinearKalmanFilter()
  {
    kf_.init(n_states_, n_measurements_, n_inputs_, CV_64F);
    // 设置过程噪声
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));
    // 设置测量噪声
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-4));
    // 设置误差协方差
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

    /* DYNAMIC MODEL */

    // position
    kf_.transitionMatrix.at<double>(0, 3) = dt_;
    kf_.transitionMatrix.at<double>(1, 4) = dt_;
    kf_.transitionMatrix.at<double>(2, 5) = dt_;
    kf_.transitionMatrix.at<double>(3, 6) = dt_;
    kf_.transitionMatrix.at<double>(4, 7) = dt_;
    kf_.transitionMatrix.at<double>(5, 8) = dt_;
    kf_.transitionMatrix.at<double>(0, 6) = 0.5 * std::pow(dt_, 2);
    kf_.transitionMatrix.at<double>(1, 7) = 0.5 * std::pow(dt_, 2);
    kf_.transitionMatrix.at<double>(2, 8) = 0.5 * std::pow(dt_, 2);

    // orientation
    kf_.transitionMatrix.at<double>(9, 12)  = dt_;
    kf_.transitionMatrix.at<double>(10, 13) = dt_;
    kf_.transitionMatrix.at<double>(11, 14) = dt_;
    kf_.transitionMatrix.at<double>(12, 15) = dt_;
    kf_.transitionMatrix.at<double>(13, 16) = dt_;
    kf_.transitionMatrix.at<double>(14, 17) = dt_;
    kf_.transitionMatrix.at<double>(9, 15)  = 0.5 * std::pow(dt_, 2);
    kf_.transitionMatrix.at<double>(10, 16) = 0.5 * std::pow(dt_, 2);
    kf_.transitionMatrix.at<double>(11, 17) = 0.5 * std::pow(dt_, 2);

    /* MEASUREMENT MODEL */

    kf_.measurementMatrix.at<double>(0, 0)  = 1; // x
    kf_.measurementMatrix.at<double>(1, 1)  = 1; // y
    kf_.measurementMatrix.at<double>(2, 2)  = 1; // z
    kf_.measurementMatrix.at<double>(3, 9)  = 1; // roll
    kf_.measurementMatrix.at<double>(4, 10) = 1; // pitch
    kf_.measurementMatrix.at<double>(5, 11) = 1; // yaw
  }
};
