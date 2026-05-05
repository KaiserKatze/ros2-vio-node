#pragma once

#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>

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

namespace EuRoC
{

struct EuRoC
{
  static constexpr double fu0{458.654};
  static constexpr double fv0{457.296};
  static constexpr double cu0{367.215};
  static constexpr double cv0{248.375};
  static constexpr double k01{-0.28340811};
  static constexpr double k02{0.07395907};
  static constexpr double p01{0.00019359};
  static constexpr double p02{1.76187114e-05};
  static constexpr double fu1{457.587};
  static constexpr double fv1{456.134};
  static constexpr double cu1{379.999};
  static constexpr double cv1{255.238};
  static constexpr double k11{-0.28368365};
  static constexpr double k12{0.07451284};
  static constexpr double p11{-0.00010473};
  static constexpr double p12{-3.55590700e-05};
  static constexpr int image_width{752};
  static constexpr int image_height{480};

  // https://libeigen.gitlab.io/eigen/docs-3.1/TopicStructHavingEigenMembers.html
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d mat_cam_intrinsic_rectified_;
  Eigen::Vector3d vec_cam_translation_rectified_;

  Eigen::Matrix4d T_C1C0;
  cv::Size imageSize{752, 480};

  // Output 4×4 disparity-to-depth mapping matrix
  cv::Mat Q;

  cv::Mat map0x;
  cv::Mat map0y;
  cv::Mat map1x;
  cv::Mat map1y;

  double focal_length_rectified_{NAN};
  double baseline_length_{NAN};

  EuRoC()
  {
    // 1. 初始化矩阵（确保使用 double 类型）

    // https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html
    // https://docs.opencv.org/4.x/d0/daf/group__core__eigen.html

    cv::Mat cameraMatrix0;
    cv::eigen2cv(
        Eigen::Matrix3d{
            {fu0, 0.0, cu0},
            {0.0, fv0, cv0},
            {0.0, 0.0, 1.0},
        },
        cameraMatrix0);
    cv::Mat distCoeffs0;
    cv::eigen2cv(
        Eigen::Vector4d{
            k01,
            k02,
            p01,
            p02,
        },
        distCoeffs0);

    cv::Mat cameraMatrix1;
    cv::eigen2cv(
        Eigen::Matrix3d{
            {fu1, 0.0, cu1},
            {0.0, fv1, cv1},
            {0.0, 0.0, 1.0},
        },
        cameraMatrix1);
    cv::Mat distCoeffs1;
    cv::eigen2cv(
        Eigen::Vector4d{
            k11,
            k12,
            p11,
            p12,
        },
        distCoeffs1);

    const Eigen::Matrix4d T_BC0{
        {0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975},
        {0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768},
        {-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949},
        {0.0, 0.0, 0.0, 1.0},
    };

    const Eigen::Matrix4d T_BC1{
        {0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556},
        {0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024},
        {-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038},
        {0.0, 0.0, 0.0, 1.0},
    };

    // X_B = T_BC0 * X_C0
    // X_B = T_BC1 * X_C1
    // X_C1 = T_C1C0 * X_C0 = (T_BC1.inverse() * T_BC0) * X_C0
    T_C1C0 = T_BC1.inverse() * T_BC0;

    // 2. 使用更安全的方法提取 R 和 T

    // Rotation matrix from the coordinate system of the first camera to the second camera
    cv::Mat stereoR(3, 3, CV_64FC1);
    {
      const Eigen::Matrix3d eigenMatR{
          T_C1C0(Eigen::seq(0, 2), Eigen::seq(0, 2))};
      // 提取左上角 3x3 矩阵作为旋转矩阵
      cv::eigen2cv(eigenMatR, stereoR);
      const Eigen::AngleAxisd rot_vec{eigenMatR};
      const double stereoRnorm{rot_vec.angle()};
      std::cerr << "变换 T_C1C0 对应的旋转向量 = " << eigenMatR << "\n"
                << "\t角度 = " << stereoRnorm << "\n";
    }

    // Translation vector from the coordinate system of the first camera to the second camera
    cv::Mat stereoT(3, 1, CV_64FC1);
    {
      const Eigen::Vector3d eigenVecT{
          T_C1C0(Eigen::seq(0, 2), Eigen::seq(3, 3))};
      // 提取第 4 列的前 3 行作为平移向量
      cv::eigen2cv(eigenVecT, stereoT);
      const double stereoTnorm{eigenVecT.norm()};
      std::cerr << "变换 T_C1C0 对应的平移向量 = " << eigenVecT
                << "\n"
                   "\t范数 = "
                << stereoTnorm << "\n";
      baseline_length_ = stereoTnorm;
    }

    // 3. 调用立体校正

    // Output 3x3 rectification transform (rotation matrix) for the first camera.
    // This matrix brings points given in the unrectified first camera's
    // coordinate system to points in the rectified first camera's coordinate
    // system. In more technical terms, it performs a change of basis from the
    // unrectified first camera's coordinate system to the rectified first
    // camera's coordinate system
    cv::Mat R0;
    // Output 3x3 rectification transform (rotation matrix) for the second camera.
    // This matrix brings points given in the unrectified second camera's
    // coordinate system to points in the rectified second camera's coordinate
    // system. In more technical terms, it performs a change of basis from the
    // unrectified second camera's coordinate system to the rectified second
    // camera's coordinate system
    cv::Mat R1;
    // Output 3x4 projection matrix in the new (rectified) coordinate systems for
    // the first camera, i.e. it projects points given in the rectified first
    // camera coordinate system into the rectified first camera's image
    cv::Mat P0;
    // Output 3x4 projection matrix in the new (rectified) coordinate systems for
    // the second camera, i.e. it projects points given in the rectified first
    // camera coordinate system into the rectified second camera's image
    cv::Mat P1;

    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    cv::stereoRectify(cameraMatrix0, distCoeffs0, cameraMatrix1, distCoeffs1,
                      imageSize, stereoR, stereoT, R0, R1, P0, P1, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);
    focal_length_rectified_ = P0.at<double>(0, 0);

    mat_cam_intrinsic_rectified_ = Eigen::Matrix3d{
        {P1.at<double>(0, 0), P1.at<double>(0, 1), P1.at<double>(0, 2)},
        {P1.at<double>(1, 0), P1.at<double>(1, 1), P1.at<double>(1, 2)},
        {P1.at<double>(2, 0), P1.at<double>(2, 1), P1.at<double>(2, 2)},
    };
    vec_cam_translation_rectified_ = Eigen::Vector3d{
        P1.at<double>(0, 3),
        P1.at<double>(1, 3),
        P1.at<double>(2, 3),
    };

    std::cerr << "R0 = " << R0 << "\n"
              << "R1 = " << R1 << "\n"
              << "P0 = " << P0 << "\n"
              << "P1 = " << P1 << "\n"
              << "Q = " << Q << "\n";

    std::cerr << "focal_length_rectified_ = " << focal_length_rectified_ << "\n"
              << "baseline_length_ = " << baseline_length_ << "\n";

    // 4. 初始化映射表

    // https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga0d37b45f780b32f63ed19c21aa9fd333
    cv::initUndistortRectifyMap(cameraMatrix0, distCoeffs0, R0, P0, imageSize,
                                CV_32FC1, map0x, map0y);
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize,
                                CV_32FC1, map1x, map1y);

    // cv::cv2eigen(P0, rectifiedCameraMatrix0);
    // cv::cv2eigen(P1, rectifiedCameraMatrix1);

    printf("EuRoC setup done\n");
  }

  EuRoC(const EuRoC &)            = delete;
  EuRoC &operator=(const EuRoC &) = delete;
  EuRoC(EuRoC &&)                 = delete;
  EuRoC &operator=(EuRoC &&)      = delete;

  std::pair<cv::Mat, cv::Mat> remap(const cv::Mat &image0,
                                    const cv::Mat &image1) const
  {
    // printf("Left Image Size: %d x %d\n", image0.size().width, image0.size().height);
    // printf("Right Image Size: %d x %d\n", image1.size().width, image1.size().height);
    cv::Mat rectified0;
    cv::Mat rectified1;
    // https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
    cv::remap(image0, rectified0, map0x, map0y, cv::INTER_LINEAR);
    cv::remap(image1, rectified1, map1x, map1y, cv::INTER_LINEAR);
    return std::make_pair(rectified0, rectified1);
  }

  std::pair<cv::Mat, cv::Mat> grayscale(const cv::Mat &rectified0,
                                        const cv::Mat &rectified1) const
  {
    cv::Mat gray0;
    cv::Mat gray1;
    // https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    cv::cvtColor(rectified0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectified1, gray1, cv::COLOR_BGR2GRAY);
    return std::make_pair(gray0, gray1);
  }
};

} // namespace EuRoC
