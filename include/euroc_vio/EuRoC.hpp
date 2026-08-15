#pragma once

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <print>
#include <stdexcept>
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

#include "euroc_vio/SensorYaml.hpp"
#include "euroc_vio/VisionMode.hpp"

namespace EuRoC
{

struct EuRoC
{
  // https://libeigen.gitlab.io/eigen/docs-3.1/TopicStructHavingEigenMembers.html
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 图像分辨率, 由 cam0/sensor.yaml 的 resolution 字段给出
  int image_width{0};
  int image_height{0};

  Eigen::Matrix3d mat_cam_intrinsic_rectified_;
  Eigen::Vector3d vec_cam_translation_rectified_;

  Eigen::Matrix4d T_C1C0{Eigen::Matrix4d::Identity()};
  // 矫正后左目相机系到体坐标系(IMU)的变换 T_B,rectifiedC0。
  // 立体矫正把原始左目系又旋转了 R0, 因此 T_B,rectC0 = T_BC0 * R0^T,
  // ESKF 需要它才能把三角化出的相机系路标点搬到统一的体/世界坐标系
  Eigen::Matrix4d T_B_rectifiedC0{Eigen::Matrix4d::Identity()};
  cv::Size imageSize;

  // Output 3x4 projection matrix in the new (rectified) coordinate systems for
  // the first camera, i.e. it projects points given in the rectified first
  // camera coordinate system into the rectified first camera's image
  cv::Mat P0;
  // Output 3x4 projection matrix in the new (rectified) coordinate systems for
  // the second camera, i.e. it projects points given in the rectified first
  // camera coordinate system into the rectified second camera's image
  cv::Mat P1;
  // Output 4×4 disparity-to-depth mapping matrix
  cv::Mat Q;

  cv::Mat map0x;
  cv::Mat map0y;
  cv::Mat map1x;
  cv::Mat map1y;

  double focal_length_rectified_{NAN};
  double baseline_length_{NAN};

  // 标定参数全部来自数据集的 cam0/cam1 sensor.yaml, 不再硬编码,
  // 从而支持 MH / V1 / V2 等不同序列各自的内参与外参
  explicit EuRoC(const std::filesystem::path &path_mav0)
  {
    // 运行时检测数据集相机数目: 双目走立体校正, 单目走去畸变
    if (FastVIO::DetectVisionMode(path_mav0) == FastVIO::VisionMode::kStereo)
    {
      InitializeStereo(path_mav0);
    }
    else
    {
      InitializeMono(path_mav0);
    }
  }

private:
  // 保留完整: 立体校正初始化的中间量 (外参、stereoRectify 输入输出、矫正映射)
  // 全部写入成员, 拆分子函数需传递 8+ 个 cv::Mat/Eigen 参数, 故整体保留
  void InitializeStereo(const std::filesystem::path &path_mav0)
  {
    const FastVIO::SensorYaml config_cam0{
        ReadCameraYaml(path_mav0 / "cam0" / "sensor.yaml"),
    };
    const FastVIO::SensorYaml config_cam1{
        ReadCameraYaml(path_mav0 / "cam1" / "sensor.yaml"),
    };

    image_width  = config_cam0.resolution_.x();
    image_height = config_cam0.resolution_.y();
    imageSize    = cv::Size{image_width, image_height};
    if (config_cam1.resolution_ != config_cam0.resolution_)
    {
      throw std::runtime_error{
          std::format("左右目分辨率不一致: cam0 {}x{}, cam1 {}x{}.",
                      config_cam0.resolution_.x(), config_cam0.resolution_.y(),
                      config_cam1.resolution_.x(), config_cam1.resolution_.y())
      };
    }

    // 1. 初始化矩阵（确保使用 double 类型）

    // https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html
    // https://docs.opencv.org/4.x/d0/daf/group__core__eigen.html

    cv::Mat cameraMatrix0;
    cv::eigen2cv(MakeIntrinsicMatrix(config_cam0.intrinsics_), cameraMatrix0);
    cv::Mat distCoeffs0;
    cv::eigen2cv(Eigen::Vector4d{config_cam0.distortion_coefficients_},
                 distCoeffs0);

    cv::Mat cameraMatrix1;
    cv::eigen2cv(MakeIntrinsicMatrix(config_cam1.intrinsics_), cameraMatrix1);
    cv::Mat distCoeffs1;
    cv::eigen2cv(Eigen::Vector4d{config_cam1.distortion_coefficients_},
                 distCoeffs1);

    const Eigen::Matrix4d T_BC0{config_cam0.transform_matrix_};
    const Eigen::Matrix4d T_BC1{config_cam1.transform_matrix_};

    const auto rmat_BC0{T_BC0.template block<3, 3>(0, 0)};
    const auto tvec_BC0{T_BC0.template block<3, 1>(0, 3)};
    const auto rmat_BC1{T_BC1.template block<3, 3>(0, 0)};
    const auto tvec_BC1{T_BC1.template block<3, 1>(0, 3)};

    // X_B = T_BC0 * X_C0
    // X_B = T_BC1 * X_C1
    // X_C1 = T_C1C0 * X_C0 = (T_BC1.inverse() * T_BC0) * X_C0
    // T_C1C0 = T_BC1.inverse() * T_BC0;
    T_C1C0.block<3, 3>(0, 0) = rmat_BC1.transpose() * rmat_BC0;
    T_C1C0.block<3, 1>(0, 3) = rmat_BC1.transpose() * (tvec_BC0 - tvec_BC1);

    // 2. 使用更安全的方法提取 R 和 T

    // Rotation matrix from the coordinate system of the first camera to the second camera
    cv::Mat stereoR(3, 3, CV_64FC1);
    {
      const Eigen::Matrix3d eigenMatR{T_C1C0(Eigen::seq(0, 2),
                                             Eigen::seq(0, 2))};
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
      const Eigen::Vector3d eigenVecT{T_C1C0(Eigen::seq(0, 2),
                                             Eigen::seq(3, 3))};
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

    // https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
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

    {
      Eigen::Matrix3d R0_eigen;
      Eigen::Matrix3d R1_eigen;
      cv::cv2eigen(R0, R0_eigen);
      cv::cv2eigen(R1, R1_eigen);

      // 矫正后左目系 -> 体坐标系: 先由 R0^T 转回原始左目系, 再经 T_BC0 到体系
      T_B_rectifiedC0.block<3, 3>(0, 0) = rmat_BC0 * R0_eigen.transpose();
      T_B_rectifiedC0.block<3, 1>(0, 3) = tvec_BC0;

      Eigen::Matrix3d rmat_R0R1{R0_eigen * rmat_BC0.transpose() * rmat_BC1
                                * R1_eigen.transpose()};
      std::print(
          stderr,
          "C_R0R1 =\n"
          "\t[[{:.2f}, {:.2f}, {:.2f}],\n"
          "\t [{:.2f}, {:.2f}, {:.2f}],\n"
          "\t [{:.2f}, {:.2f}, {:.2f}]].\n",
          rmat_R0R1(0, 0), rmat_R0R1(0, 1), rmat_R0R1(0, 2), //
          rmat_R0R1(1, 0), rmat_R0R1(1, 1), rmat_R0R1(1, 2), //
          rmat_R0R1(2, 0), rmat_R0R1(2, 1), rmat_R0R1(2, 2)  //
      );
    }
  }

  // 单目初始化: 无立体校正, 投影矩阵 P0 = [K | 0], 去畸变映射仅左目;
  // P1/Q/map1x/map1y 保持为空, 双目专属成员不参与单目流程
  void InitializeMono(const std::filesystem::path &path_mav0)
  {
    const std::filesystem::path camera_directory
        = FastVIO::SelectMonoCameraDirectory(path_mav0);
    const FastVIO::SensorYaml config_camera{
        ReadCameraYaml(camera_directory / "sensor.yaml"),
    };

    image_width  = config_camera.resolution_.x();
    image_height = config_camera.resolution_.y();
    imageSize    = cv::Size{image_width, image_height};

    cv::Mat camera_matrix;
    cv::eigen2cv(MakeIntrinsicMatrix(config_camera.intrinsics_), camera_matrix);
    cv::Mat distortion_coefficients;
    cv::eigen2cv(config_camera.distortion_coefficients_,
                 distortion_coefficients);

    P0 = cv::Mat::zeros(3, 4, CV_64FC1);
    camera_matrix.copyTo(P0.colRange(0, 3));
    focal_length_rectified_ = camera_matrix.at<double>(0, 0);
    baseline_length_        = 0.0;
    mat_cam_intrinsic_rectified_
        = MakeIntrinsicMatrix(config_camera.intrinsics_);
    vec_cam_translation_rectified_ = Eigen::Vector3d::Zero();

    // 恒等矫正旋转: 映射只去畸变不改变相机系
    cv::initUndistortRectifyMap(camera_matrix, distortion_coefficients,
                                cv::Mat::eye(3, 3, CV_64F), camera_matrix,
                                imageSize, CV_32FC1, map0x, map0y);

    T_B_rectifiedC0 = config_camera.transform_matrix_;

    std::print("EuRoC setup done (mono)\n");
  }

public:
  EuRoC(const EuRoC &)            = delete;
  EuRoC &operator=(const EuRoC &) = delete;
  EuRoC(EuRoC &&)                 = delete;
  EuRoC &operator=(EuRoC &&)      = delete;

private:
  // 读取相机 sensor.yaml; SensorYaml 只在 sensor_type 为 camera 时填充内参,
  // 因此此处显式校验分辨率, 避免把默认零值当作有效标定继续往下算
  static FastVIO::SensorYaml
  ReadCameraYaml(const std::filesystem::path &path_sensor_yaml)
  {
    const auto config{FastVIO::SensorYaml::ReadSensorYaml(path_sensor_yaml)};
    if (!config.has_value())
    {
      throw std::runtime_error{std::format("无法解析相机标定 '{}'.",
                                           path_sensor_yaml.string())};
    }
    if (config->resolution_.x() <= 0 || config->resolution_.y() <= 0)
    {
      throw std::runtime_error{
          std::format("'{}' 不是有效的相机标定 (缺少 resolution/intrinsics, "
                      "请确认 sensor_type 为 camera).",
                      path_sensor_yaml.string())
      };
    }
    return config.value();
  }

  // intrinsics 按 EuRoC 约定为 [fu, fv, cu, cv]
  static Eigen::Matrix3d MakeIntrinsicMatrix(const Eigen::Vector4d &intrinsics)
  {
    return Eigen::Matrix3d{
        {intrinsics(0), 0.0, intrinsics(2)},
        {0.0, intrinsics(1), intrinsics(3)},
        {0.0, 0.0, 1.0},
    };
  }

public:
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
