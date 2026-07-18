// clang-format on

#include <algorithm>
#include <chrono> // time module
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "euroc_vio/msg/imu.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform.h>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>
#include <message_filters/time_synchronizer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <sensor_msgs/msg/temperature.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

#include "euroc_vio/QuEst.hpp"
#include "euroc_vio/main.h"

using namespace sensor_msgs;
using namespace message_filters;
using namespace std::chrono_literals;

inline static constexpr char PATH_CSV_FILE[] = "euroc_vio.csv";

template <typename VectorType>
std::vector<std::size_t> KeepSmallestElement(const VectorType &input,
                                             std::size_t maxCount)
{
  if (maxCount == 0)
  {
    return {};
  }
  const auto buildIndexedVec = [](const VectorType &vec)
  {
    std::vector<std::pair<typename VectorType::value_type, std::size_t>>
        indexed;
    indexed.reserve(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
      indexed.emplace_back(vec[i], i);
    }
    return indexed;
  };
  auto indexed = buildIndexedVec(input);
  auto compare = [](const auto &a, const auto &b) { return a.first < b.first; };
  if (maxCount >= indexed.size())
  {
    std::sort(indexed.begin(), indexed.end(), compare);
    std::vector<std::size_t> indices;
    indices.reserve(indexed.size());
    for (const auto &p : indexed)
    {
      indices.push_back(p.second);
    }
    return indices;
  }
  std::nth_element(indexed.begin(), indexed.begin() + maxCount - 1,
                   indexed.end(), compare);
  // 前 maxCount 个元素不一定有序，需要排序
  std::sort(indexed.begin(), indexed.begin() + maxCount, compare);
  std::vector<std::size_t> indices;
  indices.reserve(maxCount);
  for (std::size_t i = 0; i < maxCount; ++i)
  {
    indices.push_back(indexed[i].second);
  }
  return indices;
}

/*****************************
 * ROS2 Node
 *****************************/

using MsgImage = sensor_msgs::msg::Image;
using MsgImu   = euroc_vio::msg::Imu;
using ApproximateTime_t
    = message_filters::sync_policies::ApproximateTime<MsgImage, MsgImage,
                                                      MsgImu>;
using Synchronizer_t = message_filters::Synchronizer<ApproximateTime_t>;

template <typename RosMsgType>
auto ConvertImage(const RosMsgType &msg)
{
  return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
}

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

  // https://libeigen.gitlab.io/eigen/docs-3.1/TopicStructHavingEigenMembers.html
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d rectifiedCameraMatrix0;
  Eigen::Matrix3d rectifiedCameraMatrix1;
  cv::Mat T_C1C0;
  cv::Size imageSize{752, 480};
  cv::Mat map0x;
  cv::Mat map0y;
  cv::Mat map1x;
  cv::Mat map1y;

  EuRoC()
  {
    // 1. 初始化矩阵（确保使用 double 类型）
    cv::Mat cameraMatrix0 = (cv::Mat_<double>(3, 3) << fu0, 0.0, cu0, 0.0, fv0,
                             cv0, 0.0, 0.0, 1.0);
    cv::Mat distCoeffs0   = (cv::Mat_<double>(4, 1) << k01, k02, p01, p02);
    cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) << fu1, 0.0, cu1, 0.0, fv1,
                             cv1, 0.0, 0.0, 1.0);
    cv::Mat distCoeffs1   = (cv::Mat_<double>(4, 1) << k11, k12, p11, p12);

    cv::Mat T_BC0
        = (cv::Mat_<double>(4, 4) << 0.0148655429818, -0.999880929698,
           0.00414029679422, -0.0216401454975, 0.999557249008, 0.0149672133247,
           0.025715529948, -0.064676986768, -0.0257744366974, 0.00375618835797,
           0.999660727178, 0.00981073058949, 0.0, 0.0, 0.0, 1.0);

    cv::Mat T_BC1
        = (cv::Mat_<double>(4, 4) << 0.0125552670891, -0.999755099723,
           0.0182237714554, -0.0198435579556, 0.999598781151, 0.0130119051815,
           0.0251588363115, 0.0453689425024, -0.0253898008918, 0.0179005838253,
           0.999517347078, 0.00786212447038, 0.0, 0.0, 0.0, 1.0);

    T_C1C0 = T_BC1.inv() * T_BC0;

    // 2. 使用更安全的方法提取 R 和 T
    // 提取左上角 3x3 矩阵作为旋转矩阵
    // Rotation matrix from the coordinate system of the first camera to the second camera
    cv::Mat stereoR = T_C1C0(cv::Range(0, 3), cv::Range(0, 3)).clone();
    // 提取第 4 列的前 3 行作为平移向量
    // Translation vector from the coordinate system of the first camera to the second camera
    cv::Mat stereoT = T_C1C0(cv::Range(0, 3), cv::Range(3, 4)).clone();

    // 3. 打印一下确认矩阵是否为空 (调试用)
    if (stereoR.empty())
    {
      throw std::runtime_error{"stereoR is empty"};
    }
    if (stereoT.empty())
    {
      throw std::runtime_error{"stereoT is empty"};
    }

    // 4. 调用立体校正

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
    // Output 4×4 disparity-to-depth mapping matrix
    cv::Mat Q;

    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    cv::stereoRectify(cameraMatrix0, distCoeffs0, cameraMatrix1, distCoeffs1,
                      imageSize, stereoR, stereoT, R0, R1, P0, P1, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);

    // 5. 初始化映射表
    // https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga0d37b45f780b32f63ed19c21aa9fd333
    cv::initUndistortRectifyMap(cameraMatrix0, distCoeffs0, R0, P0, imageSize,
                                CV_32FC1, map0x, map0y);
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize,
                                CV_32FC1, map1x, map1y);

    cv::cv2eigen(P0, rectifiedCameraMatrix0);
    cv::cv2eigen(P1, rectifiedCameraMatrix1);

    printf("EuRoC setup done\n");
  }

  EuRoC(const EuRoC &)            = delete;
  EuRoC &operator=(const EuRoC &) = delete;
  EuRoC(EuRoC &&)                 = delete;
  EuRoC &operator=(EuRoC &&)      = delete;

  auto remap(const cv::Mat &image0, const cv::Mat &image1) const
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

  auto grayscale(const cv::Mat &rectified0, const cv::Mat &rectified1) const
  {
    cv::Mat gray0;
    cv::Mat gray1;
    // https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    cv::cvtColor(rectified0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectified1, gray1, cv::COLOR_BGR2GRAY);
    return std::make_pair(gray0, gray1);
  }

  template <typename PointType>
  auto Solve(std::vector<PointType> const &pts0,
             std::vector<PointType> const &pts1) const
  {
    const std::size_t numPts{pts0.size()};
    if (numPts != pts1.size())
    {
      throw std::runtime_error{"pts0 and pts1 must have the same size"};
    }
    Eigen::Matrix3Xd M0(3, numPts);
    Eigen::Matrix3Xd M1(3, numPts);
    for (std::size_t i{0}; i < numPts; ++i)
    {
      const PointType &pt0{pts0[i]};
      const PointType &pt1{pts1[i]};
      M0(0, i) = pt0.x;
      M0(1, i) = pt0.y;
      M0(2, i) = 1;
      M1(0, i) = pt1.x;
      M1(1, i) = pt1.y;
      M1(2, i) = 1;
    }
    return QuEst_Solve(M0, M1, rectifiedCameraMatrix0, rectifiedCameraMatrix1);
  }
};

struct OpticalFLow_ShiTomasi
{
};

struct OpticalFlow_Fast
{
  static constexpr int fastThreshold{20};
  static constexpr bool fastNonmaxSuppression{true};
  static constexpr cv::FastFeatureDetector::DetectorType fastType{
      cv::FastFeatureDetector::TYPE_9_16
  };
  cv::Ptr<cv::FastFeatureDetector> fastFeatureDetector;

  OpticalFlow_Fast()
  {
    fastFeatureDetector
        = cv::FastFeatureDetector::create(fastThreshold, fastNonmaxSuppression,
                                          fastType);
  }
};

struct Corners
{
  std::vector<cv::Point2i> corners_l0;
  std::vector<cv::Point2i> corners_r0;
  std::vector<cv::Point2i> corners_l1;
  std::vector<cv::Point2i> corners_r1;
};

struct CornerPair
{
  std::size_t frame_index;
  cv::Point2i corner_left;
  cv::Point2i corner_right;
};

// struct SLAM
// {
//   std::vector<cv::Point3f> Recover(std::vector<cv::Point2i> const &pts0,
//                                    std::vector<cv::Point2i> const &pts1,
//                                    Eigen::Vector4d const &quadRotation,
//                                    Eigen::Vector3d const &vecTranslation) const
//   {
//     if (pts0.size() != pts1.size())
//     {
//       throw std::invalid_argument{"pts0 and pts1 must have the same size"};
//     }

//     Eigen::Matrix3d matRotation;
//     Q2R_3by3(matRotation, quadRotation);
//     (void) vecTranslation;

//     std::vector<cv::Point3f> result;
//     result.reserve(pts0.size());

//     for (std::size_t i{0}; i < pts0.size(); ++i)
//     {
//       const cv::Point2i &pt0{pts0[i]};
//       const cv::Point2i &pt1{pts1[i]};
//       (void) pt0;
//       (void) pt1;
//       // TODO
//     }
//   }
// };

class VirsualInertialOdemetry : public rclcpp::Node,
#if CORNER_DETECTION_ALGORITHM == CORNER_USE_FAST
                                public OpticalFlow_Fast
#elif CORNER_DETECTION_ALGORITHM == CORNER_USE_SHITOMASI
                                public OpticalFLow_ShiTomasi
#else
#error                                                                         \
    "CORNER_DETECTION_ALGORITHM must be CORNER_USE_FAST or CORNER_USE_SHITOMASI"
#endif
{
private:
  std::shared_ptr<Synchronizer_t> sync;
  message_filters::Subscriber<MsgImage> cam0_sub;
  message_filters::Subscriber<MsgImage> cam1_sub;
  message_filters::Subscriber<MsgImu> imu_sub;

  bool first{true};
  struct
  {
    cv::Mat rectified0;
    cv::Mat rectified1;
    cv::Mat gray0;
    cv::Mat gray1;
    struct
    {
      double w;
      double x;
      double y;
      double z;
    } rotation;
    struct
    {
      double x;
      double y;
      double z;
    } translation;
  } prev;

  /* QuEst 算法支持 5 个以上角点 */
  static constexpr std::size_t minCorners{5};
  static constexpr int maxCorners{100};
  static constexpr double qualityLevel{0.3};
  static constexpr double minDistance{7.0};
  static constexpr double blockSize{7.0};
  static constexpr bool useHarrisDetector{false};
  static constexpr double freeParamHarisDetector{0.04};
  static constexpr double atol_parallax{2.0};
  static constexpr double atol_coincidence{1.0};

  const cv::Size winSize{15, 15};
  static constexpr int maxLevel{2};

  std::vector<cv::Scalar> colors;
  static constexpr std::size_t nColors{255};

  const EuRoC euroc{};

  // 维护一个表，统计各个角点的 age 值
  std::vector<std::vector<CornerPair>> corner_track_store;

  // 记录已经处理的图像帧的个数
  std::size_t frame_counter{0};

  std::fstream file;

  /**
   * @brief 初始化角点
   *
   * @param gray 灰度图像
   */
  void InitCorners(const cv::Mat &gray, std::vector<cv::Point2f> &corners) const
  {
    corners.clear();
    if constexpr (CORNER_DETECTION_ALGORITHM == CORNER_USE_SHITOMASI)
    {
      cv::Mat mask{};
      // https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
      cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel,
                              minDistance, mask, blockSize, useHarrisDetector,
                              freeParamHarisDetector);
    }
    else // FAST 算法
    {
      // 使用 FAST 角点检测
      // 文档：
      // - FastFeatureDetector（C++）：https://docs.opencv.org/4.13.0/df/d74/classcv_1_1FastFeatureDetector.html
      // - FAST 教程（Python）：https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html
      std::vector<cv::KeyPoint> keypoints;
      // https://docs.opencv.org/4.13.0/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
      fastFeatureDetector->detect(gray, keypoints);
      if (!keypoints.empty())
      {
        // 以响应值从高到低排序，截断至 maxCorners，便于与 Shi-Tomasi 行为保持一致
        std::sort(keypoints.begin(), keypoints.end(),
                  [](const cv::KeyPoint &a, const cv::KeyPoint &b)
                  { return a.response > b.response; });
        if (static_cast<int>(keypoints.size()) > maxCorners)
        {
          keypoints.resize(maxCorners);
        }
        // KeyPoint -> Point2f
        cv::KeyPoint::convert(keypoints, corners);
      }
    }
  }

  auto FilterWithStatus(std::vector<cv::Point2f> &pts,
                        std::vector<unsigned char> const &status) const
  {
    std::vector<cv::Point2f> result;
    for (std::size_t i{0}; i < pts.size(); ++i)
    {
      if (status[i] == 1)
      {
        result.push_back(pts[i]);
      }
    }
    pts = result;
  }

  template <typename RotationType>
  void ConvertRotation(RotationType const &rotation,
                       Eigen::Matrix3d &matRotation) const
  {

    // 旋转四元数
    Eigen::Vector4d quatRotation;
    quatRotation << rotation.w, rotation.x, rotation.y, rotation.z;
    // 将旋转四元数转化为旋转矩阵
    Q2R_3by3(matRotation, quatRotation);
  }

  template <typename TranslationType>
  void ConvertTranslation(TranslationType const &translation,
                          Eigen::Vector3d &vecTranslation) const
  {
    // 平移向量
    vecTranslation << translation.x, translation.y, translation.z;
  }

  // void RecoverLandmark(cv::Point2f const &ptImage,
  //                      Eigen::Matrix3d const &matRotation,
  //                      Eigen::Vector3d const &vecTranslation,
  //                      Eigen::Matrix3d const &matCamera,
  //                      Eigen::Vector3d &vecLandmarkNonhomo) const
  // {
  //   // 像素点的齐次坐标
  //   Eigen::Vector3d vecImageHomo;
  //   vecImageHomo << ptImage.x, ptImage.y, 1.0;
  //   // 恢复三维点的非齐次坐标
  //   vecLandmarkNonhomo = (matCamera * matRotation)
  //                            .colPivHouseholderQr()
  //                            .solve(vecImageHomo - matCamera * vecTranslation);
  // }

  cv::Point2f PredictNextCorner(cv::Point2f const &prevCorner,
                                Eigen::Matrix3d const &matRotation,
                                Eigen::Vector3d const &vecTranslation,
                                Eigen::Matrix3d const &matCamera) const
  {
    Eigen::Vector3d vecImageHomo;
    vecImageHomo << prevCorner.x, prevCorner.y, 1.0;
    Eigen::Vector3d vecNewImageHomo{
        matRotation
        * (matRotation * matCamera.inverse() * vecImageHomo + vecTranslation)
    };
    return cv::Point2f(
        static_cast<float>(vecNewImageHomo(0, 0) / vecNewImageHomo(2, 0)),
        static_cast<float>(vecNewImageHomo(1, 0) / vecNewImageHomo(2, 0))
    );
  }

  void PredictNextCorners(std::vector<cv::Point2f> const &prevCorners,
                          std::vector<cv::Point2f> &nextCorners,
                          Eigen::Matrix3d const &matRotation,
                          Eigen::Vector3d const &vecTranslation,
                          Eigen::Matrix3d const &matCamera) const
  {
    nextCorners.reserve(prevCorners.size());
    for (cv::Point2f const &corner : prevCorners)
    {
      nextCorners.push_back(PredictNextCorner(corner, matRotation,
                                              vecTranslation, matCamera));
    }
  }

  void PredictNextCornersForSequentialFrame(
      std::vector<cv::Point2f> const &prevCorners,
      std::vector<cv::Point2f> &nextCorners, Eigen::Matrix3d const &matCamera
  ) const
  {
    Eigen::Matrix3d matRotation;
    Eigen::Vector3d vecTranslation;
    ConvertRotation(prev.rotation, matRotation);
    ConvertTranslation(prev.translation, vecTranslation);
    return PredictNextCorners(prevCorners, nextCorners, matRotation,
                              vecTranslation, matCamera);
  }

  void
  PredictNextCornersForStereoFrame(std::vector<cv::Point2f> const &prevCorners,
                                   std::vector<cv::Point2f> &nextCorners,
                                   Eigen::Matrix3d const &matCamera,
                                   bool direction) const
  {
    cv::Mat T_direction = direction ? T_C1C0 : T_C1C0.inv();
    // 提取左上角 3x3 矩阵作为旋转矩阵
    cv::Mat stereoR = T_direction(cv::Range(0, 3), cv::Range(0, 3)).clone();
    // 提取第 4 列的前 3 行作为平移向量
    cv::Mat stereoT = T_direction(cv::Range(0, 3), cv::Range(3, 4)).clone();
    // 将旋转矩阵和平移向量转换为 Eigen::Matrix3d 和 Eigen::Vector3d
    Eigen::Matrix3d matRotation;
    Eigen::Vector3d vecTranslation;
    cv::cv2eigen(stereoR, matRotation);
    cv::cv2eigen(stereoT, vecTranslation);
    return PredictNextCorners(prevCorners, nextCorners, matRotation,
                              vecTranslation, matCamera);
  }

  std::vector<unsigned char>
  MatchCorners0(const cv::Mat &prevGray, const cv::Mat &nextGray,
                std::vector<cv::Point2f> &prevCorners,
                std::vector<cv::Point2f> &nextCorners,
                Eigen::Matrix3d const &matCamera) const
  {
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::TermCriteria criteria(
        (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03
    );
    PredictNextCorners(prevCorners, nextCorners, matCamera);
    // https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    cv::calcOpticalFlowPyrLK(prevGray, nextGray, prevCorners, nextCorners,
                             status, err, winSize, maxLevel, criteria);

    FilterWithStatus(prevCorners, status);
    FilterWithStatus(nextCorners, status);
    return status;
  }

  /**
   * @brief 匹配角点
   *
   * @param prevGray 上一帧灰度图像
   * @param nextGray 当前帧灰度图像
   * @param prevCorners 上一帧角点
   * @param nextCorners 当前帧角点
   * @param listCorners 其他角点
   */
  void MatchCorners(
      const cv::Mat &prevGray, const cv::Mat &nextGray,
      std::vector<cv::Point2f> &prevCorners,
      std::vector<cv::Point2f> &nextCorners,
      const std::vector<std::reference_wrapper<std::vector<cv::Point2f>>>
          &&listCorners,
      Eigen::Matrix3d const &matCamera
  ) const
  {
    auto status{MatchCorners0(prevGray, nextGray, prevCorners, nextCorners,
                              matCamera)};
    for (auto otherCorners : listCorners)
    {
      FilterWithStatus(otherCorners.get(), status);
    }
  }

  template <typename VectorType, typename MaskType>
  VectorType FilterWithMask(VectorType &pts, MaskType &mask) const
  {
    VectorType filteredPts;
    for (std::size_t i{0}; i < pts.size(); ++i)
    {
      if (mask[i])
      {
        filteredPts.push_back(pts[i]);
      }
    }
    return filteredPts;
  }

  template <typename MaskType>
  std::size_t MaskLength(MaskType mask) const
  {
    return std::ranges::count_if(mask, [](bool e) { return e; });
  }

  auto FilterParallax0(std::vector<cv::Point2f> &ptsLeft,
                       std::vector<cv::Point2f> &ptsRight,
                       std::vector<double> &error, double atol = 1.0) const
  {
    if (ptsLeft.size() != ptsRight.size())
    {
      throw std::invalid_argument{
          "ptsLeft and ptsRight must have the same size"
      };
    }
    if (atol <= 0.0)
    {
      throw std::invalid_argument{"atol must be positive"};
    }
    const auto compare = [atol, &error](cv::Point2f ptMinus) -> bool
    {
      const auto absParallaxY{std::abs(ptMinus.y)};
      if (absParallaxY < atol && ptMinus.x > 0.0)
      {
        error.push_back(absParallaxY);
        return true;
      }
      return false;
    };
    const std::vector<bool> mask{CreateMask(ptsLeft, ptsRight, compare)};
    // 得到的 error 向量与筛选后的 ptsLeft 等长
    if (const std::size_t countCorners{MaskLength(mask)};
        countCorners < minCorners)
    {
      std::stringstream ss;
      ss << "Not enough corners after parallax filtering, "
         << "only " << countCorners << " corners left";
      throw std::runtime_error{ss.str()};
    }
    ptsLeft  = std::move(FilterWithMask(ptsLeft, mask));
    ptsRight = std::move(FilterWithMask(ptsRight, mask));
    return mask;
  }

  /**
   * @brief 依据“视差一致性”原则，进行筛选
   *
   * @param ptsLeft 第一组角点
   * @param ptsRight 第二组角点
   * @param atol 绝对误差阈值
   * @return true 视差验证通过
   * @return false 视差验证失败
   */
  bool FilterParallax(
      std::vector<cv::Point2f> &ptsLeft, std::vector<cv::Point2f> &ptsRight,
      const std::vector<std::reference_wrapper<std::vector<cv::Point2f>>>
          &&listPts,
      std::vector<double> &sumL1Error, double atol = 1.0
  ) const
  {
    try
    {
      std::vector<double> error;
      auto mask{FilterParallax0(ptsLeft, ptsRight, error, atol)};

      for (auto otherPts : listPts)
      {
        otherPts.get() = std::move(FilterWithMask(otherPts.get(), mask));
      }

      sumL1Error = std::move(FilterWithMask(sumL1Error, mask));
      if (sumL1Error.size() != error.size())
      {
        throw std::runtime_error{
            "sumL1Error and error must have the same size"
        };
      }
      for (std::size_t i{0}; i < sumL1Error.size(); ++i)
      {
        sumL1Error[i] += error[i];
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << std::endl;
      return false;
    }
    return true;
  }

  template <typename Compare>
  std::vector<bool> CreateMask(const std::vector<cv::Point2f> &pts0,
                               const std::vector<cv::Point2f> &pts1,
                               Compare compare) const
  {
    std::vector<bool> mask(pts0.size(), false);
    for (std::size_t i{0}; i < pts0.size(); ++i)
    {
      const cv::Point2f ptMinus{pts0[i] - pts1[i]};
      if (compare(ptMinus))
      {
        mask[i] = true;
      }
    }
    return mask;
  }

  /**
   * @brief 检查角点的横纵坐标是否都足够接近，进行筛选
   *
   * @param ptsInit 初始组角点
   * @param ptsLast 最后组角点
   * @param atol 绝对误差阈值
   * @return true 一致性验证通过
   * @return false 一致性验证失败
   */
  bool FilterCoincidence(
      std::vector<cv::Point2f> &ptsInit, std::vector<cv::Point2f> &ptsLast,
      const std::vector<std::reference_wrapper<std::vector<cv::Point2f>>>
          &&listPts,
      std::vector<double> &sumL1Error, double atol = 1.0
  ) const
  {
    if (ptsInit.size() != ptsLast.size())
    {
      throw std::invalid_argument{
          "ptsInit and ptsLast must have the same size"
      };
    }
    if (atol <= 0.0)
    {
      throw std::invalid_argument{"atol must be positive"};
    }

    std::vector<double> error;
    const auto compare = [atol, &error](cv::Point2f ptMinus) -> bool
    {
      const auto absParallaxX{std::abs(ptMinus.x)};
      const auto absParallaxY{std::abs(ptMinus.y)};
      if (absParallaxX < atol && absParallaxY < atol)
      {
        error.push_back(absParallaxX + absParallaxY);
        return true;
      }
      return false;
    };
    std::vector<bool> mask{CreateMask(ptsInit, ptsLast, compare)};

    if (MaskLength(mask) < minCorners)
    {
      return false;
    }
    ptsInit = std::move(FilterWithMask(ptsInit, mask));
    ptsLast = std::move(FilterWithMask(ptsLast, mask));

    for (auto otherPts : listPts)
    {
      otherPts.get() = std::move(FilterWithMask(otherPts.get(), mask));
    }

    sumL1Error = std::move(FilterWithMask(sumL1Error, mask));
    if (sumL1Error.size() != error.size())
    {
      throw std::runtime_error{"sumL1Error and error must have the same size"};
    }
    for (std::size_t i{0}; i < sumL1Error.size(); ++i)
    {
      sumL1Error[i] += error[i];
    }
    return true;
  }

  /**
   * @brief 找到左右目图像中的角点
   *
   * @param gray_l1 左目灰度图像
   * @param gray_r1 右目灰度图像
   */
  bool FindCorners(const cv::Mat &gray_l1, const cv::Mat &gray_r1,
                   Corners &corners, const std::size_t frame_index)
  {
    std::vector<double> sumL1Error;
    // 上一帧左目角点
    std::vector<cv::Point2f> corners_l0;

    InitCorners(prev.gray0, corners_l0);

    // TODO 在 corners_l0, corners.corners_l1 中找出共同点，跟踪角点的 age 数值（即角点在相同相机、不同图像帧中出现的次数）

    printf("Found %zu corners in prev.gray0\n", corners_l0.size());
    if (corners_l0.size() < minCorners)
    {
      return false;
    }
    // 上一帧右目角点
    std::vector<cv::Point2f> corners_r0;
    MatchCorners0(prev.gray0, prev.gray1, corners_l0, corners_r0);
    printf("Found %zu matching corners in prev.gray1\n", corners_r0.size());
    if (corners_r0.size() < minCorners)
    {
      return false;
    }
    try
    {
      FilterParallax0(corners_l0, corners_r0, sumL1Error, atol_parallax);
      // 首次调用 FilterParallax0 以后，所得 sumL1Error 向量就是筛选通过后各个角点的视差的垂直分量的绝对值
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << std::endl;
      return false;
    }
    printf(
        "Found %zu matching corners in prev.gray1 after parallax filtering\n",
        corners_r0.size()
    );
    // 当前帧右目角点
    std::vector<cv::Point2f> corners_r1;
    MatchCorners(prev.gray1, gray_r1, corners_r0, corners_r1, {corners_l0});
    printf("Found %zu matching corners in gray_r1\n", corners_r1.size());
    if (corners_r1.size() < minCorners)
    {
      return false;
    }
    // 当前帧左目角点
    std::vector<cv::Point2f> corners_l1;
    MatchCorners(gray_r1, gray_l1, corners_r1, corners_l1,
                 {corners_l0, corners_r0});
    printf("Found %zu matching corners in gray_l1\n", corners_l1.size());
    if (corners_l1.size() < minCorners)
    {
      return false;
    }
    if (!FilterParallax(corners_l1, corners_r1, {corners_l0, corners_r0},
                        sumL1Error, atol_parallax))
    {
      return false;
    }
    printf("Found %zu matching corners in gray_l1 after parallax filtering\n",
           corners_l1.size());
    // 回到上一帧左目角点
    std::vector<cv::Point2f> corners_loopback;
    MatchCorners(gray_l1, prev.gray0, corners_l1, corners_loopback,
                 {corners_l0, corners_r0, corners_r1});
    printf("Found %zu matching corners in prev.gray0\n",
           corners_loopback.size());
    if (corners_loopback.size() < minCorners)
    {
      return false;
    }
    // 检查 corners_l0 与 corners_loopback 是否足够接近
    if (!FilterCoincidence(corners_l0, corners_loopback,
                           {corners_r0, corners_l1, corners_r1}, sumL1Error,
                           atol_coincidence))
    {
      return false;
    }
    printf("Found %zu matching corners in prev.gray0 after coincidence "
           "filtering\n",
           corners_loopback.size());

    // 根据 sumL1Error 将角点排序，保留误差最小的 20 个交点
    static constexpr std::size_t maxCorners{20};
    const std::vector<std::size_t> bestIndices{KeepSmallestElement(sumL1Error,
                                                                   maxCorners)};

    corners.corners_l0.reserve(maxCorners);
    corners.corners_r0.reserve(maxCorners);
    corners.corners_l1.reserve(maxCorners);
    corners.corners_r1.reserve(maxCorners);
    for (const std::size_t bestIndex : bestIndices)
    {
      const cv::Point2i best_corner_l0{
          static_cast<cv::Point2i>(corners_l0[bestIndex])
      };
      const cv::Point2i best_corner_r0{
          static_cast<cv::Point2i>(corners_r0[bestIndex])
      };
      const cv::Point2i best_corner_l1{
          static_cast<cv::Point2i>(corners_l1[bestIndex])
      };
      const cv::Point2i best_corner_r1{
          static_cast<cv::Point2i>(corners_r1[bestIndex])
      };

      corners.corners_l0.push_back(best_corner_l0);
      corners.corners_r0.push_back(best_corner_r0);
      corners.corners_l1.push_back(best_corner_l1);
      corners.corners_r1.push_back(best_corner_r1);

      CornerPair oldCornerPair{frame_index - 1, best_corner_l0, best_corner_r0};
      CornerPair newCornerPair{frame_index, best_corner_l1, best_corner_r1};

      bool found_matching_pair{false};

      if (!first)
      {
        for (auto &corner_track : corner_track_store)
        {
          const CornerPair &corner_pair{corner_track.back()};
          const cv::Point2i &last_left{corner_pair.corner_left};
          const cv::Point2i &last_right{corner_pair.corner_right};
          // 将 [last_left, last_right] 与 [best_corner_l0, best_corner_r0] 进行比较，判断是否同一个角点
          if (last_left == best_corner_l0 && last_right == best_corner_r0)
          {
            corner_track.push_back(newCornerPair);
            found_matching_pair = true;
            break;
          }
        }
      }

      if (!found_matching_pair)
      {
        // 将现有点对加入仓库
        corner_track_store.push_back(std::vector<CornerPair>{
            oldCornerPair,
            newCornerPair,
        });
      }
    }

    return true;
  }

  template <typename PointType>
  void PlotFlow(cv::Mat &flow, std::vector<PointType> const &pts0,
                std::vector<PointType> const &pts1, cv::Size offset0,
                cv::Size offset1, std::string_view label) const
  {
    for (std::size_t index{0}; index < pts0.size(); ++index)
    {
      cv::Point2f pt0{pts0[index]};
      cv::Point2f pt1{pts1[index]};
      pt0.x += offset0.width;
      pt0.y += offset0.height;
      pt1.x += offset1.width;
      pt1.y += offset1.height;
      const cv::Scalar lineColor{this->colors[index % this->nColors]};
      const int lineThickness{2};
      cv::line(flow, pt0, pt1, lineColor, lineThickness);
    }
    const cv::Point2f textPos{10.0f + offset0.width, 30.0f + offset0.height};
    const cv::Scalar textColor{0, 255, 0};
    const int fontFace{cv::FONT_HERSHEY_SIMPLEX};
    const double fontScale{1.0};
    const int textThickness{2};
    cv::putText(flow, label, textPos, fontFace, fontScale, textColor,
                textThickness);
  }

  void SyncCallback(const MsgImage::ConstSharedPtr &cam0_msg,
                    const MsgImage::ConstSharedPtr &cam1_msg,
                    const MsgImu::ConstSharedPtr &imu_msg)
  {
    (void) cam0_msg;
    (void) cam1_msg;
    const std::int64_t timestamp{
        static_cast<std::int64_t>(imu_msg->header.stamp.sec) * 1'000'000'000ll
        + imu_msg->header.stamp.nanosec
    };
    const auto &vec3angularVelocity{imu_msg->angular_velocity};
    const auto &vec3linearAcceleration{imu_msg->linear_acceleration};

    if (first)
    {
      RCLCPP_INFO(this->get_logger(), "SyncCallback timestamp, gyro_x, gyro_y, "
                                      "gyro_z, accel_x, accel_y, accel_z");
    }
    do
    {
      std::stringstream ss;
      ss << "SyncCallback " << timestamp << ", " << std::fixed
         << std::setprecision(18) << vec3angularVelocity.x << ", "
         << vec3angularVelocity.y << ", " << vec3angularVelocity.z << ", "
         << vec3linearAcceleration.x << ", " << vec3linearAcceleration.y << ", "
         << vec3linearAcceleration.z;
      RCLCPP_INFO(this->get_logger(), ss.str().c_str());
    } while (false);

#if 0
    auto image0                   = ConvertImage(cam0_msg);
    auto image1                   = ConvertImage(cam1_msg);
    auto [rectified0, rectified1] = euroc.remap(image0, image1);
    auto [gray0, gray1]           = euroc.grayscale(rectified0, rectified1);

    if (first)
    {
      prev.rectified0    = rectified0;
      prev.rectified1    = rectified1;
      prev.gray0         = gray0;
      prev.gray1         = gray1;
      prev.rotation.w    = 1.0;
      prev.rotation.x    = 0.0;
      prev.rotation.y    = 0.0;
      prev.rotation.z    = 0.0;
      prev.translation.x = 0.0;
      prev.translation.y = 0.0;
      prev.translation.z = 0.0;
      first              = false;
      return;
    }

    cv::Mat top;
    cv::Mat bottom;
    cv::Mat vis;
    cv::hconcat(prev.rectified0, prev.rectified1, top); // 上排：前一帧左右目
    cv::hconcat(rectified0, rectified1, bottom);        // 下排：当前帧左右目
    cv::vconcat(top, bottom, vis);                      // 纵向拼接成 2x2

    // 寻找角点
    Corners corners;
    if (FindCorners(gray0, gray1, corners, frame_counter))
    {
      printf("FindCorners success: %ld\n", corners.corners_l0.size());

      auto rq0{euroc.Solve(corners.corners_l0, corners.corners_l1)};
      auto rq1{euroc.Solve(corners.corners_r0, corners.corners_r1)};

      // printf("Left Camera:\n\tTranslation=(%.2f, %.2f, %.2f)\n"
      //        "\tRotation=(%.2f, %.2f, %.2f, %.2f)\n",
      //        rq0.translation.x, rq0.translation.y, rq0.translation.z,
      //        rq0.rotation.w, rq0.rotation.x, rq0.rotation.y, rq0.rotation.z);
      // printf("Right Camera:\n\tTranslation=(%.2f, %.2f, %.2f)\n"
      //        "\tRotation=(%.2f, %.2f, %.2f, %.2f)\n",
      //        rq1.translation.x, rq1.translation.y, rq1.translation.z,
      //        rq1.rotation.w, rq1.rotation.x, rq1.rotation.y, rq1.rotation.z);

      // 写入文件
      file << cam0_msg->header.stamp.sec << '.'
           << cam0_msg->header.stamp.nanosec << "," << rq0.rotation.w << ","
           << rq0.rotation.x << "," << rq0.rotation.y << "," << rq0.rotation.z
           << "," << rq0.translation.x << "," << rq0.translation.y << ","
           << rq0.translation.z << "," << rq1.rotation.w << ","
           << rq1.rotation.x << "," << rq1.rotation.y << "," << rq1.rotation.z
           << "," << rq1.translation.x << "," << rq1.translation.y << ","
           << rq1.translation.z << std::endl;

      // TODO 估计上一帧到当前帧的相对刚体变换

      // 绘制光流
      const cv::Size flowSize{vis.size()};
      cv::Mat flow{cv::Mat::zeros(flowSize, rectified0.type())};
      const cv::Size imageSize{rectified0.size()};
      PlotFlow(flow, corners.corners_l0, corners.corners_r0, cv::Size{0, 0},
               cv::Size{imageSize.width, 0}, "Previous Frame Left");
      PlotFlow(flow, corners.corners_r0, corners.corners_r1,
               cv::Size{imageSize.width, 0}, imageSize, "Previous Frame Right");
      PlotFlow(flow, corners.corners_r1, corners.corners_l1, imageSize,
               cv::Size{0, imageSize.height}, "Current Frame Right");
      PlotFlow(flow, corners.corners_l1, corners.corners_l0,
               cv::Size{0, imageSize.height}, cv::Size{0, 0},
               "Current Frame Left");

      cv::add(flow, vis, vis);

      // 更新上一帧的旋转和平移
      prev.rotation.w    = (rq0.rotation.w + rq1.rotation.w) * 0.5;
      prev.rotation.x    = (rq0.rotation.x + rq1.rotation.x) * 0.5;
      prev.rotation.y    = (rq0.rotation.y + rq1.rotation.y) * 0.5;
      prev.rotation.z    = (rq0.rotation.z + rq1.rotation.z) * 0.5;
      prev.translation.x = (rq0.translation.x + rq1.translation.x) * 0.5;
      prev.translation.y = (rq0.translation.y + rq1.translation.y) * 0.5;
      prev.translation.z = (rq0.translation.z + rq1.translation.z) * 0.5;
    }
    else
    {
      printf("FindCorners failed\n");
    }

    cv::imshow("VIO", vis);

    // 将上一帧更新为当前帧
    prev.rectified0 = rectified0;
    prev.rectified1 = rectified1;
    prev.gray0      = gray0;
    prev.gray1      = gray1;
    ++frame_counter;
#endif
  }

public:
  VirsualInertialOdemetry(const char *cam0_topic, const char *cam1_topic,
                          const char *imu_topic) :
    Node("VIO"), file{PATH_CSV_FILE, std::ios::out | std::ios::trunc}
  {
    rclcpp::QoS qos(10);
    cam0_sub.subscribe(this, cam0_topic, qos.get_rmw_qos_profile());
    cam1_sub.subscribe(this, cam1_topic, qos.get_rmw_qos_profile());
    imu_sub.subscribe(this, imu_topic, qos.get_rmw_qos_profile());

    uint32_t queue_size{10};
    sync = std::make_shared<Synchronizer_t>(ApproximateTime_t(queue_size),
                                            cam0_sub, cam1_sub, imu_sub);

    sync->setAgePenalty(0.50);

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;

    sync->registerCallback(std::bind(&VirsualInertialOdemetry::SyncCallback,
                                     this, _1, _2, _3));

#if 0
    // 生成随机颜色
    cv::RNG rng;
    for (std::size_t i = 0; i < nColors; ++i)
    {
      colors.push_back(cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                  rng.uniform(0, 256)));
    }

    file << "timestamp[ns],left_rw,left_rx,left_ry,left_rz,left_tx,"
            "left_ty,left_tz,right_rw,right_rx,right_ry,right_rz,"
            "right_tx,right_ty,right_tz"
         << std::endl;
#endif
  }
};

int main(int argc, char **argv)
{
  using VIO = VirsualInertialOdemetry;
  printf("Node 'euroc_vio' started\n");
  rclcpp::init(argc, argv);
  const char *cam0_topic{"/cam0/image_raw"};
  const char *cam1_topic{"/cam1/image_raw"};
  const char *imu_topic{"/imu0"};
  auto node = std::make_shared<VIO>(cam0_topic, cam1_topic, imu_topic);
#if 0
  cv::namedWindow("VIO", cv::WINDOW_NORMAL);
  cv::moveWindow("VIO", 0, 0);
  cv::resizeWindow("VIO", cv::Size{1280, 720});
#endif
  while (true)
  {
    rclcpp::spin_some(node);
#if 0
    const auto key{cv::waitKey(10) & 0xFF};
    if (key == 27)
    {
      break;
    }
#endif
  }
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
