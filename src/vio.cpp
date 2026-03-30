// clang-format on

#include <algorithm>
#include <chrono> // time module
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
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
// #include <opencv2/xfeatures2d.hpp>

#include "euroc_vio/QuEst.hpp"
#include "euroc_vio/main.h"

using namespace sensor_msgs;
using namespace message_filters;
using namespace std::chrono_literals;

/*****************************
 * ROS2 Node
 *****************************/

using MsgImage = sensor_msgs::msg::Image;
using MsgImu   = euroc_vio::msg::Imu;
using ApproximateTime_t
    = message_filters::sync_policies::ApproximateTime<MsgImage, MsgImage,
                                                      MsgImu>;
using Synchronizer_t = message_filters::Synchronizer<ApproximateTime_t>;

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
    const size_t numPts{pts0.size()};
    if (numPts != pts1.size())
    {
      throw std::runtime_error{"pts0 and pts1 must have the same size"};
    }
    Eigen::Matrix3Xd M0(3, numPts);
    Eigen::Matrix3Xd M1(3, numPts);
    for (size_t i{0}; i < numPts; ++i)
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

class VirsualInertialOdemetry : public rclcpp::Node
{
private:
  std::shared_ptr<Synchronizer_t> sync;
  message_filters::Subscriber<MsgImage> cam0_sub;
  message_filters::Subscriber<MsgImage> cam1_sub;
  message_filters::Subscriber<MsgImu> imu_sub;

  bool first{true};

  const EuRoC euroc{};

  void SyncCallback(const MsgImage::ConstSharedPtr &cam0_msg,
                    const MsgImage::ConstSharedPtr &cam1_msg,
                    const MsgImu::ConstSharedPtr &imu_msg)
  {
    (void) cam0_msg;
    (void) cam1_msg;
    const std::int64_t timestamp{
        static_cast<std::int64_t>(imu_msg->header.stamp.sec) * 1'000'000'000ll
        + imu_msg->header.stamp.nanosec};
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
  }

public:
  VirsualInertialOdemetry(const char *cam0_topic, const char *cam1_topic,
                          const char *imu_topic)
      : Node("VIO")
  {
    rclcpp::QoS qos(10);
    const auto &profile{qos.get_rmw_qos_profile()};
    cam0_sub.subscribe(this, cam0_topic, profile);
    cam1_sub.subscribe(this, cam1_topic, profile);
    imu_sub.subscribe(this, imu_topic, profile);

    uint32_t queue_size{10};
    sync = std::make_shared<Synchronizer_t>(ApproximateTime_t(queue_size),
                                            cam0_sub, cam1_sub, imu_sub);

    sync->setAgePenalty(0.50);

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;

    sync->registerCallback(
        std::bind(&VirsualInertialOdemetry::SyncCallback, this, _1, _2, _3));
  }
};

int main(int argc, char **argv)
{
  printf("Node 'euroc_vio' started\n");
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VirsualInertialOdemetry>(
      "/cam0/image_raw", "/cam1/image_raw", "/imu9");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
