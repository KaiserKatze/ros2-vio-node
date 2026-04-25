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
#include "euroc_vio/EuRoC.hpp"

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
