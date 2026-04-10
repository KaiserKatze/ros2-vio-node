// clang-format on

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/groundtruthpublisher.hpp"
#include "euroc_vio/imupublisher.hpp"
#include "euroc_vio/imuworker.hpp"

using namespace std::chrono_literals;

/*****************************
 * ROS2 Node
 *****************************/
class ImuNode : public rclcpp::Node
{
public:
  ImuNode(const char *input_imu_topic, const char *input_groundtruth_topic,
          const char *output_imu_topic, const char *output_groundtruth_topic)
      : Node("IMU")
  {
    this->declare_parameter("estimator", "rk4");
    this->declare_parameter("use_filter", true);
    this->declare_parameter("initial_position_x", 0.0);
    this->declare_parameter("initial_position_y", 0.0);
    this->declare_parameter("initial_position_z", 0.0);
    const std::string estimator_str{
        this->get_parameter("estimator").as_string()};
    // const bool use_filter{this->get_parameter("use_filter").as_bool()};
    // const double init_px{this->get_parameter("initial_position_x").as_double()};
    // const double init_py{this->get_parameter("initial_position_y").as_double()};
    // const double init_pz{this->get_parameter("initial_position_z").as_double()};

    EstimatorType estimator;
    if (estimator_str == "rk4")
    {
      estimator = EstimatorType::RK4;
    }
    else if (estimator_str == "mahony")
    {
      estimator = EstimatorType::MAHONY;
    }
    else if (estimator_str == "madgwick")
    {
      estimator = EstimatorType::MADGWICK;
    }
    else
    {
      throw std::invalid_argument("Invalid estimator type: " + estimator_str);
    }
    pub_imu = std::make_unique<ImuPathPublisher>(
        this, input_imu_topic, output_imu_topic, ImuWorker{estimator});
    pub_gt = std::make_unique<GroundTruthPublisher>(
        this, input_groundtruth_topic, output_groundtruth_topic);
  }

private:
  std::unique_ptr<GroundTruthPublisher> pub_gt;
  std::unique_ptr<ImuPathPublisher> pub_imu;
};

int main(int argc, char **argv)
{
  std::cout << "Node 'euroc_imu' started\n";
  rclcpp::init(argc, argv);

  const char *input_imu_topic{"/imu0"};
  const char *input_groundtruth_topic{"/vicon/firefly_sbx/firefly_sbx"};
  const char *output_imu_topic{"/path_imu"};
  const char *output_groundtruth_topic{"/path_groundtruth"};

  const auto node{
      std::make_shared<ImuNode>(input_imu_topic, input_groundtruth_topic,
                                output_imu_topic, output_groundtruth_topic)};
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
