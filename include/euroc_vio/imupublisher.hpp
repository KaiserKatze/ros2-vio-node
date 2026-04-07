#ifndef IMUPUBLISHER_HPP
#define IMUPUBLISHER_HPP
// clang-format on

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/imuworker.hpp"

class ImuPathPublisher
{
public:
  ImuPathPublisher(rclcpp::Node *node_ptr, const char *input_imu_topic,
                   const char *output_imu_topic, ImuWorker &&worker)
      : node_ptr_{node_ptr}, imu_worker_{std::move(worker)}
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);
    subscriber_ = node_ptr_->create_subscription<MsgImu>(
        input_imu_topic, qos,
        std::bind(&ImuPathPublisher::SubscriberCallback, this, _1));
    publisher_ = node_ptr_->create_publisher<MsgPath>(output_imu_topic, qos);
    // 设置路径消息的坐标系
    this->msg_path_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void HandlePose(const MsgPose &msg)
  {
    this->msg_path_.header.stamp = msg.header.stamp;
    this->msg_path_.poses.push_back(msg);
    this->publisher_->publish(this->msg_path_);
  }

private:
  void PrintAverageSampleRate(const MsgImu::ConstSharedPtr &msg)
  {
    static size_t msg_counter{0};
    static double first_timestamp{0.0};
    const double msg_timestamp{
        static_cast<rclcpp::Time>(msg->header.stamp).seconds()};
    if (msg_counter == 0)
    {
      first_timestamp = msg_timestamp;
    }
    else
    {
      const double elapsed_time{msg_timestamp - first_timestamp};
      const double average_sample_rate{msg_counter / elapsed_time};
      RCLCPP_INFO(node_ptr_->get_logger(), "Average Sample Rate (IMU): %.1f Hz",
                  average_sample_rate);
    }
    ++msg_counter;
  }

  void SubscriberCallback(const MsgImu::ConstSharedPtr &msg)
  {
    PrintAverageSampleRate(msg);
    this->imu_worker_.Work(msg, this);
  }

  rclcpp::Node *node_ptr_;
  rclcpp::Subscription<MsgImu>::SharedPtr subscriber_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_;
  MsgPath msg_path_;
  ImuWorker imu_worker_;
};

#endif /* IMUPUBLISHER_HPP */
