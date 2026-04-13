#ifndef GROUNDTRUTHPUBLISHER_HPP
#define GROUNDTRUTHPUBLISHER_HPP
// clang-format on

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/imuworker.hpp"

class GroundTruthPublisher
{
public:
  GroundTruthPublisher(rclcpp::Node *node_ptr,
                       const char *input_groundtruth_topic,
                       const char *output_groundtruth_topic);

private:
  template <typename T>
  void SubscriberCallback(const typename T::ConstSharedPtr msg);

  rclcpp::Node *node_ptr_;
  rclcpp::Subscription<MsgGroundTruth>::SharedPtr subscriber_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_;
  MsgPath msg_path_;
  MsgPose msg_pose_;
};

// 特化情况 : 处理 PoseStamped
template <>
void GroundTruthPublisher::SubscriberCallback<geometry_msgs::msg::PoseStamped>(
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
{
  msg_path_.header.stamp = msg->header.stamp;
  msg_path_.poses.push_back(*msg);
  publisher_->publish(msg_path_);
}

// 特化情况 : 处理 TransformStamped
template <>
void GroundTruthPublisher::SubscriberCallback<
    geometry_msgs::msg::TransformStamped>(
    const geometry_msgs::msg::TransformStamped::ConstSharedPtr msg)
{
  const auto stamp{msg->header.stamp};
  msg_path_.header.stamp = stamp;

  msg_pose_.header.stamp       = stamp;
  msg_pose_.pose.position.x    = msg->transform.translation.x;
  msg_pose_.pose.position.y    = msg->transform.translation.y;
  msg_pose_.pose.position.z    = msg->transform.translation.z;
  msg_pose_.pose.orientation.w = msg->transform.rotation.w;
  msg_pose_.pose.orientation.x = msg->transform.rotation.x;
  msg_pose_.pose.orientation.y = msg->transform.rotation.y;
  msg_pose_.pose.orientation.z = msg->transform.rotation.z;

  msg_path_.poses.push_back(msg_pose_);
  publisher_->publish(msg_path_);
}

GroundTruthPublisher::GroundTruthPublisher(rclcpp::Node *node_ptr,
                                           const char *input_groundtruth_topic,
                                           const char *output_groundtruth_topic)
    : node_ptr_{node_ptr}
{
  using std::placeholders::_1;
  const rclcpp::QoS qos(10);

  subscriber_ = node_ptr_->create_subscription<MsgGroundTruth>(
      input_groundtruth_topic, qos,
      std::bind(&GroundTruthPublisher::SubscriberCallback<MsgGroundTruth>, this,
                _1));

  publisher_
      = node_ptr_->create_publisher<MsgPath>(output_groundtruth_topic, qos);

  msg_pose_.header.frame_id = DEFAULT_FRAME_ID;
  msg_path_.header.frame_id = DEFAULT_FRAME_ID;
}

#endif /* GROUNDTRUTHPUBLISHER_HPP */
