
// 基于松耦合的线性卡尔曼滤波，融合单目视觉与 IMU 数据

#include <Eigen/Dense>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "LinearKalmanFilter.hpp"
#include "euroc_vio/main.h"

struct FuseKalman : public rclcpp::Node
{
  const std::string imu_topic_name_{"/imu0"};
  // 单目视觉估计的姿态 (位置向量和朝向四元数)
  const std::string cam_topic_name_{"/path_fast_est"};
  // 融合单目视觉和 IMU 数据以后输出的估计轨迹
  const std::string est_path_topic_name_{"/path_fuse_est"};
  // 融合单目视觉和 IMU 数据以后输出的估计姿态
  const std::string est_pose_topic_name_{"/pose_fuse_est"};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscriber_imu_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscriber_pose_cam_;

  nav_msgs::msg::Path path_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_est_;
  nav_msgs::msg::Path pose_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_est_;

  rclcpp::TimerBase::SharedPtr timer_;

  FuseKalman() : Node("FuseKalman")
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);
    subscriber_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_name_, qos,
        std::bind(&FuseKalman::SubscriberImuCallback, this, _1));
    subscriber_pose_cam_ = create_subscription<nav_msgs::msg::Path>(
        cam_topic_name_, qos,
        std::bind(&FuseKalman::SubscriberCamCallback, this, _1));
    timer_ = create_wall_timer(std::chrono::seconds(1), // 1 Hz 发布
                               std::bind(&FuseKalman::PublishTrajectory, this));

    path_msg_.header.frame_id = DEFAULT_FRAME_ID;
    pose_msg_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void SubscriberImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    (void) msg;
    // TODO
  }

  void SubscriberCamCallback(
      const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
  {
    (void) msg;
    // TODO
  }

  // IMU 采样率是 200 Hz
  // 单目相机采样率 20 Hz，单目视觉估计姿态的输出数据率大约也是 20 Hz
  // 那么应该怎么融合不同频率的两种传感器的数据呢？

  void PublishTrajectory()
  {
    const auto timestamp{now()};
    path_msg_.header.stamp = timestamp;
    publisher_path_est_->publish(path_msg_);

    static size_t index{0};
    const auto &pose{path_msg_.poses[index]};
    pose_msg_.header.stamp = pose.header.stamp;
    pose_msg_.poses.push_back(pose);
    index = (index + 1) % path_msg_.poses.size();
  }

  geometry_msgs::msg::PoseStamped
  CreateFusedPose(std::int64_t timestamp, const Eigen::Vector3d &position,
                  const Eigen::Quaterniond &attitude)
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id    = DEFAULT_FRAME_ID;
    msg.header.stamp       = rclcpp::Time{timestamp};
    msg.pose.position.x    = position.x();
    msg.pose.position.y    = position.y();
    msg.pose.position.z    = position.z();
    msg.pose.orientation.w = attitude.w();
    msg.pose.orientation.x = attitude.x();
    msg.pose.orientation.y = attitude.y();
    msg.pose.orientation.z = attitude.z();
    path_msg_.poses.push_back(msg);
    return msg;
  }
};
