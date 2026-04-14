#ifndef ABSTRACTPUBLISHER_HPP
#define ABSTRACTPUBLISHER_HPP

#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>

class AbstractPublisher
{
protected:
  void UpdateBoundary(const Eigen::Vector3d &&pos)
  {
    boundary_.x() = std::max(boundary_.x(), std::abs(pos.x()));
    boundary_.y() = std::max(boundary_.y(), std::abs(pos.y()));
    boundary_.z() = std::max(boundary_.z(), std::abs(pos.z()));
  }

  void UpdateBoundary(const geometry_msgs::msg::PoseStamped &msg)
  {
    const auto &msg_pos{msg.pose.position};
    Eigen::Vector3d vec_pos{msg_pos.x, msg_pos.y, msg_pos.z};
    UpdateBoundary(std::move(vec_pos));
  }

private:
  void PrintBoundary() const
  {
    RCLCPP_INFO(rclcpp::get_logger("AbstractPublisher"),
                "Current boundary: [x: %.3f, y: %.3f, z: %.3f]", boundary_.x(),
                boundary_.y(), boundary_.z());
  }

public:
  ~AbstractPublisher()
  {
    PrintBoundary();
  }

private:
  Eigen::Vector3d boundary_{Eigen::Vector3d::Zero()};
};

#endif /* ABSTRACTPUBLISHER_HPP */
