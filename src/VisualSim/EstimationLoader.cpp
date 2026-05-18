#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include "MeshPlot.hpp"
#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"

using namespace std::chrono_literals;

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "euroc_vio/main.h"

struct EstimationLoader : public rclcpp::Node
{
private:
  const std::filesystem::path path_home{
      std::getenv("HOME"),
  };
  const std::filesystem::path path_estimation_csv{
      path_home / "vio_ws" / "estimated_quats.csv",
  };

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_{
      create_publisher<nav_msgs::msg::Path>("/path_pure_quat", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_;

  static std::string_view trim(std::string_view str)
  {
    const auto first = str.find_first_not_of(" \t\n\r\v\f");
    if (first == std::string_view::npos)
    {
      return {};
    }

    const auto last = str.find_last_not_of(" \t\n\r\v\f");
    return str.substr(first, last - first + 1);
  }

  static std::int64_t get_item_as_int64(std::stringstream &ss)
  {
    std::string item;
    std::getline(ss, item, ',');
    auto sv{trim(item)};
    std::int64_t result{0};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse int64: " + std::string(sv)};
    }
    return result;
  }

  static float get_item_as_float(std::stringstream &ss)
  {
    std::string item;
    std::getline(ss, item, ',');
    auto sv{trim(item)};
    float result{0.0f};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse float: " + std::string(sv)};
    }
    return result;
  }

  Room<float> room_{};
  Path<float> path_{};

  template <typename value_type>
  void PublishPath(const Eigen::Quaternion<value_type> &attitude,
                   const Eigen::Vector<value_type, 3> &position)
  {
    rclcpp::Time now{this->get_clock()->now()};

    msg_path_.header.stamp = now;

    geometry_msgs::msg::PoseStamped msg_pose;
    msg_pose.header.stamp = now;

    msg_pose.pose.position.x = position.x();
    msg_pose.pose.position.y = position.y();
    msg_pose.pose.position.z = position.z();

    msg_pose.pose.orientation.w = attitude.w();
    msg_pose.pose.orientation.x = attitude.x();
    msg_pose.pose.orientation.y = attitude.y();
    msg_pose.pose.orientation.z = attitude.z();

    msg_path_.poses.push_back(msg_pose);
    publisher_path_->publish(msg_path_);

    std::print(stderr, "[INFO] 成功在话题 {} 发布轨迹消息\n",
               publisher_path_->get_topic_name());
  }

public:
  EstimationLoader() : Node("StereoSlam1")
  {
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
    std::print(stderr, "EstimationLoader ready ...\n");
  }

  void Start()
  {
    std::ifstream file(path_estimation_csv);
    std::string line;
    size_t line_num{0};

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::print(stderr, "[DEBUG] 从朝向估计 CSV 文件中读取第 {} 行数据 ...\n",
                 ++line_num);
      // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
      if (!rclcpp::ok())
      {
        break;
      }

      std::stringstream ss(line);

      const float timestamp{
          static_cast<float>(get_item_as_int64(ss) * 1e-9), // in nanoseconds
      };
      const auto qw{
          get_item_as_float(ss),
      };
      const auto qx{
          get_item_as_float(ss),
      };
      const auto qy{
          get_item_as_float(ss),
      };
      const auto qz{
          get_item_as_float(ss),
      };
      // publish
      using Point3     = Eigen::Vector<float, 3>;
      using Quaternion = Eigen::Quaternion<float>;
      Point3 true_position{Point3::Zero()};
      Quaternion true_attitude{Quaternion::Identity()};
      std::tie(true_position, true_attitude) = path_.GetPose(
          room_, timestamp, Path<float>::OrientationMode::LookAtCenter);
      Quaternion est_attitude{qw, qx, qy, qz};

      PublishPath(est_attitude, true_position);
      std::this_thread::sleep_for(50ms);
    }
  }
};

int main(int argc, char *argv[])
{
  // 初始化 ROS 2
  rclcpp::init(argc, argv);

  try
  {
    EstimationLoader{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }

  // 关闭 ROS 2 实例
  rclcpp::shutdown();
  return 0;
}
