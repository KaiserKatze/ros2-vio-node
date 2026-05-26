#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include "Path.hpp"
#include "Room.hpp"

using namespace std::chrono_literals;

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "euroc_vio/main.h"

#define PUBLISH_STYLE_ALL_AT_ONCE 0
#define PUBLISH_STYLE_STEP_BY_STEP 1

/**
 * @brief 从指定文件中，读取角位移向量和单位化平移向量，通过一阶积分计算姿态、轨迹
 */
struct EstimationLoader : public rclcpp::Node
{
  using Point3     = Eigen::Vector<float, 3>;
  using Attitude   = Eigen::Matrix<float, 3, 3>;
  using Quaternion = Eigen::Quaternion<float>;

private:
  const std::filesystem::path path_home{
      std::getenv("HOME"),
  };
  const std::filesystem::path path_estimation_csv{
      path_home / "vio_ws" / "estimated_motion.csv",
      // path_home / "vio_ws" / "fake" / "data_camera.csv",
      // path_home / "vio_ws" / "fake" / "data_world.csv",
      // path_home / "vio_ws" / "estimated_motion_Tangent.csv"
      // path_home / "vio_ws" / "estimated_motion_LookAtCenter.csv",
  };

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_{
      create_publisher<nav_msgs::msg::Path>("/path_fast_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_{
      create_publisher<nav_msgs::msg::Path>("/pose_fast_est", rclcpp::QoS{10}),
  };

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
      throw std::runtime_error{std::format("Failed to parse float: '{}'.", sv)};
    }
    return result;
  }

  Room<float> room_{};
  Path<float> path_{};

  std::pair<Point3, Attitude> GetPose(float time) const
  {
    constexpr auto mode{Path<float>::OrientationMode::LookAtCenter};
    return path_.GetPose(room_, time, mode);
  }

  template <typename value_type>
  geometry_msgs::msg::PoseStamped
  PushPose(std::int64_t timestamp,
           const Eigen::Quaternion<value_type> &attitude,
           const Eigen::Vector<value_type, 3> &position)
  {
    geometry_msgs::msg::PoseStamped msg_pose;
    msg_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_pose.header.stamp    = rclcpp::Time{timestamp};

    msg_pose.pose.position.x = position.x();
    msg_pose.pose.position.y = position.y();
    msg_pose.pose.position.z = position.z();

    msg_pose.pose.orientation.w = attitude.w();
    msg_pose.pose.orientation.x = attitude.x();
    msg_pose.pose.orientation.y = attitude.y();
    msg_pose.pose.orientation.z = attitude.z();

    msg_path_.poses.push_back(msg_pose);
    return msg_pose;
  }

  void PublishPath()
  {
    if (msg_path_.poses.empty())
    {
      return;
    }
    msg_path_.header.stamp    = msg_path_.poses.back().header.stamp;
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
    publisher_path_->publish(msg_path_);
  }

  void PublishPose(const geometry_msgs::msg::PoseStamped &msg_pose)
  {
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_->publish(msg_path_pose);
  }

public:
  EstimationLoader() : Node("StereoSlam1")
  {
    std::print(stderr, "EstimationLoader ready ...\n");
  }

  void Start()
  {
    Point3 position{Point3::Zero()};
    Quaternion attitude{Quaternion::Identity()};

    // 位姿初始化
    std::tie(position, attitude) = GetPose(0.0f);

    // std::print(stderr,
    //            "[DEBUG] 初始位姿 {{ "
    //            "timestamp={:2.2f}, "
    //            "attitude=[{:.4f}, {:.4f}, {:.4f}, {:.4f}], "
    //            "position=[{:.4f}, {:.4f}, {:.4f}]"
    //            " }}\n",
    //            0.0,                                                    //
    //            attitude.w(), attitude.x(), attitude.y(), attitude.z(), //
    //            position.x(), position.y(), position.z());

    std::ifstream file(path_estimation_csv);
    std::string line;
#if (PUBLISH_STYLE_ALL_AT_ONCE)
    size_t line_num{0};
#endif

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
#if (PUBLISH_STYLE_ALL_AT_ONCE)
      ++line_num;
#endif
      // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
      if (!rclcpp::ok())
      {
        break;
      }

      std::stringstream ss(line);

      // 读取时间戳
      const std::int64_t timestamp{
          get_item_as_int64(ss), // in nanoseconds
      };
      // 读取旋转角度
      const float wxt{get_item_as_float(ss)};
      const float wyt{get_item_as_float(ss)};
      const float wzt{get_item_as_float(ss)};
      // 读取位移方向
      const float tx{get_item_as_float(ss)};
      const float ty{get_item_as_float(ss)};
      const float tz{get_item_as_float(ss)};

      const Point3 delta_position{tx, ty, tz};
      const Point3 delta_rotation_vector{wxt, wyt, wzt};
      const Eigen::AngleAxisf delta_rotation_angle_axis{
          delta_rotation_vector.norm(),
          delta_rotation_vector.normalized(),
      };
      const Quaternion delta_rotation{delta_rotation_angle_axis};

      // 如果数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      // 那么应该使用以下状态更新方程
      position = position + attitude * delta_position;
      attitude = (attitude * delta_rotation).normalized();

      // 如果数据集 path_estimation_csv 提供的旋转向量、平移向量是在世界坐标系下的表示
      // 那么应该使用以下状态更新方程
      // position = position + delta_position;
      // attitude = (delta_rotation * attitude).normalized();

#if (PUBLISH_STYLE_STEP_BY_STEP)
      auto msg_pose{PushPose(timestamp, attitude, position)};
      PublishPath();
      PublishPose(msg_pose);
      // 保持 20 Hz 的话题发布频率
      std::this_thread::sleep_for(50ms);
#else
      PushPose(timestamp, attitude, position);
#endif
    } // end while

#if (PUBLISH_STYLE_ALL_AT_ONCE)
    for (size_t counter{0}; rclcpp::ok(); counter = (counter + 1) % line_num)
    {
      // std::print(stderr, "[DEBUG] 正在循环发布已有数据 (总计 {} 帧数据)!\n", line_num);
      PublishPath();
      PublishPose(msg_path_.poses[counter]);

      std::this_thread::sleep_for(50ms);
    } // end for
#endif
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
