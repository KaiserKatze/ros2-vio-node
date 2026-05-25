#include <cstdint>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/main.h"

struct Datum
{
  std::int64_t timestamp;
  // px, py, pz, qw, qx, qy, qz
  std::array<double, 7> value;

  static std::vector<Datum> ReadCsv(const std::string &filename,
                                    bool skip_header = true, char delim = ',')
  {
    std::vector<Datum> data;
    std::ifstream file(filename);
    std::string line;
    if (skip_header)
    {
      std::getline(file, line); // 跳过表头
    }
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      Datum datum;
      datum.timestamp = AbstractLoader::get_item_as_int64(ss, delim);
      for (size_t i = 0; i < datum.value.size(); ++i)
      {
        datum.value[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      data.push_back(datum);
    }
    return data;
  }
};

class TrajectoryPublisher : public rclcpp::Node
{
public:
  // 构造函数：传入文件路径和发布话题名（可选）
  explicit TrajectoryPublisher() : Node("TrajectoryPublisher")
  {
    // 目标数据文件
    this->declare_parameter("csv_file", "");
    const std::string csv_file{
        this->get_parameter("csv_file").as_string(),
    };

    // 话题名称
    this->declare_parameter("topic_name", "/path_traj");
    const std::string topic_name{
        this->get_parameter("topic_name").as_string(),
    };

    // 是否存在需要跳过的表头
    this->declare_parameter("skip_header", true);
    const bool skip_header{
        this->get_parameter("skip_header").as_bool(),
    };

    // 间隔符
    this->declare_parameter("delim", ",");
    const char delim{
        this->get_parameter("delim").as_string()[0],
    };

    // 1. 获取文件路径并解析
    raw_data_ = Datum::ReadCsv(csv_file, skip_header, delim);
    if (raw_data_.empty())
    {
      throw std::runtime_error("CSV 文件为空或无有效数据");
    }

    // 2. 构建完整的 Path 消息（将所有位姿一次性加入）
    path_msg_.header.frame_id = DEFAULT_FRAME_ID;
    for (const auto &d : raw_data_)
    {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp       = rclcpp::Time{d.timestamp};
      pose.header.frame_id    = DEFAULT_FRAME_ID;
      pose.pose.position.x    = d.value[0];
      pose.pose.position.y    = d.value[1];
      pose.pose.position.z    = d.value[2];
      pose.pose.orientation.w = d.value[3];
      pose.pose.orientation.x = d.value[4];
      pose.pose.orientation.y = d.value[5];
      pose.pose.orientation.z = d.value[6];
      path_msg_.poses.push_back(pose);
    }

    // 3. 创建发布者
    publisher_path_
        = this->create_publisher<nav_msgs::msg::Path>(topic_name, 10);

    // 4. 使用定时器周期性发布完整轨迹（确保新订阅者也能收到）
    timer_ = this->create_wall_timer(
        std::chrono::seconds(1), // 1 Hz 发布
        std::bind(&TrajectoryPublisher::publish_trajectory, this));
  }

private:
  void publish_trajectory()
  {
    const auto now{this->now()};
    path_msg_.header.stamp = now;
    publisher_path_->publish(path_msg_);

    static size_t index{0};
    const Datum &d{raw_data_[index]};
    nav_msgs::msg::Path path_pose_msg;
    path_pose_msg.header.frame_id = DEFAULT_FRAME_ID;
    path_pose_msg.header.stamp    = now;
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id    = DEFAULT_FRAME_ID;
    pose.pose.position.x    = d.value[0];
    pose.pose.position.y    = d.value[1];
    pose.pose.position.z    = d.value[2];
    pose.pose.orientation.w = d.value[3];
    pose.pose.orientation.x = d.value[4];
    pose.pose.orientation.y = d.value[5];
    pose.pose.orientation.z = d.value[6];
    path_pose_msg.poses.push_back(pose);
    index = (index + 1) % raw_data_.size();
  }

  std::vector<Datum> raw_data_;
  nav_msgs::msg::Path path_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    auto node = std::make_shared<TrajectoryPublisher>();
    rclcpp::spin(node); // 保持节点运行，定时发布
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "[ERROR] {}", ex.what());
  }

  rclcpp::shutdown();
  return 0;
}
