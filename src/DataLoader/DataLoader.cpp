#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>

#include "euroc_vio/AbstractLoader.hpp"

static constexpr char DEFAULT_FRAME_ID[]{"map"};

namespace FastVIO
{

/**
 * @brief CSV 位姿数据的物理结构映射。
 */
struct DatumTrajectory
{
  std::int64_t timestamp;
  // px, py, pz, qw, qx, qy, qz
  std::array<double, 7> value;

  /**
   * @brief 从 CSV 文件中读取时间戳与位置、姿态四元数。
   * @param filename CSV 文件的绝对路径。
   * @param skip_header 是否跳过第一行头部注释。
   * @param delim 列分隔字符。
   * @return std::vector<DatumTrajectory> 读取出的数据集。
   */
  static std::vector<DatumTrajectory> ReadCsv(const std::string &filename,
                                              bool skip_header = true,
                                              char delim       = ',')
  {
    std::vector<DatumTrajectory> data;
    std::ifstream file{filename};
    if (!file.is_open())
    {
      throw std::runtime_error{std::format("Cannot open requested CSV file: {}",
                                           filename)};
    }
    std::string line;
    if (skip_header)
    {
      std::getline(file, line);
    }
    while (std::getline(file, line))
    {
      if (line.empty() || line[0] == '#')
      {
        continue;
      }
      std::stringstream ss(line);
      DatumTrajectory datum;
      datum.timestamp = AbstractLoader::get_item_as_int64(ss, delim);
      for (std::size_t i = 0; i < datum.value.size(); ++i)
      {
        datum.value[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      data.push_back(datum);
    }
    return data;
  }
};

/**
 * @brief 将指定路径的位姿数据加载并在 ROS 2 框架下发布为 Path 的节点类。
 */
class TrajectoryPublisher : public rclcpp::Node
{
public:
  /**
   * @brief 构造函数。
   */
  explicit TrajectoryPublisher() : Node("TrajectoryPublisher")
  {
    this->declare_parameter("csv_file", "");
    csv_file_ = this->get_parameter("csv_file").as_string();

    if (csv_file_.empty())
    {
      throw std::runtime_error{"Required configuration path cannot be empty."};
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(csv_file_, ec))
    {
      throw std::runtime_error{
          std::format("Required configuration path '{}' not found.", csv_file_)
      };
    }

    this->declare_parameter("topic_name", "/trajectory_est");
    topic_name_ = this->get_parameter("topic_name").as_string();

    this->declare_parameter("skip_header", true);
    skip_header_ = this->get_parameter("skip_header").as_bool();

    this->declare_parameter("delim", ",");
    delim_ = this->get_parameter("delim").as_string()[0];

    const rclcpp::QoS qos{10};

    publisher_path_ = this->create_publisher<nav_msgs::msg::Path>(
        std::format("{}/path", topic_name_), qos
    );
    publisher_pose_ = this->create_publisher<nav_msgs::msg::Path>(
        std::format("{}/pose", topic_name_), qos
    );

    LoadCsv();

    // 开启周期定时器发布轨迹话题与实时重试
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&TrajectoryPublisher::publish_trajectory, this)
    );
  }

private:
  /**
   * @brief 尝试加载 CSV 数据文件，若存在且解析成功，则完成 Path 格式组装。
   */
  void LoadCsv()
  {
    auto raw_data{DatumTrajectory::ReadCsv(csv_file_, skip_header_, delim_)};
    if (raw_data.empty())
    {
      throw std::runtime_error{"CSV 文件为空或无有效数据"};
    }
    path_msg_.header.frame_id = DEFAULT_FRAME_ID;
    for (const auto &d : raw_data)
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
    std::print(stderr, "[INFO] Loaded {} poses from file: {}\n",
               raw_data.size(), csv_file_);
  }

  /**
   * @brief 定时发布轨迹和最新位置位姿的函数。
   */
  void publish_trajectory()
  {
    const auto now{this->now()};
    path_msg_.header.stamp = now;
    publisher_path_->publish(path_msg_);

    if (path_msg_.poses.empty())
    {
      return;
    }

    static std::size_t index{0};
    const auto &msg_pose{path_msg_.poses[index]};
    nav_msgs::msg::Path path_pose_msg;
    path_pose_msg.header.frame_id = DEFAULT_FRAME_ID;
    path_pose_msg.header.stamp    = msg_pose.header.stamp;
    path_pose_msg.poses.push_back(msg_pose);
    publisher_pose_->publish(path_pose_msg);
    index = (index + 1) % path_msg_.poses.size();
  }

  std::string csv_file_;
  std::string topic_name_;
  bool skip_header_;
  char delim_;

  nav_msgs::msg::Path path_msg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace FastVIO

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<FastVIO::TrajectoryPublisher>();
    rclcpp::spin(node);
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "[ERROR] {}", ex.what());
  }
  rclcpp::shutdown();
  return 0;
}
