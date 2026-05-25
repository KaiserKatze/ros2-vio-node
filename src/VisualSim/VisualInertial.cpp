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
#include <string>
#include <thread>

using namespace std::chrono_literals;

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "LinearKalmanFilter.hpp"
#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/main.h"

/**
 * @note
      因为数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      所以应该使用以下状态更新方程:

      position = position + attitude * delta_position;
      attitude = (attitude * delta_rotation).normalized();
 */
struct DatumFast
{
  std::int64_t timestamp_;
  Eigen::Vector3f angular_displacement_;
  Eigen::Vector3f normalized_translation_;

  static std::vector<DatumFast> Load()
  {
    static const std::filesystem::path path_home{
        std::getenv("HOME"),
    };
    static const std::filesystem::path path_estimation_csv{
        path_home / "vio_ws" / "estimated_motion.csv",
    };
    std::vector<DatumFast> data;

    std::ifstream file(path_estimation_csv);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);

      // 读取时间戳
      const std::int64_t timestamp{
          AbstractLoader::get_item_as_int64(ss), // in nanoseconds
      };
      // 读取旋转角度
      const float wxt{AbstractLoader::get_item_as_float(ss)};
      const float wyt{AbstractLoader::get_item_as_float(ss)};
      const float wzt{AbstractLoader::get_item_as_float(ss)};
      // 读取位移方向
      const float tx{AbstractLoader::get_item_as_float(ss)};
      const float ty{AbstractLoader::get_item_as_float(ss)};
      const float tz{AbstractLoader::get_item_as_float(ss)};

      const DatumFast datum_fast{
          timestamp,
          {wxt, wyt, wzt},
          {tx, ty, tz},
      };
      data.push_back(datum_fast);
    } // end while
    return data;
  }
};

struct DatumImu
{
  std::int64_t timestamp_;
  Eigen::Vector3f angular_velocity_;
  Eigen::Vector3f linear_acceleration_;

  static std::vector<DatumImu> Load()
  {
    static const std::filesystem::path path_imu_csv{
        "/mnt/e/Documents/mav0/imu0/data.csv",
    };
    std::vector<DatumImu> data;

    std::ifstream file(path_imu_csv);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);

      // 读取时间戳
      const std::int64_t timestamp{
          AbstractLoader::get_item_as_int64(ss), // in nanoseconds
      };
      // 读取旋转角度
      const float gx{AbstractLoader::get_item_as_float(ss)};
      const float gy{AbstractLoader::get_item_as_float(ss)};
      const float gz{AbstractLoader::get_item_as_float(ss)};
      // 读取位移方向
      const float ax{AbstractLoader::get_item_as_float(ss)};
      const float ay{AbstractLoader::get_item_as_float(ss)};
      const float az{AbstractLoader::get_item_as_float(ss)};

      const DatumImu datum_fast{
          timestamp,
          {gx, gy, gz},
          {ax, ay, az},
      };
      data.push_back(datum_fast);
    } // end while
    return data;
  }
};

/**
 * @brief 从指定文件中，读取角位移向量和单位化平移向量，通过一阶积分计算姿态、轨迹
 */
struct VisualInertial : public rclcpp::Node
{
private:
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_fast_{
      create_publisher<nav_msgs::msg::Path>("/path_fast_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_fast_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_fast_{
      create_publisher<nav_msgs::msg::Path>("/pose_fast_est", rclcpp::QoS{10}),
  };

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_fuse_{
      create_publisher<nav_msgs::msg::Path>("/path_fuse_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_fuse_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_fuse_{
      create_publisher<nav_msgs::msg::Path>("/pose_fuse_est", rclcpp::QoS{10}),
  };

  std::vector<DatumFast> data_fast_{DatumFast::Load()};
  std::vector<DatumImu> data_imu_{DatumImu::Load()};

  LinearKalmanFilter filter_;

  void PushPose(nav_msgs::msg::Path &msg_path, const std::int64_t timestamp,
                const Eigen::Quaternionf &attitude,
                const Eigen::Vector3f &position)
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

    msg_path.poses.push_back(msg_pose);
  }

  void PublishPathFast()
  {
    if (msg_path_fast_.poses.empty())
    {
      return;
    }
    msg_path_fast_.header.stamp = msg_path_fast_.poses.back().header.stamp;
    publisher_path_fast_->publish(msg_path_fast_);
  }

  void PublishPathFuse()
  {
    if (msg_path_fuse_.poses.empty())
    {
      return;
    }
    msg_path_fuse_.header.stamp = msg_path_fuse_.poses.back().header.stamp;
    publisher_path_fuse_->publish(msg_path_fuse_);
  }

  void PublishPoseFast(size_t index)
  {
    const auto &msg_pose{msg_path_fast_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_fast_->publish(msg_path_pose);
  }

  void PublishPoseFuse(size_t index)
  {
    const auto &msg_pose{msg_path_fuse_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_fuse_->publish(msg_path_pose);
  }

  void EstimateFast()
  {
    // 初始状态
    Eigen::Vector3f estimated_position_fast{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf estimated_attitude_fast{Eigen::Quaternionf::Identity()};

    for (const DatumFast &datum_fast : data_fast_)
    {
      const Eigen::Quaternionf delta_rotation{
          Eigen::AngleAxisf{
              datum_fast.angular_displacement_.norm(),
              datum_fast.angular_displacement_.normalized(),
          },
      };

      // 因为数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      // 所以应该使用以下状态更新方程
      estimated_position_fast
          = estimated_position_fast
            + estimated_attitude_fast * datum_fast.normalized_translation_;
      estimated_attitude_fast
          = (estimated_attitude_fast * delta_rotation).normalized();

      PushPose(msg_path_fast_, datum_fast.timestamp_, estimated_attitude_fast,
               estimated_position_fast);
    } // end for
  }

  void EstimateFuse()
  {
    // TODO 参考 FuseKalman.cpp 中使用的基于松耦合的线性卡尔曼滤波
    // 融合单目视觉提供的角位移向量、单位化平移向量信息与 IMU 提供的角速度向量、线加速度向量信息
    // 注意: FuseKalman 中融合的是单目视觉提供的位置和朝向信息与 IMU 提供的角速度向量、线加速度向量信息
    // 这里需要做相应修改
    LinearKalmanFilter filter;
  }

public:
  VisualInertial() : Node("StereoSlam1")
  {
    std::print(stderr, "VisualInertial ready ...\n");
    msg_path_fast_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_fuse_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void Start()
  {
    EstimateFast();
    EstimateFuse();

    size_t index_fast{0};
    size_t index_fuse{0};
    while (rclcpp::ok())
    {
      PublishPathFast();
      PublishPoseFast(index_fast);
      PublishPathFuse();
      PublishPoseFuse(index_fast);

      index_fast = (index_fast + 1) % msg_path_fast_.poses.size();
      index_fuse = (index_fuse + 1) % msg_path_fuse_.poses.size();

      std::this_thread::sleep_for(50ms);
    } // end while
  }
};

int main(int argc, char *argv[])
{
  // 初始化 ROS 2
  rclcpp::init(argc, argv);

  try
  {
    VisualInertial{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }

  // 关闭 ROS 2 实例
  rclcpp::shutdown();
  return 0;
}
