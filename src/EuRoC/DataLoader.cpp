#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono_literals;

#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "AbstractLoader.hpp"
#include "euroc_vio/main.h"

struct GroundTruthData
{
  double timestamp{0.0};
  double position[3]{0.0};
  double orientation[4]{0.0}; // quaternion (w, x, y, z)

  // 简单 CSV 解析函数
  static std::vector<GroundTruthData> ReadCsv(const std::string &filename,
                                              const char delim = ',')
  {
    std::vector<GroundTruthData> data;
    data.reserve(32767);

    std::ifstream file(filename);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      GroundTruthData gt;
      gt.timestamp = AbstractLoader::get_item_as_int64(ss, delim);
      for (int i = 0; i < 3; ++i)
      {
        gt.position[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      // qw, qx, qy, qz
      for (int i = 0; i < 4; ++i)
      {
        gt.orientation[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      data.push_back(gt);
    }
    return data;
  }
};

struct EstimationData
{
  double timestamp{0.0};
  double orientation[4]{0.0}; // quaternion (w, x, y, z)

  // 简单 CSV 解析函数
  static std::vector<EstimationData> ReadCsv(const std::string &filename,
                                             const char delim = ',')
  {
    std::vector<EstimationData> data;
    data.reserve(32767);

    std::ifstream file(filename);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      EstimationData gt;
      gt.timestamp = AbstractLoader::get_item_as_int64(ss, delim);
      for (int i = 0; i < 4; ++i)
      {
        gt.orientation[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      data.push_back(gt);
    }
    return data;
  }
};

struct OrbSlam3Data
{
  double timestamp{0.0};
  double position[3]{0.0};
  double orientation[4]{0.0}; // quaternion (w, x, y, z)

  // 简单 CSV 解析函数
  static std::vector<OrbSlam3Data> ReadCsv(const std::string &filename,
                                           const char delim = ',')
  {
    std::vector<OrbSlam3Data> data;
    data.reserve(32767);

    std::ifstream file(filename);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      OrbSlam3Data gt;
      gt.timestamp = AbstractLoader::get_item_as_double(ss, delim);
      for (int i = 0; i < 3; ++i)
      {
        gt.position[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      // @see: https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/Examples/Monocular/mono_euroc.cc#L200
      // @see: https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/System.cc#L1109
      // qx, qy, qz, qw
      for (int i : {1, 2, 3, 0})
      {
        gt.orientation[i] = AbstractLoader::get_item_as_double(ss, delim);
      }
      data.push_back(gt);
    }
    return data;
  }
};

// 插值查找最近的 GroundTruthData 或 OrbSlam3Data
template <typename DataType>
static DataType Interpolate(const std::vector<DataType> &data, double timestamp)
{
  // timestamp 单位: ns
  if (data.empty())
  {
    throw std::runtime_error("GroundTruth 数据为空");
  }
  if (timestamp <= data.front().timestamp)
  {
    return data.front();
  }
  if (timestamp >= data.back().timestamp)
  {
    return data.back();
  }
  // 二分查找
  size_t left = 0, right = data.size() - 1;
  while (left + 1 < right)
  {
    size_t mid = (left + right) / 2;
    if (data[mid].timestamp < timestamp)
    {
      left = mid;
    }
    else
    {
      right = mid;
    }
  }
  const auto &gt0 = data[left];
  const auto &gt1 = data[right];
  const double t0 = gt0.timestamp;
  const double t1 = gt1.timestamp;
  const double alpha
      = (t1 > t0) ? std::clamp((timestamp - t0) / (t1 - t0), 0.0, 1.0) : 0.0;
  DataType interp = gt0;
  // 位置向量：线性插值
  for (int i = 0; i < 3; ++i)
  {
    interp.position[i]
        = gt0.position[i] * (1 - alpha) + gt1.position[i] * alpha;
  }
  // 朝向四元数：球面线性插值
  Eigen::Quaterniond q0(gt0.orientation[0], gt0.orientation[1],
                        gt0.orientation[2], gt0.orientation[3]);
  Eigen::Quaterniond q1(gt1.orientation[0], gt1.orientation[1],
                        gt1.orientation[2], gt1.orientation[3]);
  Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
  interp.orientation[0]       = q_interp.w();
  interp.orientation[1]       = q_interp.x();
  interp.orientation[2]       = q_interp.y();
  interp.orientation[3]       = q_interp.z();
  interp.timestamp            = timestamp;
  return interp;
}

class DataLoader : public rclcpp::Node
{
private:
  const std::filesystem::path path_home_{
      std::getenv("HOME"),
  };
  const std::filesystem::path path_csv_truth_{
      path_home_ / "vio_ws" / "truth_data.csv",
  };
  const std::filesystem::path path_csv_est_{
      path_home_ / "vio_ws" / "cam0_estimated_quats.csv",
  };
  const std::filesystem::path path_csv_orbslam3_{
      path_home_ / "vio_ws" / "KeyFrameTrajectory.txt",
  };

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_truth_{
      create_publisher<nav_msgs::msg::Path>("/path_truth", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_truth_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_est_{
      create_publisher<nav_msgs::msg::Path>("/path_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_est_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_orbslam3_{
      create_publisher<nav_msgs::msg::Path>("/path_orbslam3", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_orbslam3_;

  std::vector<GroundTruthData> data_truth_;
  std::vector<EstimationData> data_est_{
      EstimationData::ReadCsv(path_csv_est_),
  };
  std::vector<OrbSlam3Data> data_orbslam3_;

public:
  DataLoader() : Node("DataLoader")
  {
    msg_path_truth_.header.frame_id    = DEFAULT_FRAME_ID;
    msg_path_est_.header.frame_id      = DEFAULT_FRAME_ID;
    msg_path_orbslam3_.header.frame_id = DEFAULT_FRAME_ID;
    std::print(stderr, "DataLoader ready ...\n");

    data_truth_.reserve(data_est_.size());
    data_orbslam3_.reserve(data_est_.size());
    const auto groundtruth_data{
        GroundTruthData::ReadCsv(path_csv_truth_),
    };
    const auto orbslam3_data{
        OrbSlam3Data::ReadCsv(path_csv_orbslam3_, ' '),
    };
    for (const EstimationData &datum_est : data_est_)
    {
      const auto datum_truth{
          Interpolate(groundtruth_data, datum_est.timestamp),
      };
      const auto datum_orbslam3{
          Interpolate(orbslam3_data, datum_est.timestamp),
      };
      data_truth_.push_back(datum_truth);
      data_orbslam3_.push_back(datum_orbslam3);
    }
  }

  void Start()
  {
    size_t frame_index{0};

    geometry_msgs::msg::PoseStamped msg_pose;
    // 把位置取为坐标原点，只比较三个数据源的朝向四元数
    msg_pose.pose.position.x = 0.0;
    msg_pose.pose.position.y = 0.0;
    msg_pose.pose.position.z = 0.0;

    for (;;)
    {
      if (!rclcpp::ok())
      {
        break;
      }

      const rclcpp::Time now{this->get_clock()->now()};

      msg_path_truth_.header.stamp    = now;
      msg_path_est_.header.stamp      = now;
      msg_path_orbslam3_.header.stamp = now;

      msg_path_truth_.poses.clear();
      msg_path_est_.poses.clear();
      msg_path_orbslam3_.poses.clear();

      frame_index %= data_est_.size();
      const auto datum_truth{data_truth_[frame_index]};
      const auto datum_est{data_est_[frame_index]};
      const auto datum_orbslam3{data_orbslam3_[frame_index]};

      std::print(stderr,
                 "正在展示时间戳 [{:.0f}] 对应的标架 ...\n"
                 "\t真值:           \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n"
                 "\t本实验的估计值:   \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n"
                 "\tOrbSlam3 估计值: \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n",
                 datum_est.timestamp, datum_truth.orientation[0],
                 datum_truth.orientation[1], datum_truth.orientation[2],
                 datum_truth.orientation[3], datum_est.orientation[0],
                 datum_est.orientation[1], datum_est.orientation[2],
                 datum_est.orientation[3], datum_orbslam3.orientation[0],
                 datum_orbslam3.orientation[1], datum_orbslam3.orientation[2],
                 datum_orbslam3.orientation[3]);

      msg_pose.header.stamp = now;

      msg_pose.pose.orientation.w = datum_truth.orientation[0];
      msg_pose.pose.orientation.x = datum_truth.orientation[1];
      msg_pose.pose.orientation.y = datum_truth.orientation[2];
      msg_pose.pose.orientation.z = datum_truth.orientation[3];
      msg_path_truth_.poses.push_back(msg_pose);
      publisher_path_truth_->publish(msg_path_truth_);

      msg_pose.pose.orientation.w = datum_est.orientation[0];
      msg_pose.pose.orientation.x = datum_est.orientation[1];
      msg_pose.pose.orientation.y = datum_est.orientation[2];
      msg_pose.pose.orientation.z = datum_est.orientation[3];
      msg_path_est_.poses.push_back(msg_pose);
      publisher_path_est_->publish(msg_path_est_);

      msg_pose.pose.orientation.w = datum_orbslam3.orientation[0];
      msg_pose.pose.orientation.x = datum_orbslam3.orientation[1];
      msg_pose.pose.orientation.y = datum_orbslam3.orientation[2];
      msg_pose.pose.orientation.z = datum_orbslam3.orientation[3];
      msg_path_orbslam3_.poses.push_back(msg_pose);
      publisher_path_orbslam3_->publish(msg_path_orbslam3_);

      std::print(stderr,
                 "=========================\n"
                 "正在发布第 {} 帧标架 ...\n"
                 "=========================\n",
                 frame_index);

      std::this_thread::sleep_for(1000ms);

      ++frame_index;
    }
  }
};

int main(int argc, char **argv)
{
  // 初始化 ROS 2
  rclcpp::init(argc, argv);

  try
  {
    DataLoader{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }

  // 关闭 ROS 2 实例
  rclcpp::shutdown();
  return 0;
}
