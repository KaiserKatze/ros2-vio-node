#include <algorithm>
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
// #include <visualization_msgs/msg/Marker.hpp>

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

    // ORBSLAM3 输出的位置信息需要绕 z 轴正方向旋转 90° 才能大致与 EuRoC 数据集提供的真值匹配
    const Eigen::Matrix3d mat_rotation{
        {0.0, -1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
    };

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

      {
        const auto px{gt.position[0]};
        const auto py{gt.position[1]};
        gt.position[0] = -py;
        gt.position[1] = px;
      }

      // @see: https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/Examples/Monocular/mono_euroc.cc#L200
      // @see: https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/System.cc#L1109
      // qx, qy, qz, qw
      for (int i : {1, 2, 3, 0})
      {
        gt.orientation[i] = AbstractLoader::get_item_as_double(ss, delim);
      }

      // 跟位置信息一样，对朝向四元数施加旋转
      {
        Eigen::Quaterniond orientation{
            gt.orientation[0],
            gt.orientation[1],
            gt.orientation[2],
            gt.orientation[3],
        };
        orientation
            = Eigen::Quaterniond(mat_rotation * orientation.toRotationMatrix())
                  .normalized();
        gt.orientation[0] = orientation.w();
        gt.orientation[1] = orientation.x();
        gt.orientation[2] = orientation.y();
        gt.orientation[3] = orientation.z();
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
      path_home_ / "vio_ws" / "CameraTrajectory.txt",
  };

  // 用来比较位置
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

  // 用来比较朝向
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_att_truth_{
      create_publisher<nav_msgs::msg::Path>("/att_truth", rclcpp::QoS{10}),
  };
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_att_est_{
      create_publisher<nav_msgs::msg::Path>("/att_est", rclcpp::QoS{10}),
  };
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_att_orbslam3_{
      create_publisher<nav_msgs::msg::Path>("/att_orbslam3", rclcpp::QoS{10}),
  };

  std::vector<GroundTruthData> data_truth_;
  std::vector<EstimationData> data_est_;
  std::vector<OrbSlam3Data> data_orbslam3_;

  double timestamp_start_{0.0};
  double timestamp_end_{0.0};

public:
  DataLoader() : Node("DataLoader")
  {
    std::print(stderr, "DataLoader ready ...\n");

    const auto estimation_data{
        EstimationData::ReadCsv(path_csv_est_),
    };
    const auto groundtruth_data{
        GroundTruthData::ReadCsv(path_csv_truth_),
    };
    const auto orbslam3_data{
        OrbSlam3Data::ReadCsv(path_csv_orbslam3_, ' '),
    };

    data_est_.reserve(orbslam3_data.size());
    data_truth_.reserve(orbslam3_data.size());
    data_orbslam3_.reserve(orbslam3_data.size());

    // std::print(stderr, "\n=== GroundTruthData : First 5 Lines ===\n"
    //                    "t, px, py, pz, qw, qx, qy, qz\n");
    // for (size_t i = 0; i < 5; ++i)
    // {
    //   const auto datum{groundtruth_data[i]};
    //   std::print("{}, {}, {}, {}, {}, {}, {}, {}\n", //
    //              datum.timestamp, datum.position[0], datum.position[1],
    //              datum.position[2], datum.orientation[0], datum.orientation[1],
    //              datum.orientation[2], datum.orientation[3]);
    // }

    // std::print(stderr, "\n=== OrbSlam3Data : First 5 Lines ===\n"
    //                    "t, px, py, pz, qw, qx, qy, qz\n");
    // for (size_t i = 0; i < 5; ++i)
    // {
    //   const auto datum{orbslam3_data[i]};
    //   std::print("{}, {}, {}, {}, {}, {}, {}, {}\n", //
    //              datum.timestamp, datum.position[0], datum.position[1],
    //              datum.position[2], datum.orientation[0], datum.orientation[1],
    //              datum.orientation[2], datum.orientation[3]);
    // }

    timestamp_start_ = std::max({
        groundtruth_data[0].timestamp,
        orbslam3_data[0].timestamp,
        estimation_data[0].timestamp,
    });
    timestamp_end_   = timestamp_start_;

    if (groundtruth_data.size() >= 1 && orbslam3_data.size() >= 1
        && estimation_data.size() >= 1)
    {
      timestamp_end_ = std::min({
          groundtruth_data[groundtruth_data.size() - 1].timestamp,
          orbslam3_data[orbslam3_data.size() - 1].timestamp,
          estimation_data[estimation_data.size() - 1].timestamp,
      });

      std::print(stderr,
                 "\n数据集时间范围:\n"
                 "\t真值:     \t[{:.1f}, {:.1f}]\n"
                 "\tORBSLAM3: \t[{:.1f}, {:.1f}]\n"
                 "\t本实验:   \t[{:.1f}, {:.1f}]\n"
                 "\t交集:     \t[{:.1f}, {:.1f}]\n",
                 groundtruth_data[0].timestamp,
                 groundtruth_data[groundtruth_data.size() - 1].timestamp,
                 orbslam3_data[0].timestamp,
                 orbslam3_data[orbslam3_data.size() - 1].timestamp,
                 estimation_data[0].timestamp,
                 estimation_data[estimation_data.size() - 1].timestamp,
                 timestamp_start_, timestamp_end_);
    }

    if (timestamp_end_ == timestamp_start_)
    {
      throw std::runtime_error{"时间范围没有交集!"};
    }

    decltype(GroundTruthData::position) delta_position{0.0};
    for (bool first_loop{true};
         const EstimationData &datum_est : estimation_data)
    {
      if (datum_est.timestamp < timestamp_start_
          || datum_est.timestamp > timestamp_end_)
      {
        // 跳过交集以外的数据
        continue;
      }

      const auto datum_truth{
          Interpolate(groundtruth_data, datum_est.timestamp),
      };
      auto datum_orbslam3{
          Interpolate(orbslam3_data, datum_est.timestamp),
      };

      // 将 ORBSLAM3 估计的相机初始位置平移到与相机初始位置真值重合
      if (first_loop)
      {
        delta_position[0]
            = datum_truth.position[0] - datum_orbslam3.position[0];
        delta_position[1]
            = datum_truth.position[1] - datum_orbslam3.position[1];
        delta_position[2]
            = datum_truth.position[2] - datum_orbslam3.position[2];
        first_loop = false;
      }
      else
      {
        datum_orbslam3.position[0] += delta_position[0];
        datum_orbslam3.position[1] += delta_position[1];
        datum_orbslam3.position[2] += delta_position[2];
      }

      data_est_.push_back(datum_est);
      data_truth_.push_back(datum_truth);
      data_orbslam3_.push_back(datum_orbslam3);
    }
  }

  void UpdatePathMessage(nav_msgs::msg::Path &msg_path_truth,
                         nav_msgs::msg::Path &msg_path_est,
                         nav_msgs::msg::Path &msg_path_orbslam3,
                         size_t frame_index)
  {
    const auto datum_truth{data_truth_[frame_index]};
    const auto datum_est{data_est_[frame_index]};
    const auto datum_orbslam3{data_orbslam3_[frame_index]};

    msg_path_truth.header.frame_id    = DEFAULT_FRAME_ID;
    msg_path_est.header.frame_id      = DEFAULT_FRAME_ID;
    msg_path_orbslam3.header.frame_id = DEFAULT_FRAME_ID;

    const rclcpp::Time now{this->get_clock()->now()};
    msg_path_truth.header.stamp    = now;
    msg_path_est.header.stamp      = now;
    msg_path_orbslam3.header.stamp = now;

    // std::print(stderr,
    //            "正在展示时间戳 [{:.0f}] 对应的标架 ...\n"
    //            "\t真值:           \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n"
    //            "\t本实验的估计值:   \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n"
    //            "\tOrbSlam3 估计值: \t[{:.10f}, {:.10f}, {:.10f}, {:.10f}]\n",
    //            datum_est.timestamp, datum_truth.orientation[0],
    //            datum_truth.orientation[1], datum_truth.orientation[2],
    //            datum_truth.orientation[3], datum_est.orientation[0],
    //            datum_est.orientation[1], datum_est.orientation[2],
    //            datum_est.orientation[3], datum_orbslam3.orientation[0],
    //            datum_orbslam3.orientation[1], datum_orbslam3.orientation[2],
    //            datum_orbslam3.orientation[3]);

    geometry_msgs::msg::PoseStamped msg_pose;

    msg_pose.header.stamp = now;

    msg_pose.pose.position.x    = datum_truth.position[0];
    msg_pose.pose.position.y    = datum_truth.position[1];
    msg_pose.pose.position.z    = datum_truth.position[2];
    msg_pose.pose.orientation.w = datum_truth.orientation[0];
    msg_pose.pose.orientation.x = datum_truth.orientation[1];
    msg_pose.pose.orientation.y = datum_truth.orientation[2];
    msg_pose.pose.orientation.z = datum_truth.orientation[3];
    msg_path_truth.poses.push_back(msg_pose);

    msg_pose.pose.orientation.w = datum_est.orientation[0];
    msg_pose.pose.orientation.x = datum_est.orientation[1];
    msg_pose.pose.orientation.y = datum_est.orientation[2];
    msg_pose.pose.orientation.z = datum_est.orientation[3];
    msg_path_est.poses.push_back(msg_pose);

    msg_pose.pose.position.x    = datum_orbslam3.position[0];
    msg_pose.pose.position.y    = datum_orbslam3.position[1];
    msg_pose.pose.position.z    = datum_orbslam3.position[2];
    msg_pose.pose.orientation.w = datum_orbslam3.orientation[0];
    msg_pose.pose.orientation.x = datum_orbslam3.orientation[1];
    msg_pose.pose.orientation.y = datum_orbslam3.orientation[2];
    msg_pose.pose.orientation.z = datum_orbslam3.orientation[3];
    msg_path_orbslam3.poses.push_back(msg_pose);
  }

  void Start()
  {
    size_t frame_index{0};

    // 构造函数已经保证 data_est_, data_truth_, data_orbslam3_ 三者等长
    for (; frame_index < data_est_.size(); ++frame_index)
    {
      if (!rclcpp::ok())
      {
        break;
      }

      UpdatePathMessage(msg_path_truth_, msg_path_est_, msg_path_orbslam3_,
                        frame_index);
    }

    for (frame_index = 0;;)
    {
      if (!rclcpp::ok())
      {
        break;
      }

      // 比较位置
      publisher_path_truth_->publish(msg_path_truth_);
      publisher_path_est_->publish(msg_path_est_);
      publisher_path_orbslam3_->publish(msg_path_orbslam3_);

      nav_msgs::msg::Path att_truth;
      nav_msgs::msg::Path att_est;
      nav_msgs::msg::Path att_orbslam3;
      UpdatePathMessage(att_truth, att_est, att_orbslam3, frame_index);

      // 比较朝向
      publisher_att_truth_->publish(att_truth);
      publisher_att_est_->publish(att_est);
      publisher_att_orbslam3_->publish(att_orbslam3);

      std::this_thread::sleep_for(100ms);

      // 循环播放
      frame_index = (frame_index + 1) % data_est_.size();
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
