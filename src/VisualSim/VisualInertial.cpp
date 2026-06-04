#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <meta>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include <boost/numeric/odeint.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "yaml-cpp/yaml.h"

#include "ImuState.hpp"
#include "LinearKalmanFilter.hpp"
#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/Interpolation.hpp"
#include "euroc_vio/main.h"
#include "zupt.hpp"

#define PUBLISH_POSE 1
#define USE_TRUE_SCALE_IN_FAST 0

#define DATASOURCE_EUROC 0x01
#define DATASOURCE_SIM 0x10
#define DATASOURCE DATASOURCE_SIM

#pragma region DATA_LOADER

struct DatumFast
{
  std::int64_t timestamp_;
  // 角位移向量
  Eigen::Vector3f angular_displacement_;
  // 单位化平移向量
  Eigen::Vector3f normalized_translation_;

  static std::vector<DatumFast> Load(const std::string &path_estimation_csv)
  {
    std::vector<DatumFast> data;

    std::ifstream file(path_estimation_csv);
    std::string line;

    // 跳过表头
    std::getline(file, line);
    size_t line_num{0};
    while (std::getline(file, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
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
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_estimation_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};

struct DatumImu
{
  std::int64_t timestamp_;
  Eigen::Vector3f angular_velocity_;
  Eigen::Vector3f linear_acceleration_;

  static std::vector<DatumImu> Load(const std::string &path_imu_csv)
  {
    std::vector<DatumImu> data;

    std::ifstream file(path_imu_csv);
    std::string line;
    size_t line_num{0};

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
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
#if (DATASOURCE == DATASOURCE_EUROC)
            // EuRoC MAV 数据集的特殊要求:
            // 将 IMU 参考系的 X 轴映射为 Z 轴;
            // 将 IMU 参考系的 Y 轴映射为 -Y 轴;
            // 将 IMU 参考系的 Z 轴映射为 X 轴.
            // 这是因为数据集的 ground truth 是由 VICON0 或 LEICA0 提供的,
            // 而 IMU0 的三轴与 VICON0 或 LEICA0 的不同,
            // 只有按上述方式重映射以后，双方的标架才近似重合.
            {az, -ay, ax},
#else
            {ax, ay, az},
#endif
        };
        data.push_back(datum_fast);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_imu_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};

struct DatumTruth
{
  std::int64_t timestamp_;
  Eigen::Vector3f position_;
  Eigen::Quaternionf attitude_;
  Eigen::Vector3f velocity_;
  Eigen::Vector3f bias_gyro_;
  Eigen::Vector3f bias_accel_;

  static std::vector<DatumTruth> Load(const std::string &path_truth_csv)
  {
    std::vector<DatumTruth> data;

    std::ifstream file(path_truth_csv);
    std::string line;
    size_t line_num{0};

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
        // 读取时间戳
        const std::int64_t timestamp{
            AbstractLoader::get_item_as_int64(ss), // in nanoseconds
        };
        // 读取位置 (m)
        const float px{AbstractLoader::get_item_as_float(ss)};
        const float py{AbstractLoader::get_item_as_float(ss)};
        const float pz{AbstractLoader::get_item_as_float(ss)};
        // 读取朝向
        const float qw{AbstractLoader::get_item_as_float(ss)};
        const float qx{AbstractLoader::get_item_as_float(ss)};
        const float qy{AbstractLoader::get_item_as_float(ss)};
        const float qz{AbstractLoader::get_item_as_float(ss)};
        // 读取速度 (m s^-1)
        const float vx{AbstractLoader::get_item_as_float(ss)};
        const float vy{AbstractLoader::get_item_as_float(ss)};
        const float vz{AbstractLoader::get_item_as_float(ss)};
        // 读取陀螺仪偏差 (rad s^-1)
        const float bwx{AbstractLoader::get_item_as_float(ss)};
        const float bwy{AbstractLoader::get_item_as_float(ss)};
        const float bwz{AbstractLoader::get_item_as_float(ss)};
        // 读取加速度计偏差 (m s^-2)
        const float bax{AbstractLoader::get_item_as_float(ss)};
        const float bay{AbstractLoader::get_item_as_float(ss)};
        const float baz{AbstractLoader::get_item_as_float(ss)};
#if (DATASOURCE == DATASOURCE_EUROC)
#elif (DATASOURCE == DATASOURCE_SIM)
#endif

        const DatumTruth datum_truth{
            timestamp,        //
            {px, py, pz},     //
            {qw, qx, qy, qz}, //
            {vx, vy, vz},     //
            {bwx, bwy, bwz},  //
            {bax, bay, baz},  //
        };
        data.push_back(datum_truth);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_truth_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};

struct SensorConfig
{
  Eigen::Matrix4d transform_matrix_;

  SensorConfig() : transform_matrix_{Eigen::Matrix4d::Identity()} {}
  SensorConfig(const Eigen::Matrix4d &) = default;
  SensorConfig(Eigen::Matrix4d &&)      = default;
  ~SensorConfig()                       = default;

  static std::optional<SensorConfig>
  ReadSensorYaml(const std::string &path_sensor_yaml)
  {
    YAML::Node node_sensor{YAML::LoadFile(path_sensor_yaml)};
    if (node_sensor["T_BS"] && node_sensor["T_BS"]["data"])
    {
      std::vector<double> T_BS_data{
          config["T_BS"]["data"].as<std::vector<double>>()
      };
      Eigen::Map<Eigen::Matrix4d> T_BS_mat{T_BS_data.data()};
      return T_BS_mat;
    }
    return std::nullopt;
  }
};

#pragma endregion

/**
 * @brief 从指定文件中，读取角位移向量和单位化平移向量，通过一阶积分计算姿态、轨迹
 */
struct VisualInertial : public rclcpp::Node
{
  float gravity_world_norm{9.81f};

private:
#pragma region PRIVATE_MEMBER_VARIABLES

  bool use_true_init_pose_{false};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_fast_{
      create_publisher<nav_msgs::msg::Path>("/path_fast_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_fast_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_fast_{
      create_publisher<nav_msgs::msg::Path>("/pose_fast_est", rclcpp::QoS{10}),
  };
#endif

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_imu_{
      create_publisher<nav_msgs::msg::Path>("/path_imu_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_imu_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_imu_{
      create_publisher<nav_msgs::msg::Path>("/pose_imu_est", rclcpp::QoS{10}),
  };
#endif

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_rk4_{
      create_publisher<nav_msgs::msg::Path>("/path_rk4_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_rk4_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_rk4_{
      create_publisher<nav_msgs::msg::Path>("/pose_rk4_est", rclcpp::QoS{10}),
  };
#endif

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      publisher_path_preintegrate_{
          create_publisher<nav_msgs::msg::Path>("/path_preintegrate_est",
                                                rclcpp::QoS{10}),
      };
  nav_msgs::msg::Path msg_path_preintegrate_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      publisher_pose_preintegrate_{
          create_publisher<nav_msgs::msg::Path>("/pose_preintegrate_est",
                                                rclcpp::QoS{10}),
      };
#endif

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_fuse_{
      create_publisher<nav_msgs::msg::Path>("/path_fuse_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_fuse_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_fuse_{
      create_publisher<nav_msgs::msg::Path>("/pose_fuse_est", rclcpp::QoS{10}),
  };
#endif

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_truth_{
      create_publisher<nav_msgs::msg::Path>("/path_truth", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_truth_;
#if (PUBLISH_POSE)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_truth_{
      create_publisher<nav_msgs::msg::Path>("/pose_truth", rclcpp::QoS{10}),
  };
#endif

  std::string path_truth_csv_;

  std::vector<DatumFast> data_fast_{};
  std::vector<DatumImu> data_imu_{};
  std::vector<DatumTruth> data_truth_{};
  SensorConfig sensor_config_cam0_{};
  SensorConfig sensor_config_imu0_{};

  LinearKalmanFilter filter_;

#pragma endregion

private:
#pragma region ROS2_UTILITY

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

  void PublishPathImuEuler()
  {
    if (msg_path_imu_.poses.empty())
    {
      return;
    }
    msg_path_imu_.header.stamp = msg_path_imu_.poses.back().header.stamp;
    publisher_path_imu_->publish(msg_path_imu_);
  }

  void PublishPathPreintegrate()
  {
    if (msg_path_preintegrate_.poses.empty())
    {
      return;
    }
    msg_path_preintegrate_.header.stamp
        = msg_path_preintegrate_.poses.back().header.stamp;
    publisher_path_preintegrate_->publish(msg_path_preintegrate_);
  }

  void PublishPathImuRK4()
  {
    if (msg_path_rk4_.poses.empty())
    {
      return;
    }
    msg_path_rk4_.header.stamp = msg_path_rk4_.poses.back().header.stamp;
    publisher_path_rk4_->publish(msg_path_rk4_);
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

  void PublishPathTruth()
  {
    publisher_path_truth_->publish(msg_path_truth_);
  }

#if (PUBLISH_POSE)
  void PublishPoseFast(size_t index)
  {
    const auto &msg_pose{msg_path_fast_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_fast_->publish(msg_path_pose);
  }

  void PublishPoseImuEuler(size_t index)
  {
    const auto &msg_pose{msg_path_imu_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_imu_->publish(msg_path_pose);
  }

  void PublishPosePreintegrate(size_t index)
  {
    const auto &msg_pose{msg_path_preintegrate_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_preintegrate_->publish(msg_path_pose);
  }

  void PublishPoseImuRK4(size_t index)
  {
    const auto &msg_pose{msg_path_rk4_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_rk4_->publish(msg_path_pose);
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

  void PublishPoseTruth(size_t index)
  {
    const auto &msg_pose{msg_path_truth_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_truth_->publish(msg_path_pose);
  }
#endif /* PUBLISH_POSE */

#pragma endregion

#pragma region POSE_ESTIMATION

  /**
   * @brief 只靠单目相机提供的角位移向量和单位化平移向量估计位姿
   */
  void EstimateFast()
  {
    // 在工作目录下，用临时文件 path_fast_est.tum 存储 TUM 格式的数据
    // 它是利用 python 模块 evo，经过 SIM(3) 变换得到的相机轨迹数据
    const std::filesystem::path path_temp_tum_file{
        "path_fast_est.tum",
    };
    std::ofstream fout_fast(path_temp_tum_file);

    // 初始状态
    Eigen::Vector3f estimated_position_fast{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf estimated_attitude_fast{Eigen::Quaternionf::Identity()};

    for (size_t i = 0; i + 1 < data_fast_.size(); ++i)
    {
      const DatumFast &datum_fast{data_fast_[i]};
#if (USE_TRUE_SCALE_IN_FAST)
      const DatumFast &datum_fast_next{data_fast_[i + 1]};
#endif
      const Eigen::Quaternionf delta_rotation{
          Eigen::AngleAxisf{
              datum_fast.angular_displacement_.norm(),
              datum_fast.angular_displacement_.normalized(),
          },
      };

      Eigen::Vector3f delta_position{datum_fast.normalized_translation_};
#if (USE_TRUE_SCALE_IN_FAST)
      // TODO 利用插值查找函数 Interpolate 获取 delta_position 对应的真值的范数
      Eigen::Vector3f true_old_position{
          Interpolate(data_truth_, datum_fast.timestamp_).position_,
      };
      Eigen::Vector3f true_new_position{
          // 时间戳 + 50 毫秒
          Interpolate(data_truth_, datum_fast_next.timestamp_).position_,
      };
      delta_position
          = (true_new_position - true_old_position).norm() * delta_position;
#endif

      // 因为数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      // 所以应该使用以下状态更新方程
      estimated_position_fast
          = estimated_position_fast + estimated_attitude_fast * delta_position;
      estimated_attitude_fast
          = (estimated_attitude_fast * delta_rotation).normalized();

      std::print(fout_fast,
                 // 时间戳
                 "{:020d}, "
                 // 位置
                 "{:.18f}, {:.18f}, {:.18f}, "
                 // 朝向
                 "{:.18f}, {:.18f}, {:.18f}, {:.18f}\n",
                 datum_fast.timestamp_, estimated_position_fast.x(),
                 estimated_position_fast.y(), estimated_position_fast.z(),
                 estimated_attitude_fast.w(), estimated_attitude_fast.x(),
                 estimated_attitude_fast.y(), estimated_attitude_fast.z());
    } // end for

    std::print(stderr, "[INFO] 估计轨迹已写入 {}\n",
               std::filesystem::absolute(path_temp_tum_file).string());
    // 利用 evo 提供的 SIM(3) 变换调整轨迹
    std::system(
        std::format("bash -c '"
                    "source .venv/bin/activate && "
                    "yes 'y' | evo_traj euroc {} --ref={} "
                    "--align --correct_scale --save_as_tum'",
                    std::filesystem::absolute(path_temp_tum_file).string(),
                    std::filesystem::absolute(path_truth_csv_).string())
            .c_str()
    );

    std::ifstream fin_fast(path_temp_tum_file);
    std::string line;
    size_t line_num{0};
    while (std::getline(fin_fast, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
        // 读取时间戳
        const std::int64_t timestamp{
            static_cast<std::int64_t>(
                AbstractLoader::get_item_as_float(ss, ' ')
            ), // in nanoseconds
        };
        // 读取位置
        const float px{AbstractLoader::get_item_as_float(ss, ' ')};
        const float py{AbstractLoader::get_item_as_float(ss, ' ')};
        const float pz{AbstractLoader::get_item_as_float(ss, ' ')};
        // 读取朝向
        const float qw{AbstractLoader::get_item_as_float(ss, ' ')};
        const float qx{AbstractLoader::get_item_as_float(ss, ' ')};
        const float qy{AbstractLoader::get_item_as_float(ss, ' ')};
        const float qz{AbstractLoader::get_item_as_float(ss, ' ')};
        estimated_attitude_fast = Eigen::Quaternionf{qw, qx, qy, qz};
        estimated_position_fast = Eigen::Vector3f{px, py, pz};
        PushPose(msg_path_fast_, timestamp, estimated_attitude_fast,
                 estimated_position_fast);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_temp_tum_file.string(), line, //
                        ex.what()),
        };
      }
    } // end while
    std::print(stderr, "[INFO] 估计轨迹已缩放.\n");
    // 删除临时文件
    std::error_code ec;
    std::filesystem::remove(path_temp_tum_file, ec);
  }

  /**
   * @brief 只靠 IMU 提供的角速度向量和加速度向量估计位姿 (梯形方法求解常微分方程)
   */
  void EstimateImuEuler()
  {
    // 世界坐标系下的重力加速度
    const Eigen::Vector3f gravity_world{0.0f, 0.0f, -gravity_world_norm};

    // 初始状态
    Eigen::Vector3f estimated_position_imu{Eigen::Vector3f::Zero()};
    Sophus::SO3f estimated_attitude_imu{/* Eigen::Quaternionf::Identity() */};
    Eigen::Vector3f estimated_linear_velocity_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_linear_acceleration_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_angular_velocity_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_angular_acceleration_imu{Eigen::Vector3f::Zero()};

    if (use_true_init_pose_ && !data_truth_.empty())
    {
      estimated_position_imu        = data_truth_[0].position_;
      estimated_attitude_imu        = Sophus::SO3f(data_truth_[0].attitude_);
      estimated_linear_velocity_imu = data_truth_[0].velocity_;
    }
    else
    {
      // 引入“零速更新”机制，检测起飞时刻
      ZUPT<float> zupt{};
      bool is_orientation_estimated{false};

      DatumImu datum_first;
      DatumImu datum_last;
      for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
      {
        if (first_loop)
        {
          datum_first = datum_imu;
          first_loop  = false;
          continue;
        }
        if (zupt.IsFull() && !is_orientation_estimated)
        { // 当样本足够多时，如果尚未预测过初始朝向，就立即进行预测
          estimated_attitude_imu   = Sophus::SO3f(zupt.EstimateOrientation());
          is_orientation_estimated = true;
        }
        if (!zupt.Update(datum_imu.linear_acceleration_,
                         datum_imu.angular_velocity_))
        {
          datum_last = datum_imu;
          break;
        }
      } // end for
      // 静止状态的时长
      const float time_elapsed_before_takeoff{
          1e-9f
              * static_cast<float>(datum_last.timestamp_
                                   - datum_first.timestamp_),
      };
      std::print(stderr, "静止时长: {:.4f} 秒.\n", time_elapsed_before_takeoff);
      // 机体处于静止状态时，机体坐标系与世界坐标系不一定是重合的。
      // 以 EuRoC MAV 数据集为例，无人机起飞前，
      // 其机体坐标系（即 IMU 坐标系）的 X,Y,Z 三轴大致上
      // 分别与世界坐标系的 Z,-Y,X 三轴对应

      if (!is_orientation_estimated)
      { // 如果尚未预测过初始朝向，就立即进行预测
        estimated_attitude_imu = Sophus::SO3f(zupt.EstimateOrientation());
      }

      {
        Eigen::Matrix3f estimated_attitude_imu_matrix{
            estimated_attitude_imu.matrix(),
        };
        std::print(stderr,
                   "[INFO] ZUPT 估计初始姿态为 = [\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "]\n",
                   estimated_attitude_imu_matrix(0, 0),
                   estimated_attitude_imu_matrix(0, 1),
                   estimated_attitude_imu_matrix(0, 2),
                   estimated_attitude_imu_matrix(1, 0),
                   estimated_attitude_imu_matrix(1, 1),
                   estimated_attitude_imu_matrix(1, 2),
                   estimated_attitude_imu_matrix(2, 0),
                   estimated_attitude_imu_matrix(2, 1),
                   estimated_attitude_imu_matrix(2, 2));
      }
    }

    // 统计信息 (记录使用欧拉法估计位置、线速度产生的绝对误差)
    std::filesystem::path path_estimation_error{"VisualInertial-Imu-Error.csv"};
    std::ofstream fout_imu_euler_estimation_error(path_estimation_error);
    std::print(fout_imu_euler_estimation_error,
               "time [s],"
               "qw [],qx [],qy [],qz[],"
               "x [m],y [m],z [m],"
               "vx [m s^-1],vy [m s^-1],vz [m s^-1],"
               "ax [m s^-2], ay [m s^-2], az [m s^-2],"
               "dvx [m s^-1],dvy [m s^-1],dvz [m s^-1],"
               "dx [m s^-1],dy [m s^-1],dz [m s^-1],"
               "err(x) [m],err(y) [m],err(z) [m],"
               "err(vx) [m s^-1],err(vy) [m s^-1],err(vz) [m s^-1]\n");

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_imu;
        first_loop = false;

        const auto datum_true{Interpolate(data_truth_, datum_imu.timestamp_)};
        Eigen::Vector3f true_position{datum_true.position_};
        Eigen::Vector3f true_velocity{datum_true.velocity_};
        // 更新统计信息
        std::print(
            fout_imu_euler_estimation_error,
            // 时间戳
            "{:020d}, "
            // 朝向
            "{:.18f},{:.18f},{:.18f},{:.18f},"
            // 位置
            "{:.18f},{:.18f},{:.18f},"
            // 线速度
            "{:.18f},{:.18f},{:.18f},"
            // 线加速度
            "{:.18f},{:.18f},{:.18f},"
            // 线速度变化量
            "{:.18f},{:.18f},{:.18f},"
            // 位置变化量
            "{:.18f},{:.18f},{:.18f},"
            // 位置绝对误差
            "{:.18f},{:.18f},{:.18f},"
            // 线速度绝对误差
            "{:.18f},{:.18f},{:.18f}\n",
            datum_imu.timestamp_,                         //
            estimated_attitude_imu.unit_quaternion().w(), //
            estimated_attitude_imu.unit_quaternion().x(), //
            estimated_attitude_imu.unit_quaternion().y(), //
            estimated_attitude_imu.unit_quaternion().z(), //
            estimated_position_imu.x(),                   //
            estimated_position_imu.y(),                   //
            estimated_position_imu.z(),                   //
            estimated_linear_velocity_imu.x(),            //
            estimated_linear_velocity_imu.y(),            //
            estimated_linear_velocity_imu.z(),            //
            datum_imu.linear_acceleration_.x(),           //
            datum_imu.linear_acceleration_.y(),           //
            datum_imu.linear_acceleration_.z(),           //
            0.0,                                          //
            0.0,                                          //
            0.0,                                          //
            0.0,                                          //
            0.0,                                          //
            0.0,                                          //
            std::abs(estimated_position_imu.x() - true_position.x()),
            std::abs(estimated_position_imu.y() - true_position.y()),
            std::abs(estimated_position_imu.z() - true_position.z()),
            std::abs(estimated_linear_velocity_imu.x() - true_velocity.x()),
            std::abs(estimated_linear_velocity_imu.y() - true_velocity.y()),
            std::abs(estimated_linear_velocity_imu.z() - true_velocity.z())
        );

        continue;
      }

      // 时间步长
      const float dt{
          1e-9f
              * static_cast<float>(datum_imu.timestamp_
                                   - datum_prev.timestamp_),
      };

      // 传感器参考系下的角速度
      Eigen::Vector3f angular_velocity_in_sensor_frame{
          datum_prev.angular_velocity_,
      };
      // 朝向变化量
      Sophus::SO3f delta_attitude{
          Sophus::SO3f::exp(angular_velocity_in_sensor_frame * dt),
      };
      // 新的朝向
      Sophus::SO3f estimated_new_attitude_imu{
          estimated_attitude_imu * delta_attitude,
      };

      // 惯性参考系下的线加速度
      Eigen::Vector3f linear_acceleration_in_world_frame{
          estimated_attitude_imu * datum_prev.linear_acceleration_
              + gravity_world,
      };
      // 线速度变化量
      Eigen::Vector3f delta_velocity{
          linear_acceleration_in_world_frame * dt,
      };
      // 位置变化量
      Eigen::Vector3f delta_position{
          (estimated_linear_velocity_imu + 0.5f * delta_velocity) * dt,
      };

      // 更新位置
      estimated_position_imu += delta_position;
      // 更新线速度
      estimated_linear_velocity_imu += delta_velocity;
      // 更新朝向
      estimated_attitude_imu = estimated_new_attitude_imu;

      PushPose(msg_path_imu_, datum_imu.timestamp_,
               estimated_attitude_imu.unit_quaternion(),
               estimated_position_imu);

      const auto datum_true{Interpolate(data_truth_, datum_imu.timestamp_)};
      Eigen::Vector3f true_position{datum_true.position_};
      Eigen::Vector3f true_velocity{datum_true.velocity_};
      // 更新统计信息
      std::print(
          fout_imu_euler_estimation_error,
          // 时间戳
          "{:020d}, "
          // 朝向
          "{:.18f},{:.18f},{:.18f},{:.18f},"
          // 位置
          "{:.18f},{:.18f},{:.18f},"
          // 线速度
          "{:.18f},{:.18f},{:.18f},"
          // 线加速度
          "{:.18f},{:.18f},{:.18f},"
          // 线速度变化量
          "{:.18f},{:.18f},{:.18f},"
          // 位置变化量
          "{:.18f},{:.18f},{:.18f},"
          // 位置绝对误差
          "{:.18f},{:.18f},{:.18f},"
          // 线速度绝对误差
          "{:.18f},{:.18f},{:.18f}\n",
          datum_imu.timestamp_,                         //
          estimated_attitude_imu.unit_quaternion().w(), //
          estimated_attitude_imu.unit_quaternion().x(), //
          estimated_attitude_imu.unit_quaternion().y(), //
          estimated_attitude_imu.unit_quaternion().z(), //
          estimated_position_imu.x(),                   //
          estimated_position_imu.y(),                   //
          estimated_position_imu.z(),                   //
          estimated_linear_velocity_imu.x(),            //
          estimated_linear_velocity_imu.y(),            //
          estimated_linear_velocity_imu.z(),            //
          datum_imu.linear_acceleration_.x(),           //
          datum_imu.linear_acceleration_.y(),           //
          datum_imu.linear_acceleration_.z(),           //
          delta_velocity.x(),                           //
          delta_velocity.y(),                           //
          delta_velocity.z(),                           //
          delta_position.x(),                           //
          delta_position.y(),                           //
          delta_position.z(),                           //
          std::abs(estimated_position_imu.x() - true_position.x()),
          std::abs(estimated_position_imu.y() - true_position.y()),
          std::abs(estimated_position_imu.z() - true_position.z()),
          std::abs(estimated_linear_velocity_imu.x() - true_velocity.x()),
          std::abs(estimated_linear_velocity_imu.y() - true_velocity.y()),
          std::abs(estimated_linear_velocity_imu.z() - true_velocity.z())
      );

      datum_prev = datum_imu;
    } // end for

    fout_imu_euler_estimation_error.flush();
    std::print(stderr, "误差评估文件已写入 {}\n",
               std::filesystem::absolute(path_estimation_error).string());
  }

  void PreintegrateImu()
  {
    // 世界坐标系下的重力加速度
    const Eigen::Vector3f gravity_world{0.0f, 0.0f, -gravity_world_norm};

    // 初始状态
    Eigen::Vector3f estimated_position_pi{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_velocity_pi{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf estimated_attitude_pi{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf delta_R{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f delta_p{Eigen::Vector3f::Zero()};
    Eigen::Vector3f delta_v{Eigen::Vector3f::Zero()};
    float delta_t{0};
    float t_prev{0};

    // 统计信息
    Eigen::Vector3f bound_pi{Eigen::Vector3f::Zero()};

    for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
    {
      float t_samp{1e-9f * static_cast<float>(datum_imu.timestamp_)};
      if (first_loop)
      {
        first_loop = false;
        t_prev     = t_samp;
        continue;
      }

      // 时间步长
      const float dt{t_samp - t_prev};
      auto drotvec{dt * datum_imu.angular_velocity_};
      Eigen::Quaternionf dR{
          Eigen::AngleAxisf{
              drotvec.norm(),
              drotvec.normalized(),
          },
      };
      auto dv{dt * datum_imu.linear_acceleration_};
      auto dp{0.5f * dt * dv};
      delta_t += dt;
      delta_p += delta_v * dt + delta_R * dp;
      delta_v += delta_R * dv;
      delta_R = delta_R * dR;
      t_prev  = t_samp;

      estimated_position_pi = estimated_position_pi
                              + delta_t * estimated_velocity_pi
                              + 0.5f * delta_t * delta_t * gravity_world
                              + estimated_attitude_pi * delta_p;
      estimated_velocity_pi = estimated_velocity_pi + delta_t * gravity_world
                              + estimated_attitude_pi * delta_v;
      estimated_attitude_pi = estimated_attitude_pi * delta_R;

      PushPose(msg_path_preintegrate_, datum_imu.timestamp_,
               estimated_attitude_pi, estimated_position_pi);

      // 更新统计信息
      bound_pi.x()
          = std::max(bound_pi.x(), std::abs(estimated_position_pi.x()));
      bound_pi.y()
          = std::max(bound_pi.y(), std::abs(estimated_position_pi.y()));
      bound_pi.z()
          = std::max(bound_pi.z(), std::abs(estimated_position_pi.z()));
    } // end for

    std::print(stderr, "\nBoundary[PI]: [x: {:.4e}, y: {:.4e}, z: {:.4e}]\n",
               bound_pi.x(), bound_pi.y(), bound_pi.z());
  }

  /**
   * @brief 只靠 IMU 提供的角速度向量和加速度向量估计位姿 (梯形方法求解常微分方程)
   */
  void EstimateImuRK4()
  {
    if (data_imu_.empty())
    {
      return;
    }

    // 世界坐标系下的重力加速度
    const Eigen::Vector3f gravity_world{0.0f, 0.0f, -gravity_world_norm};

    // 时间
    float ode_time{static_cast<float>(1e-9f * data_imu_[0].timestamp_)};
    // 初始状态
    ImuState<float> state;
    // 积分器
    boost::numeric::odeint::runge_kutta4<ImuState<float>, float,
                                         ImuDerivative<float>>
        rk4;
    // 微分方程
    struct ImuKinematicsODE
    {
      const DatumImu &datum_prev_;
      const DatumImu &datum_next_;
      const Eigen::Vector3f &gravity_world_;

      void operator()(const ImuState<float> &x, ImuDerivative<float> &dxdt,
                      const float t) const
      {
        float alpha{
            (datum_next_.timestamp_ > datum_prev_.timestamp_)
                ? std::clamp(static_cast<float>((t - datum_prev_.timestamp_)
                                                / (datum_next_.timestamp_
                                                   - datum_prev_.timestamp_)),
                             0.0f, 1.0f)
                : 0.0f,
        };
        // 传感器参考系下的角速度
        const Eigen::Vector3f ang_vel_sensor{
            datum_prev_.angular_velocity_
                + (datum_next_.angular_velocity_
                   - datum_prev_.angular_velocity_)
                      * alpha,
        };
        // 提取当前姿态四元数
        Eigen::Quaternionf att_world{x.GetAttitude()};
        // 惯性参考系下的线速度
        Eigen::Vector3f lin_vec_world{x.GetVelocity()};
        // 传感器参考系下的加速度
        Eigen::Vector3f lin_acc_sensor{
            datum_prev_.linear_acceleration_
                + (datum_next_.linear_acceleration_
                   - datum_prev_.linear_acceleration_)
                      * alpha,
        };
        // 世界参考系下的加速度
        Eigen::Vector3f lin_acc_world{
            att_world * lin_acc_sensor + gravity_world_,
        };
        Eigen::Quaternionf half_rotation{
            0.0f,
            0.5f * ang_vel_sensor.x(),
            0.5f * ang_vel_sensor.y(),
            0.5f * ang_vel_sensor.z(),
        };
        Eigen::Quaternionf att_derivative_world{att_world * half_rotation};

        // 位置导数 = 速度
        dxdt.SetVelocity(lin_vec_world);

        // 速度导数 = 加速度
        dxdt.SetAcceleration(lin_acc_world);

        // 朝向导数 = 0.5 * 朝向 ** 角速度
        dxdt.SetAttitudeDerivative(att_derivative_world);
      }
    };

    if (use_true_init_pose_ && !data_truth_.empty())
    {
      state.SetPosition(data_truth_[0].position_);
      state.SetAttitude(data_truth_[0].attitude_);
      state.SetVelocity(data_truth_[0].velocity_);
    }
    else
    {
      // 引入“零速更新”机制，检测起飞时刻
      ZUPT<float> zupt{};
      bool is_orientation_estimated{false};
      // 初始朝向
      Eigen::Quaternionf estimated_attitude_rk{Eigen::Quaternionf::Identity()};

      DatumImu datum_first;
      DatumImu datum_last;
      for (bool first_loop{true}; const DatumImu &datum_rk : data_imu_)
      {
        if (first_loop)
        {
          datum_first = datum_rk;
          first_loop  = false;
          continue;
        }
        if (zupt.IsFull() && !is_orientation_estimated)
        { // 当样本足够多时，如果尚未预测过初始朝向，就立即进行预测
          estimated_attitude_rk    = zupt.EstimateOrientation();
          is_orientation_estimated = true;
        }
        if (!zupt.Update(datum_rk.linear_acceleration_,
                         datum_rk.angular_velocity_))
        {
          datum_last = datum_rk;
          break;
        }
      } // end for
      // 静止状态的时长
      const float time_elapsed_before_takeoff{
          1e-9f
              * static_cast<float>(datum_last.timestamp_
                                   - datum_first.timestamp_),
      };
      std::print(stderr, "静止时长: {:.4f} 秒.\n", time_elapsed_before_takeoff);
      // 机体处于静止状态时，机体坐标系与世界坐标系不一定是重合的。
      // 以 EuRoC MAV 数据集为例，无人机起飞前，
      // 其机体坐标系（即 IMU 坐标系）的 X,Y,Z 三轴大致上
      // 分别与世界坐标系的 Z,-Y,X 三轴对应

      if (!is_orientation_estimated)
      { // 如果尚未预测过初始朝向，就立即进行预测
        estimated_attitude_rk = zupt.EstimateOrientation();
      }
      state.SetAttitude(estimated_attitude_rk);

      {
        Eigen::Matrix3f estimated_attitude_rk_matrix{estimated_attitude_rk};
        std::print(stderr,
                   "[INFO] ZUPT 估计初始姿态为 = [\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                   "]\n",
                   estimated_attitude_rk_matrix(0, 0),
                   estimated_attitude_rk_matrix(0, 1),
                   estimated_attitude_rk_matrix(0, 2),
                   estimated_attitude_rk_matrix(1, 0),
                   estimated_attitude_rk_matrix(1, 1),
                   estimated_attitude_rk_matrix(1, 2),
                   estimated_attitude_rk_matrix(2, 0),
                   estimated_attitude_rk_matrix(2, 1),
                   estimated_attitude_rk_matrix(2, 2));
      }
    }

    // 统计信息 (记录使用龙格贝塔法估计位置、线速度产生的绝对误差)
    std::filesystem::path path_estimation_error{
        "VisualInertial-Imu-RK4-Error.csv"
    };
    std::ofstream fout_rk_estimation_error(path_estimation_error);
    std::print(fout_rk_estimation_error,
               "time [s],"
               "qw [],qx [],qy [],qz[],"
               "x [m],y [m],z [m],"
               "vx [m s^-1],vy [m s^-1],vz [m s^-1],"
               "ax [m s^-2], ay [m s^-2], az [m s^-2],"
               "err(x) [m],err(y) [m],err(z) [m],"
               "err(vx) [m s^-1],err(vy) [m s^-1],err(vz) [m s^-1]\n");

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_rk : data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_rk;
        first_loop = false;

        const auto datum_true{Interpolate(data_truth_, datum_rk.timestamp_)};
        Eigen::Vector3f true_position{datum_true.position_};
        Eigen::Vector3f true_velocity{datum_true.velocity_};
        // 更新统计信息
        std::print(fout_rk_estimation_error,
                   // 时间戳
                   "{:020d}, "
                   // 朝向
                   "{:.18f},{:.18f},{:.18f},{:.18f},"
                   // 位置
                   "{:.18f},{:.18f},{:.18f},"
                   // 线速度
                   "{:.18f},{:.18f},{:.18f},"
                   // 线加速度
                   "{:.18f},{:.18f},{:.18f},"
                   // 位置绝对误差
                   "{:.18f},{:.18f},{:.18f},"
                   // 线速度绝对误差
                   "{:.18f},{:.18f},{:.18f}\n",
                   datum_rk.timestamp_,               //
                   state.GetAttitude().w(),           //
                   state.GetAttitude().x(),           //
                   state.GetAttitude().y(),           //
                   state.GetAttitude().z(),           //
                   state.GetPosition().x(),           //
                   state.GetPosition().y(),           //
                   state.GetPosition().z(),           //
                   state.GetVelocity().x(),           //
                   state.GetVelocity().y(),           //
                   state.GetVelocity().z(),           //
                   datum_rk.linear_acceleration_.x(), //
                   datum_rk.linear_acceleration_.y(), //
                   datum_rk.linear_acceleration_.z(), //
                   std::abs(state.GetPosition().x() - true_position.x()),
                   std::abs(state.GetPosition().y() - true_position.y()),
                   std::abs(state.GetPosition().z() - true_position.z()),
                   std::abs(state.GetVelocity().x() - true_velocity.x()),
                   std::abs(state.GetVelocity().y() - true_velocity.y()),
                   std::abs(state.GetVelocity().z() - true_velocity.z()));

        continue;
      }

      // 时间步长
      const float dt{
          1e-9f
              * static_cast<float>(datum_rk.timestamp_ - datum_prev.timestamp_),
      };

      ImuKinematicsODE ode{datum_prev, datum_rk, gravity_world};
      rk4.do_step(ode, state, ode_time, dt);
      ode_time += dt;
      state.NormalizeAttitude();

      PushPose(msg_path_rk4_, datum_rk.timestamp_, state.GetAttitude(),
               state.GetPosition());

      const auto datum_true{Interpolate(data_truth_, datum_rk.timestamp_)};
      Eigen::Vector3f true_position{datum_true.position_};
      Eigen::Vector3f true_velocity{datum_true.velocity_};
      // 更新统计信息
      std::print(fout_rk_estimation_error,
                 // 时间戳
                 "{:020d}, "
                 // 朝向
                 "{:.18f},{:.18f},{:.18f},{:.18f},"
                 // 位置
                 "{:.18f},{:.18f},{:.18f},"
                 // 线速度
                 "{:.18f},{:.18f},{:.18f},"
                 // 线加速度
                 "{:.18f},{:.18f},{:.18f},"
                 // 位置绝对误差
                 "{:.18f},{:.18f},{:.18f},"
                 // 线速度绝对误差
                 "{:.18f},{:.18f},{:.18f}\n",
                 datum_rk.timestamp_,               //
                 state.GetAttitude().w(),           //
                 state.GetAttitude().x(),           //
                 state.GetAttitude().y(),           //
                 state.GetAttitude().z(),           //
                 state.GetPosition().x(),           //
                 state.GetPosition().y(),           //
                 state.GetPosition().z(),           //
                 state.GetVelocity().x(),           //
                 state.GetVelocity().y(),           //
                 state.GetVelocity().z(),           //
                 datum_rk.linear_acceleration_.x(), //
                 datum_rk.linear_acceleration_.y(), //
                 datum_rk.linear_acceleration_.z(), //
                 std::abs(state.GetPosition().x() - true_position.x()),
                 std::abs(state.GetPosition().y() - true_position.y()),
                 std::abs(state.GetPosition().z() - true_position.z()),
                 std::abs(state.GetVelocity().x() - true_velocity.x()),
                 std::abs(state.GetVelocity().y() - true_velocity.y()),
                 std::abs(state.GetVelocity().z() - true_velocity.z()));

      datum_prev = datum_rk;
    } // end for

    fout_rk_estimation_error.flush();
    std::print(stderr, "误差评估文件已写入 {}\n",
               std::filesystem::absolute(path_estimation_error).string());
  }

  /**
   * @brief 将四元数转换为欧拉角 (Roll, Pitch, Yaw)
   * @param q 表示朝向的 Eigen 四元数
   * @return Eigen::Vector3d 对应的欧拉角 (roll, pitch, yaw) 向量，单位为弧度
   */
  Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond &q)
  {
    Eigen::Vector3d angles;
    // 计算 roll (绕 x 轴)
    double sinr_cosp{2 * (q.w() * q.x() + q.y() * q.z())};
    double cosr_cosp{1 - 2 * (q.x() * q.x() + q.y() * q.y())};
    angles.x() = std::atan2(sinr_cosp, cosr_cosp);

    // 计算 pitch (绕 y 轴)
    double sinp{2 * (q.w() * q.y() - q.z() * q.x())};
    if (std::abs(sinp) >= 1)
    {
      angles.y() = std::copysign(M_PI / 2, sinp); // 超出范围时修正为 90 度
    }
    else
    {
      angles.y() = std::asin(sinp);
    }

    // 计算 yaw (绕 z 轴)
    double siny_cosp{2 * (q.w() * q.z() + q.x() * q.y())};
    double cosy_cosp{1 - 2 * (q.y() * q.y() + q.z() * q.z())};
    angles.z() = std::atan2(siny_cosp, cosy_cosp);

    return angles;
  }

  /**
   * @brief 基于松耦合的线性卡尔曼滤波，
            融合单目视觉提供的角位移向量、单位化平移向量信息，
            与 IMU 提供的角速度向量、线加速度向量信息。
   */
  void EstimateFuse()
  {
    // 定义离线统一的时间轴事件结构体，用于交织对齐异步的视觉序列与高频 IMU 序列
    struct TimelineEvent
    {
      std::int64_t timestamp; // 纳秒级全局统一时间戳
      bool is_imu;            // 标识当前事件是否属于惯性测量单元
      size_t index; // 记录当前帧在各自容器(data_fast_或data_imu_)中的原始索引值
    };

    std::vector<TimelineEvent> events;
    events.reserve(data_fast_.size() + data_imu_.size());

    // 填充单目视觉特征帧信息至时间轴中
    for (size_t i = 0; i < data_fast_.size(); ++i)
    {
      events.push_back({data_fast_[i].timestamp_, false, i});
    }
    // 填充高频惯性特征帧信息至时间轴中
    for (size_t i = 0; i < data_imu_.size(); ++i)
    {
      events.push_back({data_imu_[i].timestamp_, true, i});
    }

    // 针对混合时间轴事件根据时间戳升序排序，若时间戳相同则让 IMU 优先处理
    std::ranges::sort(events, std::less<>{}, [](const TimelineEvent &e)
                      { return std::make_tuple(e.timestamp, !e.is_imu); });

    std::int64_t last_timestamp{-1};
    Eigen::Vector3f cam_position{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf cam_attitude{Eigen::Quaternionf::Identity()};

    // 顺序迭代离线混合时间轴上的所有传感器事件
    for (const auto &event : events)
    {
      double dt{0.0};
      if (last_timestamp != -1)
      {
        // 动态计算前后两次状态演进事件的真实秒级时间差异步长
        dt = static_cast<double>(event.timestamp - last_timestamp) * 1e-9;
      }

      // 如果时间差异间隔大于 0 且滤波器具有有效历史时间戳，则根据当前真实步长执行状态预测演进
      if (dt > 0 && last_timestamp != -1)
      {
        // 动态调配并重构状态转移矩阵中与平移位置项相关的元素
        filter_.kf_.transitionMatrix.at<double>(0, 3) = dt;
        filter_.kf_.transitionMatrix.at<double>(1, 4) = dt;
        filter_.kf_.transitionMatrix.at<double>(2, 5) = dt;
        filter_.kf_.transitionMatrix.at<double>(3, 6) = dt;
        filter_.kf_.transitionMatrix.at<double>(4, 7) = dt;
        filter_.kf_.transitionMatrix.at<double>(5, 8) = dt;
        filter_.kf_.transitionMatrix.at<double>(0, 6) = 0.5 * dt * dt;
        filter_.kf_.transitionMatrix.at<double>(1, 7) = 0.5 * dt * dt;
        filter_.kf_.transitionMatrix.at<double>(2, 8) = 0.5 * dt * dt;

        // 动态调配并重构状态转移矩阵中与旋转朝向角相关的元素
        filter_.kf_.transitionMatrix.at<double>(9, 12)  = dt;
        filter_.kf_.transitionMatrix.at<double>(10, 13) = dt;
        filter_.kf_.transitionMatrix.at<double>(11, 14) = dt;
        filter_.kf_.transitionMatrix.at<double>(12, 15) = dt;
        filter_.kf_.transitionMatrix.at<double>(13, 16) = dt;
        filter_.kf_.transitionMatrix.at<double>(14, 17) = dt;
        filter_.kf_.transitionMatrix.at<double>(9, 15)  = 0.5 * dt * dt;
        filter_.kf_.transitionMatrix.at<double>(10, 16) = 0.5 * dt * dt;
        filter_.kf_.transitionMatrix.at<double>(11, 17) = 0.5 * dt * dt;

        filter_.kf_.predict();
      }

      // 实时递推更新当前的参考基础时间戳
      if (last_timestamp == -1 || dt > 0)
      {
        last_timestamp = event.timestamp;
      }

      if (event.is_imu)
      {
        // 提取当前由滤波器通过上层 Predict 推演得到的欧拉角姿态，用作世界系转换
        double roll{filter_.kf_.statePre.at<double>(9)};
        double pitch{filter_.kf_.statePre.at<double>(10)};
        double yaw{filter_.kf_.statePre.at<double>(11)};

        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q{yawAngle * pitchAngle * rollAngle};

        // 加载 IMU 载体坐标系原生的角速度与比力测量结果
        const auto &datum_imu = data_imu_[event.index];
        Eigen::Vector3d a_body(datum_imu.linear_acceleration_.x(),
                               datum_imu.linear_acceleration_.y(),
                               datum_imu.linear_acceleration_.z());
        Eigen::Vector3d w_body(datum_imu.angular_velocity_.x(),
                               datum_imu.angular_velocity_.y(),
                               datum_imu.angular_velocity_.z());

        // 旋转对齐到全局世界坐标系并严格剔除常数重力影响分量
        Eigen::Vector3d a_world{q * a_body};
        a_world(2) -= gravity_world_norm;

        // 应用非奇异伴随运动变换，映射转换至由系统定义的对应欧拉角速率
        double sr{std::sin(roll)};
        double cr{std::cos(roll)};
        double sp{std::sin(pitch)};
        double cp{std::cos(pitch)};
        Eigen::Matrix3d T;
        if (std::abs(cp) > 1e-4)
        {
          T << 1, sr * sp / cp, cr * sp / cp, 0, cr, -sr, 0, sr / cp, cr / cp;
        }
        else
        {
          T = Eigen::Matrix3d::Identity();
        }
        Eigen::Vector3d euler_rates{T * w_body};

        // 临时动态修改卡尔曼测量映射阵至高频惯性通道，并设置相应噪声协方差
        filter_.SetImuMeasurementModel();
        cv::Mat measurement{cv::Mat::zeros(6, 1, CV_64F)};
        measurement.at<double>(0) = a_world.x();
        measurement.at<double>(1) = a_world.y();
        measurement.at<double>(2) = a_world.z();
        measurement.at<double>(3) = euler_rates.x();
        measurement.at<double>(4) = euler_rates.y();
        measurement.at<double>(5) = euler_rates.z();

        // 依据当前测量修正更新卡尔曼后验状态量
        filter_.kf_.correct(measurement);
      }
      else
      {
        // 针对单目角位移向量和单位化平移向量，应用一阶增量公式积分获取连续的全局绝对位姿
        const auto &datum_fast = data_fast_[event.index];
        const Eigen::Quaternionf delta_rotation{
            Eigen::AngleAxisf{
                datum_fast.angular_displacement_.norm(),
                datum_fast.angular_displacement_.normalized(),
            },
        };

        cam_position
            = cam_position + cam_attitude * datum_fast.normalized_translation_;
        cam_attitude = (cam_attitude * delta_rotation).normalized();

        Eigen::Quaterniond q_cam = cam_attitude.cast<double>();
        Eigen::Vector3d euler    = QuaternionToEuler(q_cam);

        // 临时切换卡尔曼测量阵至低频全局位姿模式
        filter_.SetCamMeasurementModel();
        cv::Mat measurement{cv::Mat::zeros(6, 1, CV_64F)};
        measurement.at<double>(0) = static_cast<double>(cam_position.x());
        measurement.at<double>(1) = static_cast<double>(cam_position.y());
        measurement.at<double>(2) = static_cast<double>(cam_position.z());
        measurement.at<double>(3) = euler.x();
        measurement.at<double>(4) = euler.y();
        measurement.at<double>(5) = euler.z();

        // 叠加具有极小测量噪声不确定性的视觉全局解，触发高维状态的二次 Correct 联合估计
        filter_.kf_.correct(measurement);

        // 提取最终稳定融合所得的最优空间位移和欧拉姿态解
        double est_x{filter_.kf_.statePost.at<double>(0)};
        double est_y{filter_.kf_.statePost.at<double>(1)};
        double est_z{filter_.kf_.statePost.at<double>(2)};
        double est_roll{filter_.kf_.statePost.at<double>(9)};
        double est_pitch{filter_.kf_.statePost.at<double>(10)};
        double est_yaw{filter_.kf_.statePost.at<double>(11)};

        Eigen::AngleAxisd est_rollAngle(est_roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd est_pitchAngle(est_pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd est_yawAngle(est_yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond est_q{est_yawAngle * est_pitchAngle * est_rollAngle};
        Eigen::Vector3d est_p(est_x, est_y, est_z);

        // 将离线同步得到的融合位姿有序加入轨迹容器，使得两路轨迹数据帧总数与索引保持绝对等同，规避崩溃
        PushPose(msg_path_fuse_, datum_fast.timestamp_, est_q.cast<float>(),
                 est_p.cast<float>());
      }
    }
  }

#pragma endregion

public:
  VisualInertial() : Node("StereoSlam1")
  {
    this->declare_parameter("use_true_init_pose", false);
    use_true_init_pose_ = this->get_parameter("use_true_init_pose").as_bool();

    // std::filesystem::path{std::getenv("HOME")} / "vio_ws" / "estimated_motion.csv"
    this->declare_parameter("path_estimation_csv", "estimated_motion.csv");
    const std::string path_estimation_csv{
        this->get_parameter("path_estimation_csv").as_string(),
    };

    this->declare_parameter("path_cam0_yaml", "");
    const std::string path_cam0_yaml{
        this->get_parameter("path_cam0_yaml").as_string(),
    };

    // "/mnt/e/Documents/mav0/imu0/data.csv"
    // std::filesystem::path{std::getenv("HOME")} / "vio_ws" / "mav0" / "imu0" / "data.csv"
    this->declare_parameter("path_imu_csv", "");
    const std::string path_imu_csv{
        this->get_parameter("path_imu_csv").as_string(),
    };

    this->declare_parameter("path_imu_yaml", "");
    const std::string path_imu_yaml{
        this->get_parameter("path_imu_yaml").as_string(),
    };

    // "/mnt/e/Documents/mav0/state_groundtruth_estimate0/data.csv"
    // std::filesystem::path{std::getenv("HOME")} / "vio_ws" / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    this->declare_parameter("path_truth_csv", "");
    const std::string path_truth_csv{
        this->get_parameter("path_truth_csv").as_string(),
    };
    path_truth_csv_ = path_truth_csv;

    if (path_estimation_csv.empty())
    {
      throw std::runtime_error{"'path_estimation_csv' not specified."};
    }
    if (path_cam0_yaml.empty())
    {
      throw std::runtime_error{"'path_cam0_yaml' not specified."};
    }
    if (path_imu_csv.empty())
    {
      throw std::runtime_error{"'path_imu_csv' not specified."};
    }
    if (path_imu_yaml.empty())
    {
      throw std::runtime_error{"'path_imu_yaml' not specified."};
    }
    if (path_truth_csv.empty())
    {
      throw std::runtime_error{"'path_truth_csv' not specified."};
    }

    for (auto path_obj : {path_estimation_csv, path_imu_csv, path_truth_csv})
    {
      std::error_code ec;
      if (std::filesystem::is_regular_file(path_obj, ec))
      {
        continue;
      }
      throw std::runtime_error{std::format("FileNotFound: {}!", path_obj)};
    }

    auto opt_sensor_config_cam0_{SensorConfig::ReadSensorYaml(path_cam0_yaml)};
    if (opt_sensor_config_cam0_.has_value())
    {
      sensor_config_cam0_ = std::move(opt_sensor_config_cam0_.value());
    }
    else
    {
      throw std::runtime_error{std::format("Fail to parse {}!",
                                           path_cam0_yaml)};
    }
    auto opt_sensor_config_imu0_{SensorConfig::ReadSensorYaml(path_imu_yaml)};
    if (opt_sensor_config_imu0_.has_value())
    {
      sensor_config_imu0_ = std::move(opt_sensor_config_imu0_.value());
    }
    else
    {
      throw std::runtime_error{std::format("Fail to parse {}!", path_imu_yaml)};
    }

    data_fast_  = DatumFast::Load(path_estimation_csv);
    data_imu_   = DatumImu::Load(path_imu_csv);
    data_truth_ = DatumTruth::Load(path_truth_csv);

    std::print(stderr, "VisualInertial ready ...\n");
    msg_path_fast_.header.frame_id         = DEFAULT_FRAME_ID;
    msg_path_imu_.header.frame_id          = DEFAULT_FRAME_ID;
    msg_path_preintegrate_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_rk4_.header.frame_id          = DEFAULT_FRAME_ID;
    msg_path_fuse_.header.frame_id         = DEFAULT_FRAME_ID;
    msg_path_truth_.header.frame_id        = DEFAULT_FRAME_ID;

    if (!data_truth_.empty())
    {
      Eigen::Matrix3f true_init_attitude{data_truth_[0].attitude_};
      std::print(stderr,
                 "[INFO] Ground Truth 初始姿态为 = [\n"
                 "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                 "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                 "\t[{:.2f}, {:.2f}, {:.2f}]\n"
                 "]\n",
                 true_init_attitude(0, 0), true_init_attitude(0, 1),
                 true_init_attitude(0, 2), true_init_attitude(1, 0),
                 true_init_attitude(1, 1), true_init_attitude(1, 2),
                 true_init_attitude(2, 0), true_init_attitude(2, 1),
                 true_init_attitude(2, 2));
    }

    for (const DatumTruth &datum_truth : data_truth_)
    {
      PushPose(msg_path_truth_, datum_truth.timestamp_, datum_truth.attitude_,
               datum_truth.position_);
    } // end for
  }

  void Start()
  {
    EstimateFast();
    EstimateImuEuler();
    EstimateImuRK4();
    PreintegrateImu();
    EstimateFuse();

#if (PUBLISH_POSE)
    size_t index_fast{0};
    size_t index_imu_euler{0};
    size_t index_preintegrate{0};
    size_t index_imu_rk4{0};
    size_t index_fuse{0};
    size_t index_truth{0};
#endif

    while (rclcpp::ok())
    {
      PublishPathFast();
      PublishPathImuEuler();
      PublishPathImuRK4();
      PublishPathPreintegrate();
      PublishPathFuse();
      PublishPathTruth();

#if (PUBLISH_POSE)
      PublishPoseFast(index_fast);
      PublishPoseImuEuler(index_imu_euler);
      PublishPosePreintegrate(index_preintegrate);
      PublishPoseImuRK4(index_imu_rk4);
      PublishPoseFuse(index_fuse);
      PublishPoseTruth(index_truth);

      index_fast      = (index_fast + 1) % msg_path_fast_.poses.size();
      index_imu_euler = (index_imu_euler + 1) % msg_path_imu_.poses.size();
      index_preintegrate
          = (index_preintegrate + 1) % msg_path_preintegrate_.poses.size();
      index_imu_rk4 = (index_imu_rk4 + 1) % msg_path_rk4_.poses.size();
      index_fuse    = (index_fuse + 1) % msg_path_fuse_.poses.size();
      index_truth   = (index_truth + 1) % msg_path_truth_.poses.size();
#endif

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
