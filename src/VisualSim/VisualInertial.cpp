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

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "LinearKalmanFilter.hpp"
#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/main.h"
#include "zupt.hpp"

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
          // EuRoC MAV 数据集的特殊要求:
          // 将 IMU 参考系的 X 轴映射为 Z 轴;
          // 将 IMU 参考系的 Y 轴映射为 -Y 轴;
          // 将 IMU 参考系的 Z 轴映射为 X 轴.
          // 这是因为数据集的 ground truth 是由 VICON0 或 LEICA0 提供的,
          // 而 IMU0 的三轴与 VICON0 或 LEICA0 的不同,
          // 只有按上述方式重映射以后，双方的标架才近似重合.
          {az, -ay, ax},
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

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_imu_{
      create_publisher<nav_msgs::msg::Path>("/path_imu_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_imu_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_imu_{
      create_publisher<nav_msgs::msg::Path>("/pose_imu_est", rclcpp::QoS{10}),
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

private:
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

  void PublishPathImu()
  {
    if (msg_path_imu_.poses.empty())
    {
      return;
    }
    msg_path_imu_.header.stamp = msg_path_imu_.poses.back().header.stamp;
    publisher_path_imu_->publish(msg_path_imu_);
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

  void PublishPoseImu(size_t index)
  {
    const auto &msg_pose{msg_path_imu_.poses[index]};
    nav_msgs::msg::Path msg_path_pose;
    msg_path_pose.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_pose.header.stamp    = msg_pose.header.stamp;
    msg_path_pose.poses.push_back(msg_pose);
    publisher_pose_imu_->publish(msg_path_pose);
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

  /**
   * @brief 只靠单目相机提供的角位移向量和单位化平移向量估计位姿
   */
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

  /**
   * @brief 只靠 IMU 提供的角速度向量和加速度向量估计位姿
   */
  void EstimateImu()
  {
    // 世界坐标系下的重力加速度
    const Eigen::Vector3f gravity_world{0.0f, 0.0f, -9.81f};

    // 引入“零速更新”机制，检测起飞时刻
    ZUPT<float> zupt{};

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
      if (!zupt.Update(std::make_pair(datum_imu.linear_acceleration_,
                                      datum_imu.angular_velocity_)))
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

    // 初始状态
    Eigen::Vector3f estimated_position_imu{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf estimated_attitude_imu{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f estimated_linear_velocity_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_linear_acceleration_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_angular_velocity_imu{Eigen::Vector3f::Zero()};
    Eigen::Vector3f estimated_angular_acceleration_imu{Eigen::Vector3f::Zero()};
    estimated_attitude_imu = zupt.EstimateOrientation();

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_imu;
        first_loop = false;
        continue;
      }

      // 时间步长
      const float dt{
          1e-9f
              * static_cast<float>(datum_imu.timestamp_
                                   - datum_prev.timestamp_),
      };
      // 传感器参考系下的平均线加速度
      Eigen::Vector3f average_linear_acceleration_in_sensor_frame{
          0.5
              * (datum_prev.linear_acceleration_
                 + datum_imu.linear_acceleration_),
      };
      // 惯性参考系下的平均线加速度
      Eigen::Vector3f average_linear_acceleration_in_world_frame{
          estimated_attitude_imu
                  * average_linear_acceleration_in_sensor_frame //
              + gravity_world,
      };
      Eigen::Vector3f delta_velocity{
          average_linear_acceleration_in_world_frame * dt,
      };
      Eigen::Vector3f delta_position{
          (estimated_linear_velocity_imu + 0.5f * delta_velocity) * dt,
      };
      estimated_position_imu += delta_position;
      estimated_linear_velocity_imu += delta_velocity;

      // 传感器参考系下的平均角速度
      Eigen::Vector3f average_angular_velocity_in_sensor_frame{
          0.5f * (datum_prev.angular_velocity_ + datum_imu.angular_velocity_),
      };
      // 传感器参考系下的旋转向量
      Eigen::Vector3f rotation_vector_in_sensor_frame{
          average_angular_velocity_in_sensor_frame * dt,
      };
      Eigen::Quaternionf delta_attitude(Eigen::AngleAxisf{
          rotation_vector_in_sensor_frame.norm(),
          rotation_vector_in_sensor_frame.normalized(),
      });
      estimated_attitude_imu = estimated_attitude_imu * delta_attitude;

      PushPose(msg_path_imu_, datum_imu.timestamp_, estimated_attitude_imu,
               estimated_position_imu);
    } // end for
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
        a_world(2) -= 9.81;

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

public:
  VisualInertial() : Node("StereoSlam1")
  {
    std::print(stderr, "VisualInertial ready ...\n");
    msg_path_fast_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_imu_.header.frame_id  = DEFAULT_FRAME_ID;
    msg_path_fuse_.header.frame_id = DEFAULT_FRAME_ID;
  }

  void Start()
  {
    EstimateFast();
    EstimateImu();
    EstimateFuse();

    size_t index_fast{0};
    size_t index_imu{0};
    size_t index_fuse{0};
    while (rclcpp::ok())
    {
      PublishPathFast();
      PublishPoseFast(index_fast);
      PublishPathImu();
      PublishPoseImu(index_imu);
      PublishPathFuse();
      PublishPoseFuse(index_fuse);

      index_fast = (index_fast + 1) % msg_path_fast_.poses.size();
      index_imu  = (index_imu + 1) % msg_path_imu_.poses.size();
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
