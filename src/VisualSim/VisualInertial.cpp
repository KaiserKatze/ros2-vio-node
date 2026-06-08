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
#include <string_view>
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

#include "ErrorStateKalmanFilter.hpp"
#include "ImuState.hpp"
#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/Interpolation.hpp"
#include "euroc_vio/main.h"
#include "zupt.hpp"

#include "DatumFast.hpp"
#include "DatumImu.hpp"
#include "DatumTruth.hpp"
#include "ErrorEvaluation.hpp"
#include "EvoSim3.hpp"
#include "SensorYaml.hpp"

#define PUBLISH_POSE 1

#define DATASOURCE_EUROC 0x01
#define DATASOURCE_SIM 0x10
#define DATASOURCE DATASOURCE_SIM

/**
 * @brief 从指定文件中，读取角位移向量和单位化平移向量，通过一阶积分计算姿态、轨迹
 */
struct VisualInertial : public rclcpp::Node
{
  double gravity_world_norm{9.81};

private:
#pragma region PRIVATE_MEMBER_VARIABLES

  bool use_evo_sim3_{false};
  bool use_true_translation_in_fast_{false};
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
  SensorYaml sensor_config_cam0_{};
  SensorYaml sensor_config_imu0_{};
  SensorYaml sensor_config_truth_{};

  // 引入 ESKF 松耦合姿态解算器，替代原本精度较低的 opencv 线性卡尔曼解算模型
  using ESKF = ErrorStateKalmanFilter<double>;
  ESKF filter_;

#pragma endregion

private:
#pragma region ROS2_UTILITY

  void PushPose(nav_msgs::msg::Path &msg_path, const std::int64_t timestamp,
                const Eigen::Quaterniond &attitude,
                const Eigen::Vector3d &position)
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
    std::print(stderr, "[INFO] EstimateFast invoked ...\n");

    EvoSim3 evo_sim3_fast{};

    // 初始状态
    Eigen::Vector3d estimated_position_fast{Eigen::Vector3d::Zero()};
    Sophus::SO3d estimated_attitude_fast{};

    // 统计信息
    ErrorEvaluation err_eval_fast{"VisualInertial-Fast-Error.csv", data_truth_};

    for (size_t i = 0; i + 1 < data_fast_.size(); ++i)
    {
      const DatumFast &datum_fast{data_fast_[i]};
      const auto angular_displacement_norm{
          datum_fast.angular_displacement_.norm(),
      };
      Sophus::SO3d delta_rotation{};
      if (angular_displacement_norm > 1e-6)
      {
        delta_rotation = Sophus::SO3d::exp(datum_fast.angular_displacement_);
      }

      Eigen::Vector3d delta_position{datum_fast.normalized_translation_};
      if (use_true_translation_in_fast_)
      {
        const DatumFast &datum_fast_next{data_fast_[i + 1]};
        // 利用插值查找函数 Interpolate 获取 delta_position 对应的真值的范数
        Eigen::Vector3d true_old_position{
            Interpolate(data_truth_, datum_fast.timestamp_).position_,
        };
        Eigen::Vector3d true_new_position{
            Interpolate(data_truth_, datum_fast_next.timestamp_).position_,
        };
        delta_position = true_new_position - true_old_position;
      }

      // 因为数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      // 所以应该使用以下状态更新方程
      estimated_position_fast
          = estimated_position_fast + estimated_attitude_fast * delta_position;
      estimated_attitude_fast = estimated_attitude_fast * delta_rotation;

      err_eval_fast.WriteErrorEvaluation(datum_fast.timestamp_,   //
                                         estimated_attitude_fast, //
                                         estimated_position_fast, //
                                         Eigen::Vector3d::Zero());

      Eigen::Quaterniond estimated_quaternion_fast{
          estimated_attitude_fast.unit_quaternion()
      };
      if (use_evo_sim3_)
      {
        evo_sim3_fast.Write(datum_fast.timestamp_, estimated_position_fast,
                            estimated_quaternion_fast);
      }
      else
      {
        PushPose(msg_path_fast_, datum_fast.timestamp_,
                 estimated_quaternion_fast, estimated_position_fast);
      }
    } // end for

    if (use_evo_sim3_)
    {
      evo_sim3_fast.TransformSim3(path_truth_csv_);

      evo_sim3_fast.Read(
          [this](std::int64_t timestamp, const Eigen::Quaterniond &attitude,
                 const Eigen::Vector3d &position)
          {
            this->PushPose(this->msg_path_fast_, timestamp, attitude, position);
          }
      );
    }
  }

  /**
   * @brief 只靠 IMU 提供的角速度向量和加速度向量估计位姿 (梯形方法求解常微分方程)
   */
  void EstimateImuEuler()
  {
    std::print(stderr, "[INFO] EstimateImuEuler invoked ...\n");

    // 世界坐标系下的重力加速度
    const Eigen::Vector3d gravity_world{0.0, 0.0, -gravity_world_norm};

    // 初始状态
    Eigen::Vector3d estimated_position_imu{Eigen::Vector3d::Zero()};
    Sophus::SO3d estimated_attitude_imu{/* Eigen::Quaterniond::Identity() */};
    Eigen::Vector3d estimated_linear_velocity_imu{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimated_linear_acceleration_imu{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimated_angular_velocity_imu{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimated_angular_acceleration_imu{Eigen::Vector3d::Zero()};

    if (use_true_init_pose_ && !data_truth_.empty())
    {
      estimated_position_imu        = data_truth_[0].position_;
      estimated_attitude_imu        = Sophus::SO3d(data_truth_[0].attitude_);
      estimated_linear_velocity_imu = data_truth_[0].velocity_;
    }
    else
    {
      // 引入“零速更新”机制，检测起飞时刻
      ZUPT<double> zupt{};
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
          estimated_attitude_imu   = Sophus::SO3d(zupt.EstimateOrientation());
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
      const double time_elapsed_before_takeoff{
          1e-9f
              * static_cast<double>(datum_last.timestamp_
                                    - datum_first.timestamp_),
      };
      std::print(stderr, "静止时长: {:.4f} 秒.\n", time_elapsed_before_takeoff);
      // 机体处于静止状态时，机体坐标系与世界坐标系不一定是重合的。
      // 以 EuRoC MAV 数据集为例，无人机起飞前，
      // 其机体坐标系（即 IMU 坐标系）的 X,Y,Z 三轴大致上
      // 分别与世界坐标系的 Z,-Y,X 三轴对应

      if (!is_orientation_estimated)
      { // 如果尚未预测过初始朝向，就立即进行预测
        estimated_attitude_imu = Sophus::SO3d(zupt.EstimateOrientation());
      }

      {
        Eigen::Matrix3d estimated_attitude_imu_matrix{
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
    ErrorEvaluation err_eval_imu{"VisualInertial-Imu-Euler-Error.csv",
                                 data_truth_};

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_imu;
        first_loop = false;

        err_eval_imu.WriteErrorEvaluation(datum_imu.timestamp_,   //
                                          estimated_attitude_imu, //
                                          estimated_position_imu, //
                                          estimated_linear_velocity_imu);

        continue;
      }

      // 时间步长
      const double dt{
          1e-9f
              * static_cast<double>(datum_imu.timestamp_
                                    - datum_prev.timestamp_),
      };

      // 载具参考系下的角速度 = 传感器参考系下的角速度
      // 载具参考系下前一帧角速度
      Eigen::Vector3d previous_angular_velocity_in_body_frame{
          datum_prev.angular_velocity_,
      };
      // 载具参考系下后一帧角速度
      Eigen::Vector3d current_angular_velocity_in_body_frame{
          datum_imu.angular_velocity_,
      };
      // 载具参考系下两帧角速度的平均值
      Eigen::Vector3d median_angular_velocity_in_body_frame{
          0.5
              * (previous_angular_velocity_in_body_frame
                 + current_angular_velocity_in_body_frame),
      };
      // 朝向变化量
      Sophus::SO3d delta_attitude{
          Sophus::SO3d::exp(
              median_angular_velocity_in_body_frame * dt
              + (dt * dt / 12.0)
                    * previous_angular_velocity_in_body_frame.cross(
                        current_angular_velocity_in_body_frame
                    )
          ),
      };
      // 新的朝向
      Sophus::SO3d estimated_new_attitude_imu{
          estimated_attitude_imu * delta_attitude,
      };

      // 惯性参考系下的线加速度
      Eigen::Vector3d linear_acceleration_in_world_frame{
          estimated_attitude_imu * datum_prev.linear_acceleration_
              + gravity_world,
      };
      // 线速度变化量
      Eigen::Vector3d delta_velocity{
          linear_acceleration_in_world_frame * dt,
      };
      // 位置变化量
      Eigen::Vector3d delta_position{
          (estimated_linear_velocity_imu + 0.5 * delta_velocity) * dt,
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

      err_eval_imu.WriteErrorEvaluation(datum_imu.timestamp_,   //
                                        estimated_attitude_imu, //
                                        estimated_position_imu, //
                                        estimated_linear_velocity_imu);

      datum_prev = datum_imu;
    } // end for
  }

  void PreintegrateImu()
  {
    std::print(stderr, "[INFO] PreintegrateImu invoked ...\n");

    // 世界坐标系下的重力加速度
    const Eigen::Vector3d gravity_world{0.0, 0.0, -gravity_world_norm};

    // 初始状态
    Eigen::Vector3d estimated_position_pi{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimated_velocity_pi{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond estimated_attitude_pi{Eigen::Quaterniond::Identity()};
    Eigen::Quaterniond delta_R{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d delta_p{Eigen::Vector3d::Zero()};
    Eigen::Vector3d delta_v{Eigen::Vector3d::Zero()};
    double delta_t{0.0};
    double t_prev{0.0};

    // 统计信息
    Eigen::Vector3d bound_pi{Eigen::Vector3d::Zero()};

    for (bool first_loop{true}; const DatumImu &datum_imu : data_imu_)
    {
      double t_samp{1e-9f * static_cast<double>(datum_imu.timestamp_)};
      if (first_loop)
      {
        first_loop = false;
        t_prev     = t_samp;
        continue;
      }

      // 时间步长
      const double dt{t_samp - t_prev};
      auto drotvec{dt * datum_imu.angular_velocity_};
      Eigen::Quaterniond dR{
          Eigen::AngleAxisd{
              drotvec.norm(),
              drotvec.normalized(),
          },
      };
      auto dv{dt * datum_imu.linear_acceleration_};
      auto dp{0.5 * dt * dv};
      delta_t += dt;
      delta_p += delta_v * dt + delta_R * dp;
      delta_v += delta_R * dv;
      delta_R = delta_R * dR;
      t_prev  = t_samp;

      estimated_position_pi = estimated_position_pi
                              + delta_t * estimated_velocity_pi
                              + 0.5 * delta_t * delta_t * gravity_world
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
    std::print(stderr, "[INFO] EstimateImuRK4 invoked ...\n");

    if (data_imu_.empty())
    {
      return;
    }

    // 世界坐标系下的重力加速度
    const Eigen::Vector3d gravity_world{0.0, 0.0, -gravity_world_norm};

    // 时间
    double ode_time{static_cast<double>(1e-9f * data_imu_[0].timestamp_)};
    // 初始状态
    ImuState<double> state;
    // 积分器
    boost::numeric::odeint::runge_kutta4<ImuState<double>, double,
                                         ImuDerivative<double>>
        rk4;
    // 微分方程
    struct ImuKinematicsODE
    {
      const DatumImu &datum_prev_;
      const DatumImu &datum_next_;
      const Eigen::Vector3d &gravity_world_;

      void operator()(const ImuState<double> &x, ImuDerivative<double> &dxdt,
                      const double t) const
      {
        double alpha{
            (datum_next_.timestamp_ > datum_prev_.timestamp_)
                ? std::clamp(static_cast<double>((t - datum_prev_.timestamp_)
                                                 / (datum_next_.timestamp_
                                                    - datum_prev_.timestamp_)),
                             0.0, 1.0)
                : 0.0,
        };
        // 传感器参考系下的角速度
        const Eigen::Vector3d ang_vel_sensor{
            datum_prev_.angular_velocity_
                + (datum_next_.angular_velocity_
                   - datum_prev_.angular_velocity_)
                      * alpha,
        };
        // 提取当前姿态四元数
        Eigen::Quaterniond att_world{x.GetAttitude()};
        // 惯性参考系下的线速度
        Eigen::Vector3d lin_vec_world{x.GetVelocity()};
        // 传感器参考系下的加速度
        Eigen::Vector3d lin_acc_sensor{
            datum_prev_.linear_acceleration_
                + (datum_next_.linear_acceleration_
                   - datum_prev_.linear_acceleration_)
                      * alpha,
        };
        // 世界参考系下的加速度
        Eigen::Vector3d lin_acc_world{
            att_world * lin_acc_sensor + gravity_world_,
        };
        Eigen::Quaterniond half_rotation{
            0.0,
            0.5 * ang_vel_sensor.x(),
            0.5 * ang_vel_sensor.y(),
            0.5 * ang_vel_sensor.z(),
        };
        Eigen::Quaterniond att_derivative_world{att_world * half_rotation};

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
      ZUPT<double> zupt{};
      bool is_orientation_estimated{false};
      // 初始朝向
      Eigen::Quaterniond estimated_attitude_rk{Eigen::Quaterniond::Identity()};

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
      const double time_elapsed_before_takeoff{
          1e-9f
              * static_cast<double>(datum_last.timestamp_
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
        Eigen::Matrix3d estimated_attitude_rk_matrix{estimated_attitude_rk};
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
    ErrorEvaluation err_eval_rk4{"VisualInertial-Imu-RK4-Error.csv",
                                 data_truth_};

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_rk : data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_rk;
        first_loop = false;

        err_eval_rk4.WriteErrorEvaluation(datum_rk.timestamp_,               //
                                          Sophus::SO3d{state.GetAttitude()}, //
                                          state.GetPosition(),               //
                                          state.GetVelocity());

        continue;
      }

      // 时间步长
      const double dt{
          1e-9f
              * static_cast<double>(datum_rk.timestamp_
                                    - datum_prev.timestamp_),
      };

      ImuKinematicsODE ode{datum_prev, datum_rk, gravity_world};
      rk4.do_step(ode, state, ode_time, dt);
      ode_time += dt;
      state.NormalizeAttitude();

      PushPose(msg_path_rk4_, datum_rk.timestamp_, state.GetAttitude(),
               state.GetPosition());

      err_eval_rk4.WriteErrorEvaluation(datum_rk.timestamp_,               //
                                        Sophus::SO3d{state.GetAttitude()}, //
                                        state.GetPosition(),               //
                                        state.GetVelocity());

      datum_prev = datum_rk;
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
   * @brief 基于松耦合的误差状态卡尔曼滤波 (ESKF)，
            融合单目视觉提供的角位移向量、单位化平移向量信息，
            与 IMU 提供的角速度向量、线加速度向量信息。
   */
  void EstimateFuse()
  {
    std::print(stderr, "[INFO] EstimateFuse invoked ...\n");

    EvoSim3 evo_sim3_fuse{};

#pragma region 建立事件序列

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

#pragma endregion

#pragma region 初始化 ESKF

    // 初始化 ESKF 的标称状态
    typename ESKF::NominalStateVariable init_state;
    if (use_true_init_pose_ && !data_truth_.empty())
    {
      init_state.position_           = data_truth_[0].position_;
      init_state.linear_velocity_    = data_truth_[0].velocity_;
      init_state.attitude_           = data_truth_[0].attitude_;
      init_state.accelerometer_bias_ = data_truth_[0].bias_accel_;
      init_state.gyroscope_bias_     = data_truth_[0].bias_gyro_;
    }
    else
    {
      // TODO 尚未编码专用的初始姿态解算机制
      if (!data_truth_.empty())
      {
        init_state.attitude_ = data_truth_[0].attitude_;
      }
    }
    filter_.SetNominalState(init_state);

    // 将 YAML 中读取的传感器物理特征噪声传入 ESKF 进行精确的过程协方差计算
    filter_.SetGyroscopeNoiseDensity(
        sensor_config_imu0_.gyroscope_noise_density_
    );
    filter_.SetGyroscopeRandomWalk(sensor_config_imu0_.gyroscope_random_walk_);
    filter_.SetAccelerometerNoiseDensity(
        sensor_config_imu0_.accelerometer_noise_density_
    );
    filter_.SetAccelerometerRandomWalk(
        sensor_config_imu0_.accelerometer_random_walk_
    );

#pragma endregion

    // 顺序迭代离线混合时间轴上的所有传感器事件
    for (const auto &event : events)
    {
      if (event.is_imu)
      {
        // 传递高频 IMU 采样数据，执行 ESKF 标称状态前推以及误差状态协方差的时间传播
        const auto &datum_imu{data_imu_[event.index]};
        filter_.ImuUpdate(&datum_imu);
      }
      else
      {
        // 加载当前帧低频单目视觉观测信息并调用 ESKF 的观测融合与后验误差校正
        const auto &datum_fast{data_fast_[event.index]};
        filter_.MonocularUpdate(&datum_fast);

        // 获取融合后的最新名义状态
        auto state{filter_.GetNominalState()};

        if (use_evo_sim3_)
        {
          evo_sim3_fuse.Write(datum_fast.timestamp_, state.position_,
                              state.attitude_);
        }
        else
        {
          PushPose(msg_path_fuse_, datum_fast.timestamp_, state.attitude_,
                   state.position_);
        }
      }
    } // end for

    if (use_evo_sim3_)
    {
      evo_sim3_fuse.TransformSim3(path_truth_csv_);

      evo_sim3_fuse.Read(
          [this](std::int64_t timestamp, const Eigen::Quaterniond &attitude,
                 const Eigen::Vector3d &position)
          {
            this->PushPose(this->msg_path_fuse_, timestamp, attitude, position);
          }
      );
    }
  }

#pragma endregion

public:
  VisualInertial() : Node("StereoSlam1")
  {
    this->declare_parameter("use_true_translation_in_fast", false);
    use_true_translation_in_fast_
        = this->get_parameter("use_true_translation_in_fast").as_bool();

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

    this->declare_parameter("path_truth_yaml", "");
    const std::string path_truth_yaml{
        this->get_parameter("path_truth_yaml").as_string(),
    };

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
    if (path_truth_yaml.empty())
    {
      throw std::runtime_error{"'path_truth_yaml' not specified."};
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

    auto opt_sensor_config_cam0{SensorYaml::ReadSensorYaml(path_cam0_yaml)};
    if (opt_sensor_config_cam0.has_value())
    {
      sensor_config_cam0_ = std::move(opt_sensor_config_cam0.value());
      std::print(stderr,
                 "[INFO] T_BS_cam0 =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 sensor_config_cam0_.transform_matrix_(0, 0),
                 sensor_config_cam0_.transform_matrix_(0, 1),
                 sensor_config_cam0_.transform_matrix_(0, 2),
                 sensor_config_cam0_.transform_matrix_(0, 3),
                 sensor_config_cam0_.transform_matrix_(1, 0),
                 sensor_config_cam0_.transform_matrix_(1, 1),
                 sensor_config_cam0_.transform_matrix_(1, 2),
                 sensor_config_cam0_.transform_matrix_(1, 3),
                 sensor_config_cam0_.transform_matrix_(2, 0),
                 sensor_config_cam0_.transform_matrix_(2, 1),
                 sensor_config_cam0_.transform_matrix_(2, 2),
                 sensor_config_cam0_.transform_matrix_(2, 3),
                 sensor_config_cam0_.transform_matrix_(3, 0),
                 sensor_config_cam0_.transform_matrix_(3, 1),
                 sensor_config_cam0_.transform_matrix_(3, 2),
                 sensor_config_cam0_.transform_matrix_(3, 3));
    }
    else
    {
      throw std::runtime_error{std::format("Fail to parse {}!",
                                           path_cam0_yaml)};
    }
    auto opt_sensor_config_imu0{SensorYaml::ReadSensorYaml(path_imu_yaml)};
    if (opt_sensor_config_imu0.has_value())
    {
      sensor_config_imu0_ = std::move(opt_sensor_config_imu0.value());
      std::print(stderr,
                 "[INFO] T_BS_imu0 =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 sensor_config_imu0_.transform_matrix_(0, 0),
                 sensor_config_imu0_.transform_matrix_(0, 1),
                 sensor_config_imu0_.transform_matrix_(0, 2),
                 sensor_config_imu0_.transform_matrix_(0, 3),
                 sensor_config_imu0_.transform_matrix_(1, 0),
                 sensor_config_imu0_.transform_matrix_(1, 1),
                 sensor_config_imu0_.transform_matrix_(1, 2),
                 sensor_config_imu0_.transform_matrix_(1, 3),
                 sensor_config_imu0_.transform_matrix_(2, 0),
                 sensor_config_imu0_.transform_matrix_(2, 1),
                 sensor_config_imu0_.transform_matrix_(2, 2),
                 sensor_config_imu0_.transform_matrix_(2, 3),
                 sensor_config_imu0_.transform_matrix_(3, 0),
                 sensor_config_imu0_.transform_matrix_(3, 1),
                 sensor_config_imu0_.transform_matrix_(3, 2),
                 sensor_config_imu0_.transform_matrix_(3, 3));
    }
    else
    {
      throw std::runtime_error{std::format("Fail to parse {}!", path_imu_yaml)};
    }
    auto opt_sensor_config_truth{SensorYaml::ReadSensorYaml(path_truth_yaml)};
    if (opt_sensor_config_truth.has_value())
    {
      sensor_config_truth_ = std::move(opt_sensor_config_truth.value());
      std::print(stderr,
                 "[INFO] T_BS_truth =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 sensor_config_truth_.transform_matrix_(0, 0),
                 sensor_config_truth_.transform_matrix_(0, 1),
                 sensor_config_truth_.transform_matrix_(0, 2),
                 sensor_config_truth_.transform_matrix_(0, 3),
                 sensor_config_truth_.transform_matrix_(1, 0),
                 sensor_config_truth_.transform_matrix_(1, 1),
                 sensor_config_truth_.transform_matrix_(1, 2),
                 sensor_config_truth_.transform_matrix_(1, 3),
                 sensor_config_truth_.transform_matrix_(2, 0),
                 sensor_config_truth_.transform_matrix_(2, 1),
                 sensor_config_truth_.transform_matrix_(2, 2),
                 sensor_config_truth_.transform_matrix_(2, 3),
                 sensor_config_truth_.transform_matrix_(3, 0),
                 sensor_config_truth_.transform_matrix_(3, 1),
                 sensor_config_truth_.transform_matrix_(3, 2),
                 sensor_config_truth_.transform_matrix_(3, 3));
    }
    else
    {
      throw std::runtime_error{std::format("Fail to parse {}!",
                                           path_truth_yaml)};
    }

    data_fast_ = DatumFast::Load(
        path_estimation_csv,
        Sophus::SO3d{
            sensor_config_cam0_.transform_matrix_.template block<3, 3>(0, 0),
        }
    );
    data_imu_ = DatumImu::Load(
        path_imu_csv,
        Sophus::SO3d{
            sensor_config_imu0_.transform_matrix_.template block<3, 3>(0, 0),
        }
    );
    data_truth_ = DatumTruth::Load(
        path_truth_csv,
        Sophus::SO3d{
            sensor_config_truth_.transform_matrix_.template block<3, 3>(0, 0),
        }
    );

    std::print(stderr, "[INFO] VisualInertial ready ...\n");
    msg_path_fast_.header.frame_id         = DEFAULT_FRAME_ID;
    msg_path_imu_.header.frame_id          = DEFAULT_FRAME_ID;
    msg_path_preintegrate_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_rk4_.header.frame_id          = DEFAULT_FRAME_ID;
    msg_path_fuse_.header.frame_id         = DEFAULT_FRAME_ID;
    msg_path_truth_.header.frame_id        = DEFAULT_FRAME_ID;

    if (!data_truth_.empty())
    {
      Eigen::Matrix3d true_init_attitude{data_truth_[0].attitude_};
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
    std::print(stderr, "[INFO] VisualInertial started.\n");

    EstimateFast();
    EstimateImuEuler();
    EstimateImuRK4();
    PreintegrateImu();
    EstimateFuse();

    std::print(stderr, "[INFO] Path '{}' has {} poses.\n", //
               "msg_path_fast_", msg_path_fast_.poses.size());
    std::print(stderr, "[INFO] Path '{}' has {} poses.\n", //
               "msg_path_imu_", msg_path_imu_.poses.size());
    std::print(stderr, "[INFO] Path '{}' has {} poses.\n", //
               "msg_path_rk4_", msg_path_rk4_.poses.size());
    std::print(stderr, "[INFO] Path '{}' has {} poses.\n",
               "msg_path_preintegrate_", msg_path_preintegrate_.poses.size());
    std::print(stderr, "[INFO] Path '{}' has {} poses.\n", //
               "msg_path_fuse_", msg_path_fuse_.poses.size());
    std::print(stderr, "[INFO] Path '{}' has {} poses.\n", //
               "msg_path_truth_", msg_path_truth_.poses.size());

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
