#pragma once

#include <span>

#include <Eigen/Core>

#include <sophus/so3.hpp>

#include "euroc_vio/DatumImu.hpp"

struct Preintegration
{
  // 范围内总的预积分旋转量
  Eigen::Quaterniond delta_attitude_{Eigen::Quaterniond::Identity()};
  // 范围内总的预积分平移量
  Eigen::Vector3d delta_position_{Eigen::Vector3d::Zero()};
  // 范围内总的预积分速度量
  Eigen::Vector3d delta_linear_velocity_{Eigen::Vector3d::Zero()};
  // 结合初始状态推算的世界系最终朝向
  Eigen::Quaterniond final_attitude_{Eigen::Quaterniond::Identity()};
  // 结合初始状态推算的世界系最终位置
  Eigen::Vector3d final_position_{Eigen::Vector3d::Zero()};
  // 结合初始状态推算的世界系最终速度
  Eigen::Vector3d final_linear_velocity_{Eigen::Vector3d::Zero()};

  Preintegration() noexcept {}

  Preintegration(const Eigen::Vector3d &initial_position,
                       const Eigen::Quaterniond &initial_attitude,
                       const Eigen::Vector3d &initial_linear_velocity) noexcept
    :
    final_position_{initial_position}, final_attitude_{initial_attitude},
    final_linear_velocity_{initial_linear_velocity}
  {
  }

  Preintegration(const Preintegration &) = default;

  Preintegration(Preintegration &&) = default;

  ~Preintegration() = default;

  /**
   * @brief 改进后的高精度 IMU 预积分函数（中值积分法 + 二阶交叉项补偿）
   * @param imu_data IMU 数据视图段
   * @param initial_position 起始位置 (世界坐标系)
   * @param initial_attitude 起始朝向 (世界坐标系)
   * @param initial_linear_velocity 起始线速度 (世界坐标系)
   * @param gyroscope_bias 陀螺仪零偏
   * @param accelerometer_bias 加速度计零偏
   * @return 包含预积分相对量与世界坐标系最终绝对状态的结构体
   */
  void Update(std::span<const DatumImu> imu_data,
              const Eigen::Vector3d &gyroscope_bias,
              const Eigen::Vector3d &accelerometer_bias,
              const Eigen::Vector3d &gravity_world) noexcept
  {
    if (imu_data.size() < 2)
    {
      return;
    }

    double sum_dt{0.0};

    for (size_t i = 0; i < imu_data.size() - 1; ++i)
    {
      const auto &imu_i{imu_data[i]};
      const auto &imu_j{imu_data[i + 1]};

      const double dt{
          1e-9 * static_cast<double>(imu_j.timestamp_ - imu_i.timestamp_),
      };
      if (dt <= 0.0)
      {
        continue;
      }

      sum_dt += dt;

      // 减去零偏
      Eigen::Vector3d unbias_angular_velocity_body_i{imu_i.angular_velocity_
                                                     - gyroscope_bias};
      Eigen::Vector3d unbias_angular_velocity_body_j{imu_j.angular_velocity_
                                                     - gyroscope_bias};
      Eigen::Vector3d unbias_linear_acceleration_body_i{
          imu_i.linear_acceleration_ - accelerometer_bias
      };
      Eigen::Vector3d unbias_linear_acceleration_body_j{
          imu_j.linear_acceleration_ - accelerometer_bias
      };

      // --- 旋转预积分：中值角速度 + 二阶交叉项补偿 ---
      Eigen::Vector3d unbias_angular_velocity_body_midpoint{
          0.5
          * (unbias_angular_velocity_body_i + unbias_angular_velocity_body_j)
      };
      Sophus::SO3d dR{
          Sophus::SO3d::exp(unbias_angular_velocity_body_midpoint * dt
                            + (dt * dt / 12.0)
                                  * unbias_angular_velocity_body_i.cross(
                                      unbias_angular_velocity_body_j
                                  ))
      };

      Eigen::Quaterniond new_delta_attitude{
          (this->delta_attitude_ * dR.unit_quaternion()).normalized()
      };

      // --- 平移和速度预积分：积分系下的中值加速度 ---
      Eigen::Vector3d unbias_linear_accleration_world_i{
          this->delta_attitude_ * unbias_linear_acceleration_body_i
      };
      Eigen::Vector3d unbias_linear_accleration_world_j{
          new_delta_attitude * unbias_linear_acceleration_body_j
      };
      Eigen::Vector3d unbias_linear_accleration_world_midpoint{
          0.5
          * (unbias_linear_accleration_world_i
             + unbias_linear_accleration_world_j)
      };

      this->delta_position_
          += this->delta_linear_velocity_ * dt
             + 0.5 * unbias_linear_accleration_world_midpoint * dt * dt;
      this->delta_linear_velocity_
          += unbias_linear_accleration_world_midpoint * dt;
      this->delta_attitude_ = new_delta_attitude;
    }

    // --- 计算世界坐标系下的最终绝对状态 ---

    this->final_position_ = this->final_position_
                            + this->final_linear_velocity_ * sum_dt
                            + 0.5 * gravity_world * sum_dt * sum_dt
                            + this->final_attitude_ * this->delta_position_;
    this->final_linear_velocity_
        = this->final_linear_velocity_ + gravity_world * sum_dt
          + this->final_attitude_ * this->delta_linear_velocity_;
    this->final_attitude_
        = (this->final_attitude_ * this->delta_attitude_).normalized();
  }
};
