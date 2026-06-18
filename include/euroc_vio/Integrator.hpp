#pragma once

#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include "DatumImu.hpp"

struct AbstractIntegrator
{
  // 重力加速度大小
  double gravity_world_norm{9.81};
  // 重力加速度
  Eigen::Vector3d gravity_world{-gravity_world_norm * Eigen::Vector3d::UnitZ()};
  // 初始状态
  Eigen::Vector3d estimated_position{Eigen::Vector3d::Zero()};
  Sophus::SO3d estimated_attitude{};
  Eigen::Vector3d estimated_linear_velocity{Eigen::Vector3d::Zero()};
  // 先前状态
  Sophus::SO3d previous_attitude{};
};

/**
 * @brief IMU 中点法零阶朝向积分器
 */
struct ZerothOrderAttitudeIntegrator : public AbstractIntegrator
{
  void Update(const DatumImu &datum_prev, const DatumImu &datum)
  {
    const double dt{
        1e-9f * static_cast<double>(datum.timestamp_ - datum_prev.timestamp_),
    };

    // 载具参考系下的角速度 = 传感器参考系下的角速度
    // 载具参考系下前一帧角速度
    Eigen::Vector3d previous_angular_velocity_in_body_frame{
        datum_prev.angular_velocity_,
    };
    // 载具参考系下后一帧角速度
    Eigen::Vector3d current_angular_velocity_in_body_frame{
        datum.angular_velocity_,
    };
    // 载具参考系下两帧角速度的平均值
    Eigen::Vector3d median_angular_velocity_in_body_frame{
        0.5
            * (previous_angular_velocity_in_body_frame
               + current_angular_velocity_in_body_frame),
    };
    // 朝向变化量
    Sophus::SO3d delta_attitude{
        Sophus::SO3d::exp(median_angular_velocity_in_body_frame * dt),
    };
    // 新的朝向
    Sophus::SO3d estimated_new_attitude{
        estimated_attitude * delta_attitude,
    };

    // 更新朝向
    previous_attitude  = estimated_attitude;
    estimated_attitude = estimated_new_attitude;
  }
};

/**
 * @brief IMU 中点法一阶朝向积分器
 */
struct FirstOrderAttitudeIntegrator : public AbstractIntegrator
{
  void Update(const DatumImu &datum_prev, const DatumImu &datum)
  {
    const double dt{
        1e-9f * static_cast<double>(datum.timestamp_ - datum_prev.timestamp_),
    };

    // 载具参考系下的角速度 = 传感器参考系下的角速度
    // 载具参考系下前一帧角速度
    Eigen::Vector3d previous_angular_velocity_in_body_frame{
        datum_prev.angular_velocity_,
    };
    // 载具参考系下后一帧角速度
    Eigen::Vector3d current_angular_velocity_in_body_frame{
        datum.angular_velocity_,
    };
    // 载具参考系下两帧角速度的平均值
    Eigen::Vector3d median_angular_velocity_in_body_frame{
        0.5
            * (previous_angular_velocity_in_body_frame
               + current_angular_velocity_in_body_frame),
    };
    // 朝向变化量
    Sophus::SO3d delta_attitude{
        Sophus::SO3d::exp(median_angular_velocity_in_body_frame * dt)
            + (dt * dt / 12.0)
                  * previous_angular_velocity_in_body_frame.cross(
                      current_angular_velocity_in_body_frame
                  ),
    };
    // 新的朝向
    Sophus::SO3d estimated_new_attitude{
        estimated_attitude * delta_attitude,
    };

    // 更新朝向
    previous_attitude  = estimated_attitude;
    estimated_attitude = estimated_new_attitude;
  }
};

/**
 * @brief IMU 中点法位置积分器
 */
struct MidpointPositionIntegrator : public AbstractIntegrator
{
  void Update(const DatumImu &datum_prev, const DatumImu &datum)
  {
    const double dt{
        1e-9f * static_cast<double>(datum.timestamp_ - datum_prev.timestamp_),
    };

    // 惯性参考系下的线加速度
    Eigen::Vector3d linear_acceleration_in_world_frame{
        0.5
                * (previous_attitude * datum_prev.linear_acceleration_
                   + estimated_attitude * datum.linear_acceleration_)
            + gravity_world,
    };
    // 线速度变化量
    Eigen::Vector3d delta_velocity{
        linear_acceleration_in_world_frame * dt,
    };
    // 位置变化量
    Eigen::Vector3d delta_position{
        (estimated_linear_velocity + 0.5 * delta_velocity) * dt,
    };

    // 更新位置
    estimated_position += delta_position;
    // 更新线速度
    estimated_linear_velocity += delta_velocity;
  }
};

/**
 * @brief 纯视觉姿态积分器
 */
struct VisualIntegrator : public AbstractIntegrator
{
  void Update(const Sophus::SO3d &delta_attitude,
              const Eigen::Vector3d &delta_position)
  {
    // 新的朝向
    Sophus::SO3d estimated_new_attitude{
        estimated_attitude * delta_attitude,
    };

    // 更新位置
    estimated_position += estimated_attitude * delta_position;
    // 更新朝向
    previous_attitude  = estimated_attitude;
    estimated_attitude = estimated_new_attitude;
  }
};
