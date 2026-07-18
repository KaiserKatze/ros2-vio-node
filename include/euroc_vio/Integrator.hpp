#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

template <typename T>
concept ImuDatumLike = requires {
  requires std::is_same_v<std::decay_t<decltype(std::declval<T>().timestamp_)>,
                          std::int64_t>;
  requires std::
      is_same_v<std::decay_t<decltype(std::declval<T>().angular_velocity_)>,
                Eigen::Vector3d>;
  requires std::
      is_same_v<std::decay_t<decltype(std::declval<T>().linear_acceleration_)>,
                Eigen::Vector3d>;
};

namespace FastVIO
{

struct AbstractIntegrator
{
  // 重力加速度大小
  double gravity_world_norm_{9.81};
  // 重力加速度
  Eigen::Vector3d gravity_world_{-gravity_world_norm_
                                 * Eigen::Vector3d::UnitZ()};
  // 姿态 (r^{vi}_i,C_iv)
  Sophus::SE3d pose_{};
  // 线速度
  Eigen::Vector3d linear_velocity_{Eigen::Vector3d::Zero()};
  // 先前状态
  Sophus::SO3d previous_attitude_{};
};

/**
 * @brief IMU 中点法零阶朝向积分器
 */
struct ZerothOrderAttitudeIntegrator : public AbstractIntegrator
{
  void Update(const ImuDatumLike auto &datum_prev,
              const ImuDatumLike auto &datum)
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
        pose_.so3() * delta_attitude,
    };

    // 更新朝向
    previous_attitude_ = pose_.so3();
    pose_.so3()        = estimated_new_attitude;
  }
};

/**
 * @brief IMU 中点法一阶朝向积分器
 */
struct FirstOrderAttitudeIntegrator : public AbstractIntegrator
{
  void Update(const ImuDatumLike auto &datum_prev,
              const ImuDatumLike auto &datum)
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
        Sophus::SO3d::exp(median_angular_velocity_in_body_frame * dt
                          + (dt * dt / 12.0)
                                * previous_angular_velocity_in_body_frame.cross(
                                    current_angular_velocity_in_body_frame
                                )),
    };
    // 新的朝向
    Sophus::SO3d estimated_new_attitude{
        pose_.so3() * delta_attitude,
    };

    // 更新朝向
    previous_attitude_ = pose_.so3();
    pose_.so3()        = estimated_new_attitude;
  }
};

/**
 * @brief IMU 中点法位置积分器
 */
struct MidpointPositionIntegrator : public AbstractIntegrator
{
  void Update(const ImuDatumLike auto &datum_prev,
              const ImuDatumLike auto &datum)
  {
    const double dt{
        1e-9f * static_cast<double>(datum.timestamp_ - datum_prev.timestamp_),
    };

    // 惯性参考系下的线加速度
    Eigen::Vector3d linear_acceleration_in_world_frame{
        0.5
                * (previous_attitude_ * datum_prev.linear_acceleration_
                   + pose_.so3() * datum.linear_acceleration_)
            + gravity_world_,
    };
    // 线速度变化量
    Eigen::Vector3d delta_velocity{
        linear_acceleration_in_world_frame * dt,
    };
    // 位置变化量
    Eigen::Vector3d delta_position{
        (linear_velocity_ + 0.5 * delta_velocity) * dt,
    };

    // 更新位置
    pose_.translation() += delta_position;
    // 更新线速度
    linear_velocity_ += delta_velocity;
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
        pose_.so3() * delta_attitude,
    };

    // 更新位置
    pose_.translation() += pose_.so3() * delta_position;
    // 更新朝向
    previous_attitude = pose_.so3();
    pose_.so3()       = estimated_new_attitude;
  }
};

} // namespace FastVIO
