#ifndef IMU_KINEMATICS_HPP
#define IMU_KINEMATICS_HPP

#include <Eigen/Dense>

#include "euroc_vio/SensorState.hpp"

struct ImuKinematicsParameters
{
  // 已知量
  // 1. 传感器参考系下测得的加速度和角速度
  Eigen::Vector3d a0, a1; // 上一帧和当前帧的【已扣除零偏】加速度 (机体坐标系)
  Eigen::Vector3d w0, w1; // 上一帧和当前帧的【已扣除零偏】角速度 (机体坐标系)
  double t0, t1;          // 对应的时间戳
  // 2. 在惯性参考系下的重力加速度向量
  Eigen::Vector3d g_i{0.0, 0.0, -9.81};
  // 3. 从载具参考系到传感器参考系的旋转矩阵 = 单位矩阵
  // Eigen::Quaterniond C_sv;
  // 4. 传感器参考系的原点在载具参考系下的平移向量 = 零向量
  // Eigen::Vector3d r_sv_v;

  ImuKinematicsParameters(const Eigen::Vector3d &accel0,
                          const Eigen::Vector3d &accel1,
                          const Eigen::Vector3d &gyro0,
                          const Eigen::Vector3d &gyro1, double time0,
                          double time1) :
    a0{accel0}, a1{accel1}, w0{gyro0}, w1{gyro1}, t0{time0}, t1{time1}
  {
  }
};

struct ImuKinematicsODE
{
  ImuKinematicsParameters params_;

  ImuKinematicsODE(const ImuKinematicsParameters &params) : params_{params} {}
  ImuKinematicsODE(ImuKinematicsParameters &&params) : params_{params} {}

  void operator()(const ImuState &x, ImuDerivative &dxdt, const double t) const
  {
    // 采用一阶线性插值作为保持器
    const double alpha{
        (params_.t1 > params_.t0)
            ? std::clamp((t - params_.t0) / (params_.t1 - params_.t0), 0.0, 1.0)
            : 0.0};

    // 传感器参考系下的加速度
    const Eigen::Vector3d lin_acc_sensor{params_.a0
                                         + (params_.a1 - params_.a0) * alpha};
    // 传感器参考系下的角速度
    const Eigen::Vector3d ang_vel_sensor{params_.w0
                                         + (params_.w1 - params_.w0) * alpha};

    // 提取当前姿态四元数 (载体参考系到惯性参考系的旋转 C_is)
    const Eigen::Quaterniond att_world{x.GetAttitude()};

    // 惯性参考系下的线速度
    const Eigen::Vector3d lin_vec_world{x.GetVelocity()};

    //    EuRoC MAV 数据集中 IMU 传感器参考系与载体参考系重合
    //    即 C_sv = 1, C_si = C_vi，r_sv_v = 0
    //    因此加速度转换公式简化为：
    //       accMeasured = C_si * (accInInertialFrame - gravityInInertialFrame)
    //       accInInertialFrame = C_is * accMeasured + gravityInInertialFrame
    const Eigen::Vector3d lin_acc_world{att_world * lin_acc_sensor
                                        + params_.g_i};

    const Eigen::Quaterniond half_omega_sensor{0.0, 0.5 * ang_vel_sensor.x(),
                                               0.5 * ang_vel_sensor.y(),
                                               0.5 * ang_vel_sensor.z()};
    const Eigen::Quaterniond att_derivative_world{att_world
                                                  * half_omega_sensor};

    // 位置导数 = 速度
    dxdt.SetVelocity(lin_vec_world);

    // 速度导数 = 加速度
    dxdt.SetAcceleration(lin_acc_world);

    // 朝向导数 = 0.5 * 朝向 ** 角速度
    dxdt.SetAttitudeDerivative(att_derivative_world);
  }
};

#endif /* IMU_KINEMATICS_HPP */
