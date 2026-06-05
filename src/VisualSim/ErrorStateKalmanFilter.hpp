#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <meta>
#include <print>
#include <stdexcept>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "euroc_vio/Interpolation.hpp"

/**
 * @brief 基于松耦合的误差状态卡尔曼滤波，为短航程无人机提供姿态解算功能
 */
template <typename value_type = double>
class ErrorStateKalmanFilter
{
#pragma region 公有类型

public:
  using Vector3 = Eigen::Vector<value_type, 3>;
  using Matrix3 = Eigen::Matrix<value_type, 3, 3>;

  // 朝向的四元数形式 (仅用于存储，不用于运算)
  using Quaternion = Eigen::Quaternion<value_type>;
  // 朝向的李群形式
  using Attitude = Sophus::SO3<value_type>;

  // 名义状态变量 (记作 $x$)
  struct NominalStateVariable
  {
    // 位置 $r^{iv}_i$
    Vector3 position_{Vector3::Zero()};
    // 线速度 $\dot{r}^{iv}_i$
    Vector3 linear_velocity_{Vector3::Zero()};
    // 朝向 $C_{iv}$
    Quaternion attitude_{Quaternion::Identity()};
    // 加速度计零偏
    Vector3 accelerometer_bias_{Vector3::Zero()};
    // 陀螺仪零偏
    Vector3 gyroscope_bias_{Vector3::Zero()};
  };

  // 误差状态变量 (指真实状态与估计状态之差，即 $\delta x = x_true - x_est$)
  struct ErrorStateVariable
  {
    // 位置误差
    Vector3 position_error_{Vector3::Zero()};
    // 线速度误差
    Vector3 linear_velocity_error_{Vector3::Zero()};
    // 旋转误差的轴角表示
    Vector3 rotation_error_{Vector3::Zero()};
    // 加速度计零偏误差
    Vector3 accelerometer_bias_error_{Vector3::Zero()};
    // 陀螺仪零偏误差
    Vector3 gyroscope_bias_error_{Vector3::Zero()};
  };

  // 目前不考虑将重力加速度加入状态空间!

  struct DatumImuImpl
  {
    // 时间戳 (单位: 纳秒)
    std::int64_t timestamp_;
    // 角速度向量 (单位: rad s^-1)
    Vector3 angular_velocity_;
    // 线加速度向量 (单位: m s^-2)
    Vector3 linear_acceleration_;
  };

  struct DatumFastImpl
  {
    // 时间戳 (单位: 纳秒)
    std::int64_t timestamp_;
    // 角位移向量 (单位: rad)
    Vector3 angular_displacement_;
    // 单位化平移向量 (无单位)
    Vector3 normalized_translation_;
  };

#pragma endregion

#pragma region 私有类型

private:
  // 具体取值由 DatumFastImpl 中的成员变量的种类及数量决定
  static constexpr int dimMonocularData{2 * 3};

  // 具体取值由 ErrorStateVariable 中的成员变量的种类及数量决定
  static constexpr int dimErrorState{5 * 3};

  // 实际参与运算的误差状态向量
  using ErrorStateImpl = Eigen::Vector<value_type, dimErrorState>;
  // 误差状态的协方差矩阵
  using CovarianceErrorState
      = Eigen::Matrix<value_type, dimErrorState, dimErrorState>;
  // 观测雅可比矩阵
  using JacobiMeasurement
      = Eigen::Matrix<value_type, dimMonocularData, dimErrorState>;

#pragma endregion

#pragma region 初始化

public:
  ErrorStateKalmanFilter() : last_imu_time_{-1}
  {
    // 初始协方差设定为相对保守的不确定性，避免系统初试积分发生剧烈波动
    error_state_covariance_ = CovarianceErrorState::Identity() * 1e-4;
  }

  /**
   * @brief 设置初始朝向
   * @note 调用者必须在启动 ESKF 以前，首先对静止状态下的无人机校准重力方向，估计初始朝向。
   * @note 特别是 EuRoC MAV 数据集，在它提供的数据中，
   *       静止状态下的无人机的 IMU 传感器参考系的坐标轴与世界参考系的坐标轴不是平行的。
   *       IMU 参考系的 X 轴正方向与世界参考系的 Z 轴正方向近似同向；
   *       IMU 参考系的 Y 轴正方向与世界参考系的 Y 轴负方向近似同向；
   *       IMU 参考系的 Z 轴正方向与世界参考系的 X 轴正方向近似同向。
   *       从 ESKF 的角度来看，无法保证重力加速度方向是沿 IMU 参考系的 Z 轴负方向的。
   */
  void AttitudeUpdate(const Attitude &attitude)
  {
    nominal_state_.attitude_ = attitude.unit_quaternion();
    visual_attitude_         = attitude.unit_quaternion();
    is_initialized_          = true;
  }

  void SetGyroscopeNoiseDensity(value_type noise_density)
  {
    gyroscope_noise_density_ = noise_density;
  }

  void SetGyroscopeRandomWalk(value_type random_walk)
  {
    gyroscope_random_walk_ = random_walk;
  }

  void SetAccelerometerNoiseDensity(value_type noise_density)
  {
    accelerometer_noise_density_ = noise_density;
  }

  void SetAccelerometerRandomWalk(value_type random_walk)
  {
    accelerometer_random_walk_ = random_walk;
  }

  /**
   * @brief 手动初始化名义状态，用于外部更精准地赋予初值
   */
  void SetNominalState(const NominalStateVariable &state)
  {
    nominal_state_   = state;
    visual_position_ = state.position_;
    visual_attitude_ = state.attitude_;
    is_initialized_  = true;
  }

#pragma endregion

#pragma region 状态更新与观测更新

public:
  /**
   * @brief 每当收到新的 IMU 数据时调用
   * @param imu_data IMU 数据提供的角速度向量和线加速度向量
   * @note 调用者必须保证 IMU 数据是在“体坐标系”下的表示
   */
  void ImuUpdate(const DatumImuImpl *imu_data)
  {
    if (!is_initialized_)
    {
      return;
    }

    if (last_imu_time_ == -1 || last_imu_time_ == 0)
    {
      last_imu_time_ = imu_data->timestamp_;
      last_imu_data_ = *imu_data;
      return;
    }

    double dt
        = static_cast<double>(imu_data->timestamp_ - last_imu_time_) * 1e-9;
    if (dt <= 0.0)
    {
      return;
    }

    // 中值积分获取去偏差后的平均角速度与加速度
    Vector3 unbias_gyro_prev
        = last_imu_data_.angular_velocity_ - nominal_state_.gyroscope_bias_;
    Vector3 unbias_gyro_curr
        = imu_data->angular_velocity_ - nominal_state_.gyroscope_bias_;
    Vector3 omega_m = 0.5 * (unbias_gyro_prev + unbias_gyro_curr);

    Vector3 unbias_acc_prev = last_imu_data_.linear_acceleration_
                              - nominal_state_.accelerometer_bias_;
    Vector3 unbias_acc_curr
        = imu_data->linear_acceleration_ - nominal_state_.accelerometer_bias_;
    Vector3 acc_m = 0.5 * (unbias_acc_prev + unbias_acc_curr);

    // 更新名义状态中的朝向
    Attitude R_old(nominal_state_.attitude_);
    Attitude R_new           = R_old * Attitude::exp(omega_m * dt);
    nominal_state_.attitude_ = R_new.unit_quaternion();

    // 更新名义状态中的位置与线速度 (积分转换至世界系)
    Vector3 acc_world_prev = R_old * unbias_acc_prev + gravity_world;
    Vector3 acc_world_curr = R_new * unbias_acc_curr + gravity_world;
    Vector3 acc_world_m    = 0.5 * (acc_world_prev + acc_world_curr);

    nominal_state_.position_
        += nominal_state_.linear_velocity_ * dt + 0.5 * acc_world_m * dt * dt;
    nominal_state_.linear_velocity_ += acc_world_m * dt;

    // 离散系统状态转移矩阵 F (15x15) 的快速构建
    CovarianceErrorState F = CovarianceErrorState::Identity();

    // F_pv = I * dt
    F.template block<3, 3>(0, 3) = Matrix3::Identity() * dt;

    // F_vtheta = -R_old * skew(acc_m) * dt
    Matrix3 skew_acc             = SkewSymmetric(acc_m);
    F.template block<3, 3>(3, 6) = -R_old.matrix() * skew_acc * dt;

    // F_vba = -R_old * dt
    F.template block<3, 3>(3, 9) = -R_old.matrix() * dt;

    // F_thetatheta = I - skew(omega_m) * dt
    Matrix3 skew_omega           = SkewSymmetric(omega_m);
    F.template block<3, 3>(6, 6) = Matrix3::Identity() - skew_omega * dt;

    // F_thetabg = -I * dt
    F.template block<3, 3>(6, 12) = -Matrix3::Identity() * dt;

    // 系统过程噪声协方差矩阵 Q (15x15)
    CovarianceErrorState Q = CovarianceErrorState::Zero();
    double var_v
        = (accelerometer_noise_density_ * accelerometer_noise_density_) * dt;
    double var_theta
        = (gyroscope_noise_density_ * gyroscope_noise_density_) * dt;
    double var_ba
        = (accelerometer_random_walk_ * accelerometer_random_walk_) * dt;
    double var_bg = (gyroscope_random_walk_ * gyroscope_random_walk_) * dt;

    Q.template block<3, 3>(3, 3)   = var_v * Matrix3::Identity();
    Q.template block<3, 3>(6, 6)   = var_theta * Matrix3::Identity();
    Q.template block<3, 3>(9, 9)   = var_ba * Matrix3::Identity();
    Q.template block<3, 3>(12, 12) = var_bg * Matrix3::Identity();

    // 协方差的离散时间递推更新
    error_state_covariance_ = F * error_state_covariance_ * F.transpose() + Q;

    last_imu_time_ = imu_data->timestamp_;
    last_imu_data_ = *imu_data;
  }

  /**
   * @brief 每当收到新的单目视觉数据时调用
   * @param monocular_data 单目视觉数据提供的角位移向量和单位化平移向量
   * @note 调用者必须保证单目视觉数据是在“体坐标系”下的表示
   */
  void MonocularUpdate(const DatumFastImpl *monocular_data)
  {
    if (!is_initialized_)
    {
      return;
    }

    // 视觉帧相对增量积分，计算连续的全局视觉名义参考位置与姿态
    const Eigen::Quaternion<value_type> delta_rotation{
        Eigen::AngleAxis<value_type>{
            static_cast<value_type>(
                monocular_data->angular_displacement_.norm()
            ),
            monocular_data->angular_displacement_.normalized(),
        }
    };
    visual_position_
        = visual_position_
          + visual_attitude_ * monocular_data->normalized_translation_;
    visual_attitude_ = (visual_attitude_ * delta_rotation).normalized();

    // 求解 6x1 维观测残差向量 z (位置残差和李代数切空间朝向残差)
    Eigen::Vector<value_type, 6> z;
    z.template head<3>() = visual_position_ - nominal_state_.position_;
    z.template tail<3>() = (Attitude(nominal_state_.attitude_).inverse()
                            * Attitude(visual_attitude_))
                               .log();

    // 构造 6x15 维稠密卡尔曼观测雅可比阵 H
    JacobiMeasurement H          = JacobiMeasurement::Zero();
    H.template block<3, 3>(0, 0) = Matrix3::Identity();
    H.template block<3, 3>(3, 6) = Matrix3::Identity();

    // 观测协方差设定为与视觉前端相称的较高置信度 (1e-4)
    Eigen::Matrix<value_type, 6, 6> R
        = Eigen::Matrix<value_type, 6, 6>::Identity() * 1e-4;

    // 预测残差协方差阵 S
    Eigen::Matrix<value_type, 6, 6> S
        = H * error_state_covariance_ * H.transpose() + R;

    // 求解卡尔曼增益 K (15x6)
    Eigen::Matrix<value_type, dimErrorState, 6> K
        = error_state_covariance_ * H.transpose() * S.inverse();

    // 更新计算后验误差状态
    error_state_ = K * z;

    // 更新误差状态协方差并维持对称性特征
    error_state_covariance_
        = (CovarianceErrorState::Identity() - K * H) * error_state_covariance_;
    error_state_covariance_ = 0.5
                              * (error_state_covariance_
                                 + error_state_covariance_.transpose().eval());

    // 立即向标称状态进行负反馈注入复位，重置误差空间
    InjectError();
  }

#pragma endregion

#pragma region 数据接口

public:
  /**
   * @brief 获取当前名义状态
   */
  NominalStateVariable GetNominalState() const
  {
    return nominal_state_;
  }

#pragma endregion

#pragma region 私有成员函数

private:
  /**
   * @brief 将误差状态注入名义状态
   */
  void InjectError()
  {
    // 1. 更新位置: r = r + dr
    nominal_state_.position_ += error_state_.template segment<3>(0);
    // 2. 更新速度: v = v + dv
    nominal_state_.linear_velocity_ += error_state_.template segment<3>(3);
    // 3. 更新姿态: q = q * exp(d_theta)
    Vector3 d_theta = error_state_.template segment<3>(6);
    nominal_state_.attitude_
        = (Attitude(nominal_state_.attitude_) * Attitude::exp(d_theta))
              .unit_quaternion();
    // 4. 更新零偏: b = b + db
    nominal_state_.accelerometer_bias_ += error_state_.template segment<3>(9);
    nominal_state_.gyroscope_bias_ += error_state_.template segment<3>(12);

    // 5. 根据 Sola 理论，执行协方差重置 (G 映射矩阵校正)
    CovarianceErrorState G = CovarianceErrorState::Identity();
    G.template block<3, 3>(6, 6)
        = Matrix3::Identity() - 0.5 * SkewSymmetric(d_theta);
    error_state_covariance_ = G * error_state_covariance_ * G.transpose();

    // 6. 重置 error_state_ 为零
    error_state_.setZero();
  }

  /**
   * @brief 快速构建反对称矩阵
   */
  static Matrix3 SkewSymmetric(const Vector3 &v)
  {
    Matrix3 m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
  }

#pragma endregion

#pragma region 私有成员变量

private:
  // 重力加速度大小 (单位: m s^-2)
  value_type gravity_world_norm{9.81};
  // 世界坐标系下的重力加速度
  Vector3 gravity_world{-gravity_world_norm * Vector3::UnitZ()};
  // 对获取的第一帧 IMU 数据作特殊处理
  bool is_initialized_{false};
  // 名义状态 (由于 15 维向量表示四元数不便，直接使用 NominalStateVariable 实例)
  NominalStateVariable nominal_state_{};
  // 误差状态
  ErrorStateImpl error_state_{ErrorStateImpl::Zero()};
  // 陀螺仪白噪声密度 (单位: rad / s / sqrt(Hz))
  value_type gyroscope_noise_density_{0.0};
  // 陀螺仪零偏随机游走 (单位: rad / s^2 / sqrt(Hz))
  value_type gyroscope_random_walk_{0.0};
  // 加速度计白噪声密度 (单位: m / s^2 / sqrt(Hz))
  value_type accelerometer_noise_density_{0.0};
  // 加速度计零偏随机游走 (单位: m / s^3 / sqrt(Hz))
  value_type accelerometer_random_walk_{0.0};
  // 加速度计噪声
  Vector3 accelerometer_noise_{Vector3::Zero()};
  // 陀螺仪噪声
  Vector3 gyroscope_noise_{Vector3::Zero()};
  // 加速度计零偏噪声
  Vector3 accelerometer_bias_noise_{Vector3::Zero()};
  // 陀螺仪零偏噪声
  Vector3 gyroscope_bias_noise_{Vector3::Zero()};
  // 误差状态协方差矩阵
  CovarianceErrorState error_state_covariance_{
      CovarianceErrorState::Identity()
  };
  // 上一帧 IMU 时间戳
  std::int64_t last_imu_time_;
  // 视觉位置与朝向的增量积分状态变量
  Vector3 visual_position_{Vector3::Zero()};
  Quaternion visual_attitude_{Quaternion::Identity()};
  // 上一帧 IMU 数据的缓存结构
  DatumImuImpl last_imu_data_{};

#pragma endregion
};
