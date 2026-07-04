#pragma once

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <format>
#include <meta>
#include <print>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "euroc_vio/Interpolation.hpp"

#include "DatumFast.hpp"
#include "DatumImu.hpp"

// 选择“只使用角位移”还是“使用角位移和平移方向”
#define ONLY_USE_ANGULAR_DISPLACEMENT 1

namespace VisualSim
{

/**
 * @brief 基于松耦合的误差状态卡尔曼滤波，为短航程无人机提供姿态解算功能
 */
template <typename value_type = double>
class ErrorStateKalmanFilter
{
#pragma region PUBLIC_TYPE

public:
  using Vector3  = Eigen::Vector<value_type, 3>;
  using Vector6  = Eigen::Vector<value_type, 6>;
  using Matrix3  = Eigen::Matrix<value_type, 3, 3>;
  using Matrix2X = Eigen::Matrix<value_type, 2, Eigen::Dynamic>;

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
    // 重力加速度
    Vector3 gravity_{-Vector3::UnitZ()};
  };

  // 误差状态变量 (指真实状态与估计状态之差，即 $\delta x = x_true - x_est$)
  struct ErrorStateVariable
  {
    // 位置误差
    Vector3 position_error_{Vector3::Zero()};
    // 线速度误差
    Vector3 linear_velocity_error_{Vector3::Zero()};
    // 局部旋转误差的轴角表示 $\delta\theta$
    // 满足 $\delta R = \Exp(\hat{\delta\theta})$ 和 $R_t = R \delta R$
    Vector3 rotation_error_{Vector3::Zero()};
    // 加速度计零偏误差
    Vector3 accelerometer_bias_error_{Vector3::Zero()};
    // 陀螺仪零偏误差
    Vector3 gyroscope_bias_error_{Vector3::Zero()};
    // 重力加速度误差
    Vector3 gravity_error_{Vector3::Zero()};
  };

  struct Config
  {
    value_type imu_rate_{200.0};         // Hz
    value_type max_sensor_delay_{35.0};  // milliseconds
    value_type max_sensor_jitter_{10.0}; // milliseconds
    int history_buffer_margin_{16};
    using ProjectionMatrix = Eigen::Matrix<value_type, 3, 4>;
    // 经过立体矫正后，左目相机的 3x4 投影矩阵
    ProjectionMatrix proj_left_;
    // 经过立体矫正后，右目相机的 3x4 投影矩阵
    ProjectionMatrix proj_right_;
  };

#pragma endregion

#pragma region PRIVATE_TYPE

private:
  static constexpr int dimMonocularData{(ONLY_USE_ANGULAR_DISPLACEMENT)
                                            ? 3
                                            : 2 * 3};

  static constexpr int dimErrorState{6 * 3};

  // 实际参与运算的误差状态向量
  using ErrorStateImpl = Eigen::Vector<value_type, dimErrorState>;
  // 状态转移矩阵
  using TransitionMatrix
      = Eigen::Matrix<value_type, dimErrorState, dimErrorState>;
  // 观测函数的雅可比矩阵
  using JacobiMeasurement
      = Eigen::Matrix<value_type, dimMonocularData, dimErrorState>;
  // 观测误差的协方差矩阵
  using CovarianceMeasurement
      = Eigen::Matrix<value_type, dimMonocularData, dimMonocularData>;
  // 卡尔曼增益矩阵
  using KalmanGain = Eigen::Matrix<value_type, dimErrorState, dimMonocularData>;

  struct HistoryState
  {
    // 原始 IMU 数据 (未作零偏矫正)
    DatumImu imu_;
    // 名义状态变量
    NominalStateVariable nomial_;
    // 过程噪声的协方差矩阵
    TransitionMatrix P_;
  };

  struct HistoryBuffer : public std::vector<HistoryState>
  {
    std::size_t history_head_;
    std::size_t history_size_;
  };

#pragma endregion

#pragma region INITIALIZATION

public:
  /**
   * @brief 构造函数。
   */
  ErrorStateKalmanFilter(const Config &config) : config_{config}
  {
    assert(error_state_covariance_.allFinite());
  }

  /**
   * @brief 设置陀螺仪高斯白噪声密度。
   * @param noise_density 噪声密度值。
   */
  void SetGyroscopeNoiseDensity(value_type noise_density) noexcept
  {
    gyroscope_noise_density_ = noise_density;
    std::print(stderr, "[DEBUG] 修改陀螺仪噪声密度 <- {:.6f}\n", noise_density);
  }

  /**
   * @brief 设置陀螺仪随机游走偏置噪声。
   * @param random_walk 零偏随游常数。
   */
  void SetGyroscopeRandomWalk(value_type random_walk) noexcept
  {
    gyroscope_random_walk_ = random_walk;
    std::print(stderr, "[DEBUG] 修改陀螺仪随机游走 <- {:.6f}\n", random_walk);
  }

  /**
   * @brief 设置加速度计高斯白噪声密度。
   * @param noise_density 加速度计白噪声密度值。
   */
  void SetAccelerometerNoiseDensity(value_type noise_density) noexcept
  {
    accelerometer_noise_density_ = noise_density;
    std::print(stderr, "[DEBUG] 修改加速度计噪声密度 <- {:.6f}\n",
               noise_density);
  }

  /**
   * @brief 设置加速度计零偏随机游走。
   * @param random_walk 零偏随游参数。
   */
  void SetAccelerometerRandomWalk(value_type random_walk) noexcept
  {
    accelerometer_random_walk_ = random_walk;
    std::print(stderr, "[DEBUG] 修改加速度计随机游走 <- {:.6f}\n", random_walk);
  }

  /**
   * @brief 手动初始化名义状态，用于外部更精准地赋予初值
   * @param state 输入的名义状态
   * @note 调用者必须在启动 ESKF 以前，首先对静止状态下的无人机校准重力方向，估计初始朝向。
   * @note 特别是 EuRoC MAV 数据集，在它提供的数据中，
   *       静止状态下的无人机的 IMU 传感器参考系的坐标轴与世界参考系的坐标轴不是平行的。
   *       IMU 参考系的 X 轴正方向与世界参考系的 Z 轴正方向近似同向；
   *       IMU 参考系的 Y 轴正方向与世界参考系的 Y 轴负方向近似同向；
   *       IMU 参考系的 Z 轴正方向与世界参考系的 X 轴正方向近似同向。
   *       从 ESKF 的角度来看，无法保证重力加速度方向是沿 IMU 参考系的 Z 轴负方向的。
   */
  void SetNominalState(const NominalStateVariable &state) noexcept
  {
    // 如果初始坐标系是水平的，我们可以将其初始化为 g = [0, 0, -9.81]
    // 但如果初始时刻并非水平，我们可以选择将初始姿态初始化为单位阵
    // 并让初始重力向量 g 来承担关于初始姿态的所有不确定性
    // 这种处理方式的好处在于提高了系统的线性度
    // 在初始姿态未知的情况下，处理一个未知的重力向量（已知旋转）
    // 比处理一个未知的旋转（已知重力）在线性化效果上更好
    const Vector3 gravity_body{state.attitude_.conjugate() * state.gravity_};
    const Vector3 linear_velocity_body{state.attitude_.conjugate()
                                       * state.linear_velocity_};
    nominal_state_.position_           = Vector3::Zero();
    nominal_state_.linear_velocity_    = linear_velocity_body;
    nominal_state_.attitude_           = Quaternion::Identity();
    nominal_state_.accelerometer_bias_ = Vector3::Zero();
    nominal_state_.gyroscope_bias_     = Vector3::Zero();
    nominal_state_.gravity_            = gravity_body;
    prev_position_                     = nominal_state_.position_;
    prev_attitude_                     = nominal_state_.attitude_;
  }

#pragma endregion

#pragma region STATE_UPDATE

public:
  /**
   * @brief 每当收到新 IMU 周期采样信号时，更新名义状态与误差状态协方差。
   * @param imu_data IMU 数据提供的角速度向量和线加速度向量
   * @note 调用者必须保证 IMU 数据是在“体坐标系”下的表示
   */
  void ImuUpdate(const DatumImu *imu_data) noexcept
  {
#pragma region FOOL_PROOF

    if (imu_data == nullptr)
    {
      return;
    }

    if (last_imu_time_ < 0)
    {
      last_imu_time_ = imu_data->timestamp_;
      last_imu_data_ = *imu_data;
      return;
    }

    const value_type dt{
        static_cast<value_type>((imu_data->timestamp_ - last_imu_time_) * 1e-9)
    };
    if (dt <= 0.0)
    {
      return;
    }

#pragma endregion

#pragma region WORLD_ANGULAR_VELOCITY_AND_LINEAR_ACCELERATION

    // 假设上一帧 IMU 数据 `last_imu_data_` 中的角速度、线加速度都已经去除零偏
    Vector3 unbias_gyro_prev{last_imu_data_.angular_velocity_};
    Vector3 unbias_acc_prev{last_imu_data_.linear_acceleration_};

    // 当前帧 IMU 数据 imu_data 去除零偏
    Vector3 unbias_gyro_curr{imu_data->angular_velocity_
                             - nominal_state_.gyroscope_bias_};
    Vector3 unbias_acc_curr{imu_data->linear_acceleration_
                            - nominal_state_.accelerometer_bias_};
    // 保存当前帧 IMU 数据
    last_imu_data_ = DatumImu{
        imu_data->timestamp_,
        unbias_gyro_curr,
        unbias_acc_curr,
    };

    Vector3 omega_m{static_cast<value_type>(0.5)
                    * (unbias_gyro_prev + unbias_gyro_curr)};

    Vector3 acc_m{static_cast<value_type>(0.5)
                  * (unbias_acc_prev + unbias_acc_curr)};

    Attitude R_old{nominal_state_.attitude_};
    Attitude R_new{
        R_old
        * Attitude::exp(omega_m * dt
                        + (dt * dt / 12.0)
                              * unbias_gyro_prev.cross(unbias_gyro_curr))
    };

    Vector3 acc_world_m{
        static_cast<value_type>(0.5)
                * (R_old * unbias_acc_prev + R_new * unbias_acc_curr)
            + nominal_state_.gravity_,
    };

#pragma endregion

#pragma region UPDATE_NOMINAL_STATE

    Vector3 delta_velocity{acc_world_m * dt};
    Vector3 delta_position{
        (nominal_state_.linear_velocity_
         + static_cast<value_type>(0.5) * delta_velocity)
            * dt,
    };

    nominal_state_.position_ += delta_position;
    nominal_state_.linear_velocity_ += delta_velocity;
    nominal_state_.attitude_ = R_new.unit_quaternion();

#pragma endregion

#pragma region UPDATE_ERROR_STATE_COVARIANCE

    // 离散系统状态转移矩阵 $F_x$
    TransitionMatrix Fx{prediction_create_transition(dt, R_old.matrix(),
                                                     omega_m, acc_m)};

    TransitionMatrix Q{prediction_create_covariance(dt)};

    // 协方差的离散时间递推更新
    TransitionMatrix new_error_state_covariance{
        Fx * error_state_covariance_ * Fx.transpose() + Q
    };

    assert(new_error_state_covariance.allFinite());

    error_state_covariance_ = new_error_state_covariance;

#pragma endregion

    last_imu_time_ = imu_data->timestamp_;

    // TODO 在这里保存历史状态
  }

  /**
   * @brief 每当收到新视觉观测周期采样信号时，执行卡尔曼观测融合及误差校正。
   * @param monocular_data 单目视觉数据提供的角位移向量和单位化平移向量
   * @note 调用者必须保证单目视觉数据是在“体坐标系”下的表示
   */
  void MonocularUpdate(const DatumFast *monocular_data) noexcept
  {
#pragma region FOOL_PROOF

    if (monocular_data == nullptr)
    {
      return;
    }
    if (last_cam_time_ < 0)
    {
      last_cam_time_ = monocular_data->timestamp_;
      return;
    }

    const value_type dt{static_cast<value_type>(
        (monocular_data->timestamp_ - last_cam_time_) * 1e-9
    )};
    if (dt <= 0.0)
    {
      return;
    }

#pragma endregion

    // 由 IMU 数据计算得到的角位移
    Attitude d_attitude{
        // 因为在体坐标系下，$q_t = q_0 \otimes \delta q$
        // 所以 $\delta q = q_0^* \otimes q_t$
        (prev_attitude_.conjugate() * nominal_state_.attitude_).normalized()
    };
    Vector3 angular_displacement{d_attitude.log()};

    JacobiMeasurement H{measurement_jacobian(angular_displacement)};
    CovarianceMeasurement V{covariance_measurement()};
    KalmanGain K{kalman_gain(error_state_covariance_, H, V)};

    // 观测残差向量

#if (ONLY_USE_ANGULAR_DISPLACEMENT)
    Vector3 measurement_residue{monocular_data->angular_displacement_
                                - angular_displacement};
#else
    // 由 IMU 数据计算得到的平移方向
    Vector3 delta_position{
        nominal_state_.position_ - prev_position_,
    };
    value_type delta_position_norm{delta_position.norm()};
    Vector3 normalized_translation{Vector3::Zero()};
    if (delta_position_norm > static_cast<value_type>(1e-6))
    {
      normalized_translation = delta_position / delta_position_norm;
    }

    Vector6 measurement_residue;
    measurement_residue.template head<3>()
        = monocular_data->angular_displacement_ - angular_displacement;
    measurement_residue.template tail<3>()
        = monocular_data->normalized_translation_ - normalized_translation;
#endif

    prev_attitude_ = nominal_state_.attitude_;
    prev_position_ = nominal_state_.position_;

    error_state_ = K * measurement_residue;

    measurement_update_covariance(error_state_covariance_, K, H, V);

    // 立即向标称状态进行负反馈注入复位，重置误差空间
    InjectError();
    last_cam_time_ = monocular_data->timestamp_;
  }

  void StereoUpdate(std::int64_t timestamp,
                    std::span<const uint32_t> feature_ids,
                    const Matrix2X &pts_left,
                    const Matrix2X &pts_right) noexcept
  {
    assert(pts_left.cols() == pts_right.cols()
           && "pts_left.cols() != pts_right.cols()");
  }

#pragma endregion

#pragma region DATA_INTERFACE

public:
  /**
   * @brief 获取当前系统的名义状态。
   * @return NominalStateVariable 名义状态变量。
   */
  NominalStateVariable GetNominalState() const noexcept
  {
    return nominal_state_;
  }

#pragma endregion

#pragma region PRIVATE_METHOD

private:
  /**
   * @brief 将误差状态注入名义状态，执行名义状态校正。
   */
  void InjectError() noexcept
  {
    // 更新位置: r = r + dr
    nominal_state_.position_ += error_state_.template segment<3>(0);
    // 更新速度: v = v + dv
    nominal_state_.linear_velocity_ += error_state_.template segment<3>(3);
    // 更新姿态: q = q * exp(d_theta)
    auto d_theta{error_state_.template segment<3>(6)};
    nominal_state_.attitude_
        = (Attitude{nominal_state_.attitude_} * Attitude::exp(d_theta))
              .unit_quaternion();
    // 更新零偏: b = b + db
    nominal_state_.accelerometer_bias_ += error_state_.template segment<3>(9);
    nominal_state_.gyroscope_bias_ += error_state_.template segment<3>(12);
    // 更新重力加速度
    nominal_state_.gravity_ += error_state_.template segment<3>(15);

    ResetESKF();
  }

  /**
   * @brief 注入后对误差状态和相关的协方差矩阵进行重置处理。
   */
  void ResetESKF() noexcept
  {
    TransitionMatrix G{TransitionMatrix::Identity()};
    auto d_theta{error_state_.template segment<3>(6)};
    G.template block<3, 3>(6, 6)
        = Matrix3::Identity()
          - Attitude::hat(static_cast<value_type>(0.5) * d_theta).matrix();
    // 更新误差状态协方差矩阵
    TransitionMatrix new_error_state_covariance{G * error_state_covariance_
                                                * G.transpose()};

    assert(new_error_state_covariance.allFinite());

    error_state_covariance_ = new_error_state_covariance;

    // 重置 error_state_ 为零
    error_state_.setZero();
  }

  /**
   * @brief 计算离散系统状态转移矩阵
   * @param dt 时间步长
   * @param R 旋转矩阵
   * @param gyro 角速度
   * @param acc 线加速度
   */
  TransitionMatrix prediction_create_transition(value_type dt, const Matrix3 &R,
                                                const Vector3 &gyro,
                                                const Vector3 &acc) noexcept
  {
    TransitionMatrix Fx{TransitionMatrix::Identity()};
    // F_p_v = I * dt
    Fx.template block<3, 3>(0, 3) = Matrix3::Identity() * dt;
    // F_v_theta
    Fx.template block<3, 3>(3, 6) = -R * Attitude::hat(acc).matrix() * dt;
    // F_v_ba = -R * dt
    Fx.template block<3, 3>(3, 9) = -R * dt;
    // F_v_g
    Fx.template block<3, 3>(3, 15) = Matrix3::Identity() * dt;
    // F_theta_theta
    Fx.template block<3, 3>(6, 6)
        = Attitude::exp(gyro * dt).matrix().transpose();
    // F_theta_bg
    Fx.template block<3, 3>(6, 12) = -Matrix3::Identity() * dt;
    return Fx;
  }

  /**
   * @brief 构造并传播连续系统噪声分布转换至离散协方差矩阵。
   * @param dt 时间步长
   * @return 系统过程噪声协方差矩阵
   * @note 等于 (状态转移函数对扰动的雅可比矩阵) * (扰动脉冲协方差矩阵) * (状态转移函数对扰动的雅可比矩阵).转置
   */
  TransitionMatrix prediction_create_covariance(value_type dt) noexcept
  {
    TransitionMatrix Q{TransitionMatrix::Zero()};
    // 加速度计测量值噪声的方差与时间步长的乘积
    const value_type var_v{
        (accelerometer_noise_density_ * accelerometer_noise_density_) * dt,
    };
    // 陀螺仪测量值噪声的方差与时间步长的乘积
    const value_type var_theta{
        (gyroscope_noise_density_ * gyroscope_noise_density_) * dt,
    };
    // 加速度计零偏噪声的方差与时间步长的乘积
    const value_type var_ba{
        (accelerometer_random_walk_ * accelerometer_random_walk_) * dt,
    };
    // 陀螺仪零偏噪声的方差与时间步长的乘积
    const value_type var_bg{
        (gyroscope_random_walk_ * gyroscope_random_walk_) * dt,
    };
    // 将 $V_i$, $\Theta_i$, $A_i$, $\Omega_i$ 填入系统过程噪声协方差矩阵中
    template for (int i = 1;
                  value_type var : {var_v, var_theta, var_ba, var_bg})
    {
      Q.template block<3, 3>(3 * i, 3 * i) = var * Matrix3::Identity();
      ++i;
    }
    return Q;
  }

  /**
   * @brief 计算测量函数的雅可比矩阵
   * @param angular_displacement 利用 IMU 数据估计得到的相邻两个图像帧间的角位移
   */
  JacobiMeasurement
  measurement_jacobian(const Vector3 &angular_displacement) const noexcept
  {
    JacobiMeasurement result{JacobiMeasurement::Zero()};

    // 角位移对旋转误差的导数
    result.template block<3, 3>(0, 6)
        = Attitude::leftJacobianInverse(-angular_displacement);

#if (!ONLY_USE_ANGULAR_DISPLACEMENT)
    auto velocity_norm{nominal_state_.linear_velocity_.norm()};
    if (velocity_norm > static_cast<value_type>(1e-6))
    {
      auto velocity_direction{nominal_state_.linear_velocity_ / velocity_norm};
      // 单位化平移向量对线速度的导数
      result.template block<3, 3>(3, 3)
          = (Matrix3::Identity()
             - velocity_direction * velocity_direction.transpose())
            / velocity_norm;
    }
#endif
    return result;
  }

  /**
   * @brief 获取测量协方差矩阵。
   * @note 置信度数值越小，可以相信的程度越高
   */
  CovarianceMeasurement covariance_measurement() const noexcept
  {
    CovarianceMeasurement V{CovarianceMeasurement::Identity()};
#if (ONLY_USE_ANGULAR_DISPLACEMENT)
    // 设置[角位移估计量]的置信度
    V.diagonal().setConstant(confidence_angular_displacement_);
#else
    // 设置[角位移估计量]的置信度
    V.diagonal().template segment<3>(0).setConstant(
        confidence_angular_displacement_
    );
    // 设置[平移方向估计量]的置信度
    V.diagonal().template segment<3>(3).setConstant(
        confidence_normalized_translation_
    );
#endif
    return V;
  }

  /**
   * @brief 求解卡尔曼增益。
   * @param P 过程噪声的协方差矩阵
   * @param H 测量函数的雅可比矩阵
   * @param V 测量噪声的协方差矩阵
   */
  static KalmanGain kalman_gain(const TransitionMatrix &P,
                                const JacobiMeasurement &H,
                                const CovarianceMeasurement &V) noexcept
  {
    auto hphv{H * P * H.transpose() + V};
    return hphv.ldlt().solve(H * P.transpose()).transpose();
    // return P * H.transpose() * hphv.inverse();
  }

  /**
   * @brief 使用 Joseph 稳定形式更新过程噪声的协方差矩阵。
   * @param P 过程噪声的协方差矩阵
   * @param K 卡尔曼增益
   * @param H 测量函数的雅可比矩阵
   * @param V 测量噪声的协方差矩阵
   */
  static void
  measurement_update_covariance(TransitionMatrix &P, const KalmanGain &K,
                                const JacobiMeasurement &H,
                                const CovarianceMeasurement &V) noexcept
  {
    TransitionMatrix ikh{TransitionMatrix::Identity() - K * H};
    P = ikh * P * ikh.transpose() + K * V * K.transpose();
  }

#pragma endregion

#pragma region PUBLIC_VARIABLE

public:
  // 单目视觉估计角位移置信度
  value_type confidence_angular_displacement_{1e-4};
  // 单目视觉估计平移方向置信度
  value_type confidence_normalized_translation_{1e-4};

#pragma endregion

#pragma region PRIVATE_VARIABLE

private:
  // ESKF 参数
  Config config_;
  // 名义状态
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
  // 误差状态协方差矩阵 $P$ (初始协方差设定为相对保守的不确定性，避免系统初试积分发生剧烈波动)
  TransitionMatrix error_state_covariance_{TransitionMatrix::Identity()
                                           * static_cast<value_type>(1e-4)};
  // 上一帧 IMU 时间戳
  std::int64_t last_imu_time_{-1};
  // 上一帧图像时间戳
  std::int64_t last_cam_time_{-1};
  // 上一帧图像帧的姿态
  Vector3 prev_position_{Vector3::Zero()};
  Quaternion prev_attitude_{Quaternion::Identity()};
  // 上一帧 IMU 数据的缓存结构
  DatumImu last_imu_data_{};
  // 历史状态缓冲区
  HistoryBuffer history_buffer_;

#pragma endregion
};

} // namespace VisualSim
