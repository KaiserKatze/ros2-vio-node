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
#include <unordered_map>

#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "euroc_vio/Interpolation.hpp"

#include "euroc_vio/DatumFast.hpp"
#include "euroc_vio/DatumImu.hpp"

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
  using Vector2  = Eigen::Vector<value_type, 2>;
  using Vector3  = Eigen::Vector<value_type, 3>;
  using Vector4  = Eigen::Vector<value_type, 4>;
  using Vector6  = Eigen::Vector<value_type, 6>;
  using Matrix3  = Eigen::Matrix<value_type, 3, 3>;
  using Matrix2X = Eigen::Matrix<value_type, 2, Eigen::Dynamic>;

  // 朝向的四元数形式 (仅用于存储，不用于运算)
  using Quaternion = Eigen::Quaternion<value_type>;
  // 朝向的李群形式
  using Attitude = Sophus::SO3<value_type>;
  // 位姿的李群形式
  using Pose = Sophus::SE3<value_type>;

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

  using ProjectionMatrix = Eigen::Matrix<value_type, 3, 4>;

  struct StereoCameraModel
  {
    // 经过立体矫正后，左目相机的 3x4 投影矩阵
    ProjectionMatrix proj_left_{ProjectionMatrix::Zero()};
    // 经过立体矫正后，右目相机的 3x4 投影矩阵
    ProjectionMatrix proj_right_{ProjectionMatrix::Zero()};
    // 经过立体矫正后，左目相机相对于体坐标系的变换矩阵 $T_{BS}$ (假设刚性不变)
    //    [ r^{pi}_i ; 1 ] = T_{BS} * [ r^{pv}_v ; 1 ]
    //    T_{BS} = [ C_{iv}, r^{vi}_i ; 0, 1 ]
    Pose transform_cam0_{};
  };

  struct Config
  {
    value_type imu_rate_{200.0};         // Hz
    value_type max_sensor_delay_{35.0};  // milliseconds
    value_type max_sensor_jitter_{10.0}; // milliseconds
    std::size_t history_buffer_margin_{16};
    StereoCameraModel stereo_camera_model_{};
  };

  struct StereoObservation
  {
    // 路标点 ID
    std::uint32_t feature_id_{0};
    // 左目图像中角点坐标
    Vector2 pt_left_{Vector2::Zero()};
    // 右目图像中角点坐标
    Vector2 pt_right_{Vector2::Zero()};
    // 左目图像中角点响应值 (FAST score)
    value_type response_left_{0.0};
    // 右目图像中角点响应值 (FAST score)
    value_type response_right_{0.0};
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
  // 单目估计的观测函数的雅可比矩阵
  using JacobiMeasurementFast
      = Eigen::Matrix<value_type, dimMonocularData, dimErrorState>;
  // 单目估计的观测误差的协方差矩阵
  using CovarianceMeasurementFast
      = Eigen::Matrix<value_type, dimMonocularData, dimMonocularData>;
  // 单目估计的卡尔曼增益矩阵
  using KalmanGainFast
      = Eigen::Matrix<value_type, dimErrorState, dimMonocularData>;
  // 投影矩阵对路标点相机坐标的雅克比矩阵
  using JacobiProjection = Eigen::Matrix<value_type, 2, 3>;
  // 双目估计的观测函数的雅可比矩阵
  using JacobiMeasurementStereo = Eigen::Matrix<value_type, 4, 4>;
  // 双目估计的观测误差的协方差矩阵
  using CovarianceMeasurementStereo = Eigen::Matrix<value_type, 4, 4>;

  struct HistoryState
  {
    // 当前帧、原始 IMU 数据 (未作零偏矫正)
    DatumImu raw_imu_datum_;
    // 上一帧、经过零偏矫正的 IMU 数据
    DatumImu last_imu_datum_;
    // 名义状态变量
    NominalStateVariable nominal_;
    // 过程噪声的协方差矩阵
    TransitionMatrix error_state_covariance_;

    auto GetTimestamp() const noexcept
    {
      return raw_imu_datum_.timestamp_;
    }
  };

  /**
   * @brief 固定容量历史状态环形缓冲区。
   *
   * @details
   * 该缓冲区用于保存 ESKF 的历史状态，以支持延时观测回滚
   * （Rollback）与重新传播（Replay）。
   */
  struct HistoryBuffer
  {
    using buffer_t       = boost::circular_buffer<HistoryState>;
    using iterator       = typename buffer_t::iterator;
    using const_iterator = typename buffer_t::const_iterator;

    buffer_t buffer_;

    /**
     * @brief 初始化固定容量历史缓冲区。
     * @param capacity 缓冲区容量。
     */
    HistoryBuffer(std::size_t capacity) : buffer_{capacity} {}

    std::size_t Size() const noexcept
    {
      return buffer_.size();
    }

    /**
     * @brief 返回缓冲区容量。
     */
    [[nodiscard]]
    std::size_t Capacity() const noexcept
    {
      return buffer_.capacity();
    }

    /**
     * @brief 判断缓冲区是否为空。
     */
    [[nodiscard]]
    bool Empty() const noexcept
    {
      return buffer_.empty();
    }

    /**
     * @brief 插入一条新的历史状态。
     *
     * @details
     * 若缓冲区已满，则自动覆盖最旧元素。
     */
    void Push(const HistoryState &state) noexcept
    {
      buffer_.push_back(state);
    }

    /**
     * @brief 根据时间戳查找历史状态。
     *
     * @param timestamp Unix 时间戳。
     * @return 找到返回对应迭代器；未找到返回 End()。
     */
    [[nodiscard]]
    iterator Find(std::int64_t timestamp) noexcept
    {
      auto it{std::ranges::lower_bound(buffer_, timestamp, std::less<>(),
                                       &HistoryState::GetTimestamp)};
      if (it == buffer_.begin())
      {
        return buffer_.begin();
      }
      if (it == buffer_.end())
      {
        return std::prev(buffer_.end());
      }
      if (it->GetTimestamp() > timestamp)
      {
        return std::prev(it);
      }
      return it;
    }

    /**
     * @brief const 版本时间戳查找。
     *
     * @param timestamp Unix 时间戳。
     * @return 找到返回对应常量迭代器；未找到返回 End()。
     */
    [[nodiscard]]
    const_iterator Find(std::int64_t timestamp) const noexcept
    {
      auto it{std::ranges::lower_bound(buffer_, timestamp, std::less<>(),
                                       &HistoryState::GetTimestamp)};
      if (it == buffer_.cbegin())
      {
        return buffer_.cbegin();
      }
      if (it == buffer_.cend())
      {
        return std::prev(buffer_.cend());
      }
      if (it->GetTimestamp() > timestamp)
      {
        return std::prev(it);
      }
      return it;
    }
  };

  enum class LandmarkStatus : std::uint8_t
  {
    // 新创建，仅完成一次三角化
    New,
    // 连续观测达到阈值，可参与 ESKF 更新
    Active,
    // 本帧未观测到
    Lost,
    // 等待从 Database 擦除
    Erased,
  };

  /**
   * @brief 地图点（Landmark）
   *
   * @details
   * 每个 Landmark 由唯一 Feature ID 标识，
   * 世界坐标在首次双目观测时初始化，
   * 后续 StereoUpdate 仅作为固定地图点参与滤波更新。
   */
  struct Landmark
  {
    // Feature ID
    std::uint32_t id_{};

    // 世界坐标（齐次坐标前三维）
    Vector3 position_{Vector3::Zero()};

    // 状态
    LandmarkStatus status_{LandmarkStatus::New};

    // 最近一次被观测到的时间戳
    std::int64_t timestamp_{-1};

    // 最近一次被观测到的帧号
    std::uint32_t last_frame_id_{0};

    // 连续丢失次数
    std::uint32_t lost_count_{0};

    // 连续观测次数
    std::uint32_t observed_count_{0};
  };

  struct LandmarkDatabase
  {
    std::unordered_map<std::uint32_t, Landmark> landmarks_;
  };

  struct FeatureTrack
  {
    std::uint32_t id_{};
    Vector2 left_{Vector2::Zero()};
    Vector2 right_{Vector2::Zero()};
    std::int64_t timestamp_{};
    bool stereo_valid_{false};
  };

  struct FeatureTrackTable
  {
    std::unordered_map<std::uint32_t, FeatureTrack> tracks_;
  };
#pragma endregion

#pragma region INITIALIZATION

public:
  /**
   * @brief 构造函数。
   *
   * @param config ESKF 配置参数。
   *
   * @details
   * 根据 IMU 频率、最大传感器延迟、时间抖动以及安全余量，
   * 动态计算 History Buffer 的容量。
   */
  ErrorStateKalmanFilter(const Config &config) :
    config_{config},
    history_buffer_{
        static_cast<std::size_t>(std::ceil(config_.imu_rate_
                                           * (config_.max_sensor_delay_
                                              + config_.max_sensor_jitter_)
                                           * static_cast<value_type>(1e-3)))
        + config_.history_buffer_margin_
    }
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
   * @param imu_datum IMU 数据提供的角速度向量和线加速度向量
   * @note 调用者必须保证 IMU 数据是在“体坐标系”下的表示
   */
  void ImuUpdate(const DatumImu *imu_datum, bool record_history = true) noexcept
  {
#pragma region FOOL_PROOF

    if (imu_datum == nullptr)
    {
      return;
    }

    if (last_imu_time_ < 0)
    {
      last_imu_time_  = imu_datum->timestamp_;
      last_imu_datum_ = *imu_datum;
      return;
    }

    const value_type dt{
        static_cast<value_type>((imu_datum->timestamp_ - last_imu_time_) * 1e-9)
    };
    if (dt <= 0.0)
    {
      return;
    }

#pragma endregion

#pragma region WORLD_ANGULAR_VELOCITY_AND_LINEAR_ACCELERATION

    // 假设上一帧 IMU 数据 `last_imu_datum_` 中的角速度、线加速度都已经去除零偏
    Vector3 unbias_gyro_prev{last_imu_datum_.angular_velocity_};
    Vector3 unbias_acc_prev{last_imu_datum_.linear_acceleration_};

    // 当前帧 IMU 数据 imu_datum 去除零偏
    Vector3 unbias_gyro_curr{imu_datum->angular_velocity_
                             - nominal_state_.gyroscope_bias_};
    Vector3 unbias_acc_curr{imu_datum->linear_acceleration_
                            - nominal_state_.accelerometer_bias_};
    // 保存当前帧 IMU 数据
    last_imu_datum_ = DatumImu{
        imu_datum->timestamp_,
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

    last_imu_time_ = imu_datum->timestamp_;

    if (record_history)
    {
      SaveHistoryState(*imu_datum);
    }
  }

  /**
   * @brief 每当收到新视觉观测周期采样信号时，执行卡尔曼观测融合及误差校正。
   * @param monocular_datum 单目视觉数据提供的角位移向量和单位化平移向量
   * @note 调用者必须保证单目视觉数据是在“体坐标系”下的表示
   */
  void MonocularUpdate(const DatumFast *monocular_datum) noexcept
  {
#pragma region FOOL_PROOF

    if (monocular_datum == nullptr)
    {
      return;
    }
    if (last_cam_time_ < 0)
    {
      last_cam_time_ = monocular_datum->timestamp_;
      return;
    }

    const value_type dt{static_cast<value_type>(
        (monocular_datum->timestamp_ - last_cam_time_) * 1e-9
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

    JacobiMeasurementFast H{GetMeasurementJacobiFast(angular_displacement)};
    CovarianceMeasurementFast V{GetMeasurementCovarianceFast()};
    KalmanGainFast K{GetKalmanGainFast(error_state_covariance_, H, V)};

    // 观测残差向量

#if (ONLY_USE_ANGULAR_DISPLACEMENT)
    Vector3 measurement_residue{monocular_datum->angular_displacement_
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
        = monocular_datum->angular_displacement_ - angular_displacement;
    measurement_residue.template tail<3>()
        = monocular_datum->normalized_translation_ - normalized_translation;
#endif

    prev_attitude_ = nominal_state_.attitude_;
    prev_position_ = nominal_state_.position_;

    error_state_ = K * measurement_residue;

    UpdateErrorStateCovarianceFast(error_state_covariance_, K, H, V);

    // 立即向标称状态进行负反馈注入复位，重置误差空间
    InjectError();
    last_cam_time_ = monocular_datum->timestamp_;
  }

  void StereoUpdate(std::int64_t timestamp,
                    std::span<StereoObservation> obs) noexcept
  {
    ++vision_frame_count_;

    typename HistoryBuffer::const_iterator itr{FindHistoryIndex(timestamp)};
    RollbackToHistory(itr);

    // TODO

    ReplayHistory(itr);
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
   * @brief 保存历史状态
   * @param imu_datum IMU 数据指针
   *
   * @details
   * 每一次 IMU 完成状态传播以后，都保存当前滤波器状态。
   * 后续若收到具有延迟的视觉/GPS等观测，可回滚至对应时间戳，
   * 完成 Measurement Update 后再重新积分到当前时刻。
   */
  void SaveHistoryState(const DatumImu &imu_datum) noexcept
  {
    HistoryState history_state;
    // 保存当前帧、未经零偏校正的原始 IMU 数据
    history_state.raw_imu_datum_ = imu_datum;
    // 保存上一帧、经过零偏矫正的 IMU 数据
    history_state.last_imu_datum_ = last_imu_datum_;
    // 保存当前名义状态
    history_state.nominal_ = nominal_state_;
    // 保存当前误差协方差
    history_state.error_state_covariance_ = error_state_covariance_;
    // 写入环形历史缓冲区
    history_buffer_.Push(history_state);
  }

  /**
   * @brief 根据时间戳查找历史状态。
   * @return 返回历史状态的指针
   */
  typename HistoryBuffer::const_iterator
  FindHistoryIndex(std::int64_t timestamp) const noexcept
  {
    return history_buffer_.Find(timestamp);
  }

  /**
   * @brief 回滚到指定历史状态
   */
  void
  RollbackToHistory(const typename HistoryBuffer::const_iterator &itr) noexcept
  {
    if (itr == history_buffer_.cend())
    {
      return;
    }
    nominal_state_          = itr->nominal_;
    error_state_covariance_ = itr->error_state_covariance_;
    last_imu_time_          = itr->raw_imu_datum_.timestamp_;
    last_imu_datum_         = itr->last_imu_datum_;
  }

  /**
   * @brief 从 index+1 开始重新积分 IMU
   */
  void ReplayHistory(typename HistoryBuffer::const_iterator itr) noexcept
  {
    if (itr == history_buffer_.cend())
    {
      return;
    }
    for (++itr; itr != history_buffer_.cend(); ++itr)
    {
      ImuUpdate(&itr->raw_imu_datum_, false);
    }
  }

  /**
   * @brief 更新当前帧 Feature Track Table。
   *
   * @param timestamp 图像时间戳。
   * @param feature_ids 当前帧 Feature ID。
   * @param pts_left 左目像素坐标。
   * @param pts_right 右目像素坐标。
   */
  void UpdateFeatureTrackTable(std::int64_t timestamp,
                               std::span<const uint32_t> feature_ids,
                               const Matrix2X &pts_left,
                               const Matrix2X &pts_right) noexcept
  {
    feature_track_table_.tracks_.clear();

    const Eigen::Index count{pts_left.cols()};

    for (Eigen::Index i = 0; i < count; ++i)
    {
      FeatureTrack track;

      track.id_           = feature_ids[static_cast<std::size_t>(i)];
      track.timestamp_    = timestamp;
      track.left_         = pts_left.col(i);
      track.right_        = pts_right.col(i);
      track.stereo_valid_ = true;

      feature_track_table_.tracks_.emplace(track.id_, std::move(track));
    }
  }

  /**
   * @brief 根据 Feature ID 查询 Landmark。
   *
   * @param id Feature ID。
   * @return 找到返回指针，否则返回 nullptr。
   */
  Landmark *FindLandmark(std::uint32_t id) noexcept
  {
    auto iter = landmark_database_.landmarks_.find(id);

    if (iter == landmark_database_.landmarks_.end())
    {
      return nullptr;
    }

    return &iter->second;
  }

  /**
   * @brief 创建新的 Landmark。
   *
   * @details
   * Stereo DLT 三角化得到的是当前左目相机坐标系下的三维点。
   * 为了使 Landmark 能够跨帧复用，必须将其转换到世界坐标系后再存储。
   */
  Landmark &CreateLandmark(const FeatureTrack &track) noexcept
  {
    // 双目三角化，得到当前左目相机坐标系下三维点
    const Vector3 p_cam{Triangulate(track.left_, track.right_)};

    // Camera -> Body(IMU)
    const Vector3 p_body{config_.stereo_camera_model_.transform_cam0_ * p_cam};

    // Body -> World
    const Pose T_WB{Attitude{nominal_state_.attitude_},
                    nominal_state_.position_};

    const Vector3 p_world{T_WB * p_body};

    // 创建 Landmark
    Landmark landmark;
    landmark.id_             = track.id_;
    landmark.position_       = p_world;
    landmark.timestamp_      = track.timestamp_;
    landmark.last_frame_id_  = vision_frame_count_;
    landmark.lost_count_     = 0;
    landmark.observed_count_ = 1;

    auto [iter, inserted]
        = landmark_database_.landmarks_.emplace(landmark.id_,
                                                std::move(landmark));

    return iter->second;
  }

  /**
   * @brief 更新所有 Landmark 的生命周期状态。
   *
   * @details
   * StereoUpdate() 完成所有 Feature 匹配以后调用。
   *
   * 状态转移：
   *
   * New
   *    ├── 连续观测>=3帧 ----------> Active
   *    └── 本帧丢失 ---------------> Lost
   *
   * Active
   *    ├── 本帧继续观测 -----------> Active
   *    └── 本帧未观测 -------------> Lost
   *
   * Lost
   *    ├── 再次观测 ---------------> Active
   *    └── 连续丢失>20帧 ---------> Erased
   */
  void UpdateLandmarkLifeCycle() noexcept
  {
    constexpr std::uint32_t kInitializeThreshold{3};

    for (auto &[id, landmark] : landmark_database_.landmarks_)
    {
      // 本帧没有观测到该 Landmark
      if (landmark.last_frame_id_ != vision_frame_count_)
      {
        ++landmark.lost_count_;

        if (landmark.status_ != LandmarkStatus::Erased)
        {
          landmark.status_ = LandmarkStatus::Lost;
        }

        continue;
      }

      // 本帧观测到了 Landmark
      landmark.lost_count_ = 0;

      if (landmark.status_ == LandmarkStatus::New)
      {
        if (landmark.observed_count_ >= kInitializeThreshold)
        {
          landmark.status_ = LandmarkStatus::Active;
        }
      }
      else
      {
        landmark.status_ = LandmarkStatus::Active;
      }
    }
  }

  /**
   * @brief 擦除长期失踪或已经位于相机后方的 Landmark。
   *
   * 删除条件：
   * 1. 左目或右目投影深度 w<=0；
   * 2. 连续丢失超过20帧。
   */
  void RemoveLostLandmarks() noexcept
  {
    constexpr std::uint32_t kMaxLostFrames{20};

    for (auto iter = landmark_database_.landmarks_.begin();
         iter != landmark_database_.landmarks_.end();)
    {
      auto &landmark = iter->second;

      bool erase = false;

      // 条件1：连续丢失过久
      if (landmark.lost_count_ > kMaxLostFrames)
      {
        erase = true;
      }

      // 条件2：位于双目相机后方
      if (!erase)
      {
        // World -> Body
        const Pose T_WB{
            Attitude{nominal_state_.attitude_},
            nominal_state_.position_,
        };

        const Vector3 point_body{
            T_WB.inverse() * landmark.position_,
        };

        // Body -> Camera
        const Pose T_SB{
            config_.stereo_camera_model_.transform_cam0_.inverse(),
        };

        const Vector3 point_camera{
            T_SB * point_body,
        };

        Eigen::Vector4<value_type> X;
        X << point_camera, static_cast<value_type>(1);

        const Eigen::Vector3<value_type> left_h{
            config_.stereo_camera_model_.proj_left_ * X,
        };

        const Eigen::Vector3<value_type> right_h{
            config_.stereo_camera_model_.proj_right_ * X,
        };

        if (left_h.z() <= static_cast<value_type>(0)
            || right_h.z() <= static_cast<value_type>(0))
        {
          erase = true;
        }
      }

      // 擦除
      if (erase)
      {
        iter = landmark_database_.landmarks_.erase(iter);
      }
      else
      {
        ++iter;
      }
    }
  }

  /**
   * @brief 双目三角化。
   *
   * @param pixels_left 左目像素。
   * @param pixels_right 右目像素。
   *
   * @return 世界坐标。
   */
  Vector3 Triangulate(const Vector2 &pixels_left,
                      const Vector2 &pixels_right) const noexcept
  {
    Eigen::Matrix<value_type, 4, 4> A;

    A.row(0) = pixels_left.x() * config_.proj_left_.row(2)
               - config_.proj_left_.row(0);
    A.row(1) = pixels_left.y() * config_.proj_left_.row(2)
               - config_.proj_left_.row(1);
    A.row(2) = pixels_right.x() * config_.proj_right_.row(2)
               - config_.proj_right_.row(0);
    A.row(3) = pixels_right.y() * config_.proj_right_.row(2)
               - config_.proj_right_.row(1);

    Eigen::JacobiSVD<Eigen::Matrix<value_type, 4, 4>> svd(A,
                                                          Eigen::ComputeFullV);
    Eigen::Vector4<value_type> X = svd.matrixV().col(3);

    X /= X.w();

    return X.template head<3>();
  }

  /**
   * @brief 将世界坐标系中的 Landmark 投影到指定相机像素平面。
   *
   * @param P 相机 3×4 投影矩阵（立体校正后的左目或右目）。
   * @param landmark 世界坐标系中的 Landmark。
   * @return 对应像素坐标。
   */
  Vector2 Project(const ProjectionMatrix &P,
                  const Vector3 &landmark) const noexcept
  {
    // 构造当前 IMU(Body) 在世界坐标系中的位姿 T_WB : Body -> World
    const Pose T_WB{
        Attitude{nominal_state_.attitude_},
        nominal_state_.position_,
    };

    // World -> Body
    const Vector3 point_body{
        T_WB.inverse() * landmark,
    };

    // Body -> Camera
    const Vector3 point_camera{
        config_.stereo_camera_model_.transform_cam0_.inverse() * point_body,
    };

    // 齐次投影
    Eigen::Vector4<value_type> X;
    X << point_camera, static_cast<value_type>(1);

    const Vector3 p{P * X};

    return p.template head<2>() / p.z();
  }

  Vector2 ProjectLeft(const Vector3 &landmark) const noexcept
  {
    return Project(config_.proj_left_, landmark);
  }

  Vector2 ProjectRight(const Vector3 &landmark) const noexcept
  {
    return Project(config_.proj_right_, landmark);
  }

  /**
   * @brief 根据视差估计像素方差。
   */
  value_type PixelVariance(value_type disparity) const noexcept
  {
    constexpr value_type sigma0{0.5};
    constexpr value_type k{8.0};
    constexpr value_type eps{1e-6};
    return sigma0 * sigma0 + k / (disparity * disparity + eps);
  }

  /**
   * @brief 构造双目测量噪声协方差矩阵。
   */
  CovarianceMeasurementStereo
  StereoMeasurementNoiseCovariance(const FeatureTrack &track) const noexcept
  {
    CovarianceMeasurementStereo R{CovarianceMeasurementStereo::Zero()};
    const value_type disparity{std::abs(track.left_.x() - track.right_.x())};
    const value_type var{PixelVariance(disparity)};
    R.diagonal().setConstant(var);
    return R;
  }

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
  JacobiMeasurementFast
  GetMeasurementJacobiFast(const Vector3 &angular_displacement) const noexcept
  {
    JacobiMeasurementFast result{JacobiMeasurementFast::Zero()};

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
  CovarianceMeasurementFast GetMeasurementCovarianceFast() const noexcept
  {
    CovarianceMeasurementFast V{CovarianceMeasurementFast::Identity()};
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
  static KalmanGainFast
  GetKalmanGainFast(const TransitionMatrix &P, const JacobiMeasurementFast &H,
                    const CovarianceMeasurementFast &V) noexcept
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
  UpdateErrorStateCovarianceFast(TransitionMatrix &P, const KalmanGainFast &K,
                                 const JacobiMeasurementFast &H,
                                 const CovarianceMeasurementFast &V) noexcept
  {
    TransitionMatrix ikh{TransitionMatrix::Identity() - K * H};
    P = ikh * P * ikh.transpose() + K * V * K.transpose();
  }

  /**
   * @brief 计算旋转向量对四元数的 Jacobian: d(q * a * q*) / dq
   * @param q 旋转四元数
   * @param a 被旋转的三维向量
   * @return 3x4 的雅可比矩阵
   */
  static auto Jacobian_Rotation_wrt_Quaternion(const Quaternion &q,
                                               const Vector3 &a) noexcept
  {
    value_type w{q.w()};
    auto v{q.vec()};

    Eigen::Matrix<value_type, 3, 4> J;
    // 对实部 w 的导数
    J.col(0) = static_cast<value_type>(2.0) * (w * a + v.cross(a));
    // 对虚部 v 的导数 (3x3)
    Matrix3 term1{v.dot(a) * Matrix3::Identity()};
    Matrix3 term2{v * a.transpose()};
    Matrix3 term3{a * v.transpose()};
    Matrix3 term4{w * Attitude::hat(a).matrix()};
    J.template block<3, 3>(0, 1)
        = static_cast<value_type>(2.0) * (term1 + term2 - term3 - term4);

    return J;
  }

  /**
   * @brief 计算朝向更新函数对轴角误差的雅可比矩阵
   * @param q 旋转四元数
   * @return 4x3 的雅可比矩阵
   */
  static auto Jacobian_Quaternion_wrt_dtheta(const Quaternion &q) noexcept
  {
    using RetType = Eigen::Matrix<value_type, 4, 3>;
    RetType result{
        {-q.x(), -q.y(), -q.z()},
        {q.w(), -q.z(), q.y()},
        {q.z(), q.w(), -q.x()},
        {-q.y(), q.x(), q.w()},
    };
    return static_cast<value_type>(0.5) * result;
  }

  /**
   * @brief 计算投影矩阵对路标点的雅克比矩阵
   * @param proj_mat 投影矩阵
   * @param landmark_cam 路标点在相机坐标系下的坐标
   * @return 2x3 的雅可比矩阵
   */
  static JacobiProjection
  GetProjectionJacobian(const ProjectionMatrix &proj_mat,
                        const Vector3 &landmark_cam) noexcept
  {
    const auto KR{proj_mat.template block<3, 3>(0, 0)};
    const Vector3 pixel_homo{KR * landmark_cam + proj_mat.col(3)};
    const auto p1{KR.row(0)}, p2{KR.row(1)}, p3{KR.row(2)};
    const auto x{pixel_homo.x()}, y{pixel_homo.y()}, z{pixel_homo.z()};
    const auto denom{static_cast<value_type>(1.0) / (z * z)};
    const auto dudp{(z * p1 - x * p3) * denom}, dvdp{(z * p2 - y * p3) * denom};
    JacobiProjection result;
    result << dudp, dvdp;
    return result;
  }

  /**
   * @brief 计算单个 Landmark 的双目观测雅可比矩阵
   *
   * @param landmark 世界坐标系中的 Landmark
   * @return 双目观测雅可比矩阵（4×18）
   */
  JacobiMeasurementStereo
  GetMeasurementJacobiStereo(const Vector3 &landmark) const noexcept
  {
    // [左目像素点坐标 ; 右目像素点坐标] = 观测函数(路标点的世界坐标)
    // 路标点的体坐标 = T_{BW} (路标点的世界坐标)
    // 路标点的左目坐标 = T_{SB} (路标点的体坐标)
    // 左目像素点坐标 = 左目投影矩阵 (路标点的左目坐标)
    // 右目像素点坐标 = 右目投影矩阵 (路标点的左目坐标)

    const Pose T_WB{
        Attitude{nominal_state_.attitude_},
        nominal_state_.position_,
    };

    // World -> Body
    const Vector3 p_body{
        T_WB.inverse() * landmark,
    };

    // Body -> Camera
    const Pose T_SB{
        config_.stereo_camera_model_.transform_cam0_.inverse(),
    };
    const Vector3 p_cam{
        T_SB * p_body,
    };

    // 分别计算左右目的 2×3 投影雅可比
    const JacobiProjection J_left{
        GetProjectionJacobian(config_.stereo_camera_model_.proj_left_, p_cam),
    };
    const JacobiProjection J_right{
        GetProjectionJacobian(config_.stereo_camera_model_.proj_right_, p_cam),
    };

    // 路标点的体坐标相对于误差状态的雅克比矩阵
    using JacobiLandmarkBody_wrt_ErrorState
        = Eigen::Matrix<value_type, 3, dimErrorState>;
    JacobiLandmarkBody_wrt_ErrorState J_pose{
        JacobiLandmarkBody_wrt_ErrorState::Zero()
    };
    const Matrix3 R_BW{T_WB.so3().inverse().matrix()};
    J_pose.template block<3, 3>(0, 0) = -R_BW;
    J_pose.template block<3, 3>(0, 6) = Attitude::hat(p_body).matrix();

    // 拼接左右目
    JacobiMeasurementStereo H{JacobiMeasurementStereo::Zero()};

    const Matrix3 R_SB{T_SB.so3().matrix()};
    const JacobiLandmarkBody_wrt_ErrorState J_common{R_SB * J_pose};
    H.template block<2, dimErrorState>(0, 0) = J_left * J_common;
    H.template block<2, dimErrorState>(2, 0) = J_right * J_common;

    return H;
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
  // 图像帧计数器
  std::uint32_t vision_frame_count_{0};
  // 上一帧图像帧的姿态
  Vector3 prev_position_{Vector3::Zero()};
  Quaternion prev_attitude_{Quaternion::Identity()};
  // 上一帧 IMU 数据的缓存结构
  DatumImu last_imu_datum_{};
  // 历史状态缓冲区
  HistoryBuffer history_buffer_;
  LandmarkDatabase landmark_database_;
  FeatureTrackTable feature_track_table_;

#pragma endregion
};

} // namespace VisualSim
