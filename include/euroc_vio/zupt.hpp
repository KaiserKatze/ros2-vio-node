#ifndef ZUPT_HPP
#define ZUPT_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "euroc_vio/CircularBuffer.hpp"

using Vector6d = Eigen::Matrix<double, 6, 1>;

/**
 * @brief 零速更新（ZUPT）
 *
 * 优化点：
 * - 维护滑动窗口的增量统计（O(1) 更新）
 * - 移除不必要的数据遍历，利用线性系统更新期望与方差
 */
template <size_t WindowSize = 64> class ZUPT
{
public:
  using data_type   = Vector6d;
  using chunk_type  = Eigen::Vector3d;
  using window_type = CircularBuffer<data_type, WindowSize>;

  struct Config
  {
    double gyroscope_magnitude_threshold{0.005};
    double accelerometer_variance_threshold{0.05};
    double g_tolerance{0.3};
    double local_gravity{9.81};

    bool IsAccelWithinGravityRange(const Eigen::Vector3d &acc_mean) const
    {
      const double acc_mean_norm{acc_mean.norm()};
      return std::abs(acc_mean_norm - local_gravity) <= g_tolerance;
    }
  };

  ZUPT() : ZUPT(Config{}) {}
  ZUPT(Config &&config) : config_{std::move(config)} {}

private:
  static Eigen::Vector3d GetAccel(const data_type &e)
  {
    return e.template tail<3>();
  }

  static Eigen::Vector3d GetGyro(const data_type &e)
  {
    return e.template head<3>();
  }

  static double GetGyroNorm(const data_type &e)
  {
    return GetGyro(e).norm();
  }

  Eigen::Vector3d GetAverageAccel() const
  {
    if (window_.size() == 0)
    {
      return Eigen::Vector3d::Zero();
    }
    return accel_sum_ / static_cast<double>(window_.size());
  }

public:
  /**
   * @brief 估计当前姿态四元数（仅在静止时可靠）
   * @note 返回的四元数是从世界坐标系到机体坐标系的旋转（即 C_21）
   */
  Eigen::Quaterniond EstimateOrientation() const
  {
    // 默认初始状态是静止的
    if (window_.size() == 0)
    {
      return Eigen::Quaterniond::Identity();
    }
    if (!this->is_static_)
    {
      // 当前状态被判断为非静止，无法可靠估计姿态
      throw std::runtime_error{
          "Cannot estimate orientation reliably when not static."};
    }
    // 计算平均加速度向量，作为重力方向的估计
    const Eigen::Vector3d acc_mean{GetAverageAccel()};
    if (!config_.IsAccelWithinGravityRange(acc_mean))
    {
      // 平均加速度不在重力范围内，无法可靠估计姿态
      std::stringstream ss;
      ss << std::fixed << std::setprecision(3) << "Average acceleration norm "
         << acc_mean.norm() << " is out of expected gravity range ("
         << this->config_.local_gravity - this->config_.g_tolerance << ", "
         << this->config_.local_gravity + this->config_.g_tolerance
         << "). Cannot estimate orientation reliably.";
      throw std::runtime_error{ss.str()};
    }
    const Eigen::Vector3d gravity_sensor{-acc_mean.normalized()};
    const Eigen::Vector3d gravity_world{Eigen::Vector3d::UnitX()};

    // https://runebook.dev/en/docs/eigen3/classeigen_1_1quaternion/acdb1eb44eb733b24749bc7892badde64
    const double dot_product{gravity_world.dot(gravity_sensor)};
    // 约等于 cos(1°)，用于数值稳定性判断
    static constexpr double threshold{0.9999};
    if (dot_product > threshold) // 同向
    {
      return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    }
    else if (dot_product < -threshold) // 反向
    {
      return Eigen::Quaterniond(0.0, 0.0, 1.0, 0.0);
    }

    // 构造一个旋转，使得机体坐标系的 x 轴（前向）与重力方向对齐
    // 注意：这只是一个近似的估计，实际应用中可能需要更复杂的处理
    return Eigen::Quaterniond::FromTwoVectors(gravity_world, gravity_sensor);
  }

  /**
   * @brief 更新一帧 IMU 数据并判断是否静止
   * @return 当前是否被判断为静止状态
   */
  bool Update(const data_type &imu_data)
  {
    // 记录推送前的状态，决定是否需要从统计量中“减去”被挤出的旧数据
    const bool was_full{window_.full()};

    // push 返回的是那个位置原本的数据（如果已满，那就是将要丢弃的老数据）
    const data_type old_data{window_.push(imu_data)};

    // === 计算新数据的特征量 ===
    const double new_gyro_norm{ZUPT::GetGyroNorm(imu_data)};
    const Eigen::Vector3d new_acc{ZUPT::GetAccel(imu_data)};
    const Eigen::Vector3d new_acc_sq{
        new_acc.array().square().matrix()}; // 逐元素平方

    // === 统计量加上新数据 ===
    gyro_norm_sum_ += new_gyro_norm;
    accel_sum_ += new_acc;
    accel_sq_sum_ += new_acc_sq;

    // === 统计量减去旧数据（仅当缓冲区原本已满时） ===
    if (was_full)
    {
      const double old_gyro_norm{ZUPT::GetGyroNorm(old_data)};
      const Eigen::Vector3d old_acc{ZUPT::GetAccel(old_data)};
      const Eigen::Vector3d old_acc_sq{
          old_acc.array().square().matrix()}; // 逐元素平方

      gyro_norm_sum_ -= old_gyro_norm;
      accel_sum_ -= old_acc;
      accel_sq_sum_ -= old_acc_sq;
    }

    // 缓冲区未满前，统计量还不具备统计学意义，直接返回 true
    if (!window_.full())
    {
      return this->is_static_ = true;
    }

    const double denom{1.0 / static_cast<double>(WindowSize)};

    /** =========================
     * 陀螺仪能量判断
     * ========================= */
    const double gyro_energy{gyro_norm_sum_ * denom};

    if (gyro_energy > config_.gyroscope_magnitude_threshold)
    {
      return this->is_static_ = false;
    }

    /** =========================
     * 加速度均值判断
     * ========================= */
    const Eigen::Vector3d acc_mean{accel_sum_ * denom};
    if (!config_.IsAccelWithinGravityRange(acc_mean))
    {
      return this->is_static_ = false;
    }

    /** =========================
     * 加速度各分量方差判断 (D(X) = E(X^2) - [E(X)]^2)
     * ========================= */
    const Eigen::Vector3d acc_mean_sq{
        acc_mean.array().square().matrix()}; // 逐元素平方
    Eigen::Vector3d acc_variance{accel_sq_sum_ * denom - acc_mean_sq};
    // 抵消由于浮点数精度累积误差可能导致的极微小负数
    double acc_total_variance{
        std::max(0.0, acc_variance.sum())}; // 取最大分量方差

    if (acc_total_variance > config_.accelerometer_variance_threshold)
    {
      return this->is_static_ = false;
    }

    return this->is_static_ = true;
  }

private:
  Config config_;
  window_type window_;

  // 滑动窗口的全局状态变量
  double gyro_norm_sum_{0.0};                             // 角速度向量范数和
  Eigen::Vector3d accel_sum_{Eigen::Vector3d::Zero()};    // 加速度向量和
  Eigen::Vector3d accel_sq_sum_{Eigen::Vector3d::Zero()}; // 加速度向量平方和
  bool is_static_{true};                                  // 当前是否静止
};

#endif /* ZUPT_HPP */
