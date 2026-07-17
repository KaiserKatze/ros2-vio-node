module;

#include <Eigen/Dense>

export module FastVIO:ZUPT;

import std;

import :CircularBuffer;

template <typename T /* Eigen::Vector */, typename U /* float or double */>
  requires std::is_floating_point_v<U>
T GetAverage_IQR(std::vector<std::pair<T, U>> &points)
{
  if (points.empty())
  {
    return T::Zero();
  }

  // 对 points 的 U 属性进行排序
  std::ranges::sort(points, std::less<>{},
                    [](const auto &e) { return e.second; });

  size_t n{points.size()};

  // 辅助 Lambda：使用线性插值计算任意百分位数
  auto get_percentile = [&](U percentile) -> U
  {
    if (n == 1)
    {
      return points[0].second;
    }

    U pos{(static_cast<U>(n) - static_cast<U>(1.0)) * percentile};
    size_t idx{static_cast<size_t>(std::floor(pos))};
    U frac{pos - static_cast<U>(idx)};

    if (idx + 1 < n)
    {
      return points[idx].second * (static_cast<U>(1.0) - frac)
             + points[idx + 1].second * frac;
    }
    return points[idx].second;
  };

  // 计算 Q1 (第25百分位数) 和 Q3 (第75百分位数)
  U q1{get_percentile(static_cast<U>(0.25))};
  U q3{get_percentile(static_cast<U>(0.75))};
  U iqr{q3 - q1};

  // 划定内限 (Mild Outlier Bounds)
  U mild_alpha{static_cast<U>(1.5)};
  U mild_lower_bound{q1 - mild_alpha * iqr};
  U mild_upper_bound{q3 + mild_alpha * iqr};

  // 提取内点（非异常值）对应的 T 对象，并计算平均值
  T sum_vector{T::Zero()};
  int valid_count{0};

  for (const auto &point : points)
  {
    // 只对内限范围内的点求和
    if (point.second >= mild_lower_bound && point.second <= mild_upper_bound)
    {
      sum_vector += point.first;
      valid_count++;
    }
  }

  // 防止全部被判定为异常值（虽然在 IQR 中极少发生，但需防止除以 0）
  if (valid_count == 0)
  {
    return T::Zero();
  }

  return sum_vector / static_cast<U>(valid_count);
}

export namespace FastVIO
{

/**
 * @brief 零速更新（ZUPT）
 *
 * 优化点：
 * - 维护滑动窗口的增量统计（O(1) 更新）
 * - 移除不必要的数据遍历，利用线性系统更新期望与方差
 */
template <typename value_type, std::size_t WindowSize = 64>
class ZUPT
{
public:
  using chunk_type  = Eigen::Vector<value_type, 3>;
  using data_type   = std::pair<chunk_type, chunk_type>;
  using window_type = CircularBuffer<data_type, WindowSize>;

  struct Config
  {
    value_type gyroscope_magnitude_threshold{0.005};
    value_type accelerometer_variance_threshold{0.05};
    value_type g_tolerance{0.3};
    value_type local_gravity{9.81};

    bool IsAccelWithinGravityRange(const chunk_type &acc_mean) const
    {
      const value_type acc_mean_norm{acc_mean.norm()};
      return std::abs(acc_mean_norm - local_gravity) <= g_tolerance;
    }
  };

  ZUPT() : ZUPT(Config{}) {}
  ZUPT(Config &&config) : config_{std::move(config)} {}

private:
  static chunk_type GetAccel(const data_type &e)
  {
    return std::get<0>(e);
  }

  static chunk_type GetGyro(const data_type &e)
  {
    return std::get<1>(e);
  }

  static value_type GetGyroNorm(const data_type &e)
  {
    return GetGyro(e).norm();
  }

public:
  bool IsFull() const
  {
    return window_.full();
  }

  /**
   * @brief 估计当前姿态四元数（仅在静止时可靠）
   * @note 返回的四元数是从世界坐标系到机体坐标系的旋转（即 C_21）
   */
  Eigen::Quaternion<value_type> EstimateOrientation() const
  {
    // 默认初始状态是静止的
    if (window_.size() == 0)
    {
      return Eigen::Quaternion<value_type>::Identity();
    }
    if (!this->is_static_)
    {
      // 当前状态被判断为非静止，无法可靠估计姿态
      std::print(
          stderr,
          "[WARN] Cannot estimate orientation reliably when not static.\n"
      );
    }
    // 使用 IQR 方法，去除外点，计算平均加速度向量，作为重力方向的估计
    std::vector<std::pair<chunk_type, value_type>> points;
    points.reserve(window_.size());
    std::transform(window_.cbegin(), window_.cend(), std::back_inserter(points),
                   [](const data_type &e)
                   {
                     const chunk_type &vec{GetAccel(e)};
                     // 返回一个 pair，second 存储 norm()
                     return std::make_pair(vec, vec.norm());
                   });
    // std::print(stderr, "[INFO] ZUPT 缓存池 (大小: {}) 加速度向量及其范数 =\n",
    //            points.size());
    // for (auto &[vec, vec_norm] : points)
    // {
    //   std::print(stderr, "\t[{:.4f}, {:.4f}, {:.4f}]; {:.4f}\n", //
    //              vec.x(), vec.y(), vec.z(), vec_norm);
    // }
    const chunk_type acc_mean{GetAverage_IQR(points)};
    std::print(stderr,
               "[INFO] 利用 IQR 方法计算得出 ZUPT 缓存池的平均比力 =\n"
               "\t[{:.4f}, {:.4f}, {:.4f}]\n",
               acc_mean.x(), acc_mean.y(), acc_mean.z());
    if (!config_.IsAccelWithinGravityRange(acc_mean))
    {
      // 平均加速度不在重力范围内，无法可靠估计姿态
      std::print(stderr,
                 "[WARN] Average acceleration norm {:.3f} "
                 "is out of expected gravity range ({:.3f}, {:.3f}). "
                 "Cannot estimate orientation reliably.",
                 acc_mean.norm(),
                 (this->config_.local_gravity - this->config_.g_tolerance),
                 (this->config_.local_gravity + this->config_.g_tolerance));
    }
    const chunk_type gravity_sensor{-acc_mean.normalized()};
    const chunk_type gravity_world{-chunk_type::UnitZ()};

    // https://runebook.dev/en/docs/eigen3/classeigen_1_1quaternion/acdb1eb44eb733b24749bc7892badde64
    const value_type dot_product{gravity_world.dot(gravity_sensor)};
    // 约等于 cos(1°)，用于数值稳定性判断
    static constexpr value_type threshold{0.9999};
    if (dot_product > threshold) // 同向
    {
      return Eigen::Quaternion<value_type>(1.0, 0.0, 0.0, 0.0);
    }
    else if (dot_product < -threshold) // 反向
    {
      return Eigen::Quaternion<value_type>(0.0, 0.0, 1.0, 0.0);
    }

    // 构造一个旋转，使得机体坐标系的 x 轴（前向）与重力方向对齐
    // 注意：这只是一个近似的估计，实际应用中可能需要更复杂的处理
    return Eigen::Quaternion<value_type>::FromTwoVectors(gravity_world,
                                                         gravity_sensor)
        .normalized();
  }

  /**
   * @brief 更新一帧 IMU 数据并判断是否静止
   * @return 当前是否被判断为静止状态
   */
  bool Update(const chunk_type &linear_acceleration,
              const chunk_type &angular_velocity)
  {
    // std::print(stderr,
    //            "ZUPT 缓存池新增数据:\n"
    //            "\t加速度 = [{:.4f}, {:.4f}, {:.4f}]\n"
    //            "\t角速度 = [{:.4f}, {:.4f}, {:.4f}]\n",
    //            linear_acceleration.x(), linear_acceleration.y(),
    //            linear_acceleration.z(), angular_velocity.x(),
    //            angular_velocity.y(), angular_velocity.z());
    const data_type imu_data{
        std::make_pair(linear_acceleration, angular_velocity),
    };
    // 记录推送前的状态，决定是否需要从统计量中“减去”被挤出的旧数据
    const bool was_full{IsFull()};

    // push 返回的是那个位置原本的数据（如果已满，那就是将要丢弃的老数据）
    const data_type old_data{window_.push(imu_data)};

    // === 计算新数据的特征量 ===
    const value_type new_gyro_norm{ZUPT::GetGyroNorm(imu_data)};
    const chunk_type new_acc{ZUPT::GetAccel(imu_data)};
    const chunk_type new_acc_sq{
        new_acc.array().square().matrix()
    }; // 逐元素平方

    // === 统计量加上新数据 ===
    gyro_norm_sum_ += new_gyro_norm;
    accel_sum_ += new_acc;
    accel_sq_sum_ += new_acc_sq;

    // === 统计量减去旧数据（仅当缓冲区原本已满时） ===
    if (was_full)
    {
      const value_type old_gyro_norm{ZUPT::GetGyroNorm(old_data)};
      const chunk_type old_acc{ZUPT::GetAccel(old_data)};
      const chunk_type old_acc_sq{
          old_acc.array().square().matrix(),
      }; // 逐元素平方

      gyro_norm_sum_ -= old_gyro_norm;
      accel_sum_ -= old_acc;
      accel_sq_sum_ -= old_acc_sq;
    }

    // 缓冲区未满前，统计量还不具备统计学意义，直接返回 true
    if (!IsFull())
    {
      return this->is_static_ = true;
    }

    const value_type denom{1.0 / static_cast<value_type>(WindowSize)};

    /** =========================
     * 陀螺仪能量判断
     * ========================= */
    const value_type gyro_energy{gyro_norm_sum_ * denom};

    if (gyro_energy > config_.gyroscope_magnitude_threshold)
    {
      return this->is_static_ = false;
    }

    /** =========================
     * 加速度均值判断
     * ========================= */
    const chunk_type acc_mean{accel_sum_ * denom};
    if (!config_.IsAccelWithinGravityRange(acc_mean))
    {
      return this->is_static_ = false;
    }

    /** =========================
     * 加速度各分量方差判断 (D(X) = E(X^2) - [E(X)]^2)
     * ========================= */
    const chunk_type acc_mean_sq{
        acc_mean.array().square().matrix()
    }; // 逐元素平方
    chunk_type acc_variance{accel_sq_sum_ * denom - acc_mean_sq};
    // 抵消由于浮点数精度累积误差可能导致的极微小负数
    value_type acc_total_variance{
        std::max(static_cast<value_type>(0.0), acc_variance.sum())
    }; // 取最大分量方差

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
  value_type gyro_norm_sum_{0.0};               // 角速度向量范数和
  chunk_type accel_sum_{chunk_type::Zero()};    // 加速度向量和
  chunk_type accel_sq_sum_{chunk_type::Zero()}; // 加速度向量平方和
  bool is_static_{true};                        // 当前是否静止
};

} // namespace FastVIO
