#ifndef ZUPT_HPP
#define ZUPT_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <array>

/**
 * @brief 可遍历的无锁环形缓冲区（固定容量）
 *
 * 特点：
 * - 使用单调递增的无符号计数器（head_/tail_），避免取模分支
 * - 容量 N 必须为 2 的幂，利用位运算 (index & (N-1)) 实现快速取模
 * - 提供真正“逻辑顺序”的迭代器（按时间顺序遍历）
 */
template <typename T, size_t N> class CircularBuffer
{
  static_assert(N > 0, "Capacity must be positive");
  static_assert((N & (N - 1)) == 0,
                "Capacity must be power of two for bitmask optimization");

public:
  using value_type     = T;
  using container_type = std::array<T, N>;

  /**
   * @brief 自定义 const 迭代器（按逻辑顺序遍历）
   */
  class const_iterator
  {
  public:
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = const T *;
    using reference         = const T &;
    using iterator_category = std::forward_iterator_tag;

    const_iterator(const CircularBuffer *buf, size_t pos) : buf_{buf}, pos_{pos}
    {
    }

    reference operator*() const
    {
      return buf_->data_[pos_ & (N - 1)];
    }

    pointer operator->() const
    {
      return &(**this);
    }

    const_iterator &operator++()
    {
      ++pos_;
      return *this;
    }

    const_iterator operator++(int)
    {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const const_iterator &other) const
    {
      return pos_ == other.pos_;
    }

    bool operator!=(const const_iterator &other) const
    {
      return !(*this == other);
    }

  private:
    const CircularBuffer *buf_;
    size_t pos_;
  };

  /**
   * @brief 入队（无分支覆盖最旧数据），返回被覆盖前的数据
   *
   * 利用：
   * - 无符号整数自然溢出
   * - XOR + 位运算避免显式 if(full())
   */
  T push(const T &value)
  {
    const size_t index{tail_ & (N - 1)};
    const T old{data_[index]};
    data_[index] = value;

    // 计算是否满（tail - head == N）
    const size_t size{tail_ - head_};

    // mask = 0 或 ~0（全1），无分支选择
    const size_t mask{static_cast<size_t>(-(size == N))};

    // 若满，则 head_ += 1，否则 +=0（无分支）
    head_ += (mask & 1);

    ++tail_;

    return old;
  }

  /**
   * @brief 当前元素数量
   */
  size_t size() const
  {
    return tail_ - head_;
  }

  /**
   * @brief 是否已满
   */
  bool full() const
  {
    return size() == N;
  }

  /**
   * @brief 起始迭代器（最旧元素）
   */
  const_iterator cbegin() const
  {
    return const_iterator(this, head_);
  }

  /**
   * @brief 结束迭代器（尾后）
   */
  const_iterator cend() const
  {
    return const_iterator(this, tail_);
  }

private:
  container_type data_{};
  size_t head_{0}; // 单调递增
  size_t tail_{0}; // 单调递增
};

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
  };

  ZUPT() : ZUPT(Config{}) {}
  ZUPT(Config &&config) : config_{std::move(config)} {}

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

  /**
   * @brief 更新一帧 IMU 数据并判断是否静止
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
      return true;
    }

    const double denom{1.0 / static_cast<double>(WindowSize)};

    /** =========================
     * 陀螺仪能量判断
     * ========================= */
    const double gyro_energy{gyro_norm_sum_ * denom};

    if (gyro_energy > config_.gyroscope_magnitude_threshold)
    {
      return false;
    }

    /** =========================
     * 加速度均值判断
     * ========================= */
    const Eigen::Vector3d acc_mean{accel_sum_ * denom};
    const double acc_mean_norm{acc_mean.norm()};

    if (std::abs(acc_mean_norm - config_.local_gravity) > config_.g_tolerance)
    {
      return false;
    }

    /** =========================
     * 加速度各分量方差判断 (D(X) = E(X^2) - [E(X)]^2)
     * ========================= */
    const Eigen::Vector3d acc_mean_sq{
        acc_mean.array().square().matrix()}; // 逐元素平方
    Eigen::Vector3d acc_variance{accel_sq_sum_ * denom - acc_mean_sq};
    // 抵消由于浮点数精度累积误差可能导致的极微小负数
    double acc_variance_max_component{
        std::max(0.0, acc_variance.maxCoeff())}; // 取最大分量方差

    if (acc_variance_max_component > config_.accelerometer_variance_threshold)
    {
      return false;
    }

    return true;
  }

private:
  Config config_;
  window_type window_;

  // 滑动窗口的全局状态变量
  double gyro_norm_sum_{0.0};                             // 角速度向量范数和
  Eigen::Vector3d accel_sum_{Eigen::Vector3d::Zero()};    // 加速度向量和
  Eigen::Vector3d accel_sq_sum_{Eigen::Vector3d::Zero()}; // 加速度向量平方和
};

#endif /* ZUPT_HPP */
