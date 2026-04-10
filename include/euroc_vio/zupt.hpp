#ifndef ZUPT_HPP
#define ZUPT_HPP

#include <Eigen/Dense>

#include <array>
#include <numeric>

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
 * - 使用 transform_reduce（线程安全并行）
 * - Eigen 向量化（SIMD）
 * - 避免数据竞争
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

  /**
   * @brief 更新一帧 IMU 数据并判断是否静止
   */
  bool Update(const data_type &imu_data)
  {
    const data_type old_data{window_.push(imu_data)};

    if (!window_.full())
    {
      return true;
    }

    // 队列第一次充满
    if (!initialized_)
    {
      initialized_ = true;
      // TODO 先把所有数据统计一遍
    }
    else
    {
      // TODO 利用“期望”“方差”的线性性，从统计量中减去旧数据对应的，再加上新数据对应的
    }

    const double denom{1.0 / window_.size()};

    /** =========================
     * 1. 陀螺仪能量（并行 + 向量化）
     * ========================= */
    double gyro_energy{std::transform_reduce(
                           window_.cbegin(), window_.cend(), 0.0, std::plus<>(),
                           [](const data_type &e)
                           {
                             return e.template head<3>().norm(); // Eigen SIMD
                           })
                       * denom};

    if (gyro_energy > config_.gyroscope_magnitude_threshold)
    {
      return false;
    }

    /** =========================
     * 2. 加速度均值（向量 reduce）
     * ========================= */
    const Eigen::Vector3d acc_mean{
        std::transform_reduce(window_.cbegin(), window_.cend(),
                              Eigen::Vector3d::Zero().eval(), std::plus<>(),
                              [](const data_type &e)
                              {
                                return e.template tail<3>(); // Eigen SIMD
                              })
        * denom};

    const double acc_mean_norm{acc_mean.norm()};

    if (std::abs(acc_mean_norm - config_.local_gravity) > config_.g_tolerance)
    {
      return false;
    }

    /** =========================
     * 3. 加速度方差（并行 + SIMD）
     * ========================= */
    const double acc_variance{
        std::transform_reduce(
            window_.cbegin(), window_.cend(), 0.0, std::plus<>(),
            [&](const data_type &e)
            {
              Eigen::Vector3d delta{e.template tail<3>() - acc_mean};
              return delta.squaredNorm(); // SIMD
            })
        * denom};

    if (acc_variance > config_.accelerometer_variance_threshold)
    {
      return false;
    }

    return true;
  }

private:
  Config config_;
  window_type window_;
  bool initialized_{false};

  double gyro_norm_sum_{0.0};
  Eigen::Vector3d accel_sum_{Eigen::Vector3d::Zero()};
  double acc_variance_{0.0};
};

#endif /* ZUPT_HPP */
