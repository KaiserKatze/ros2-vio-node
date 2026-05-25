#pragma once

#include <array>
#include <cstddef>

/**
 * @brief 可遍历的无锁环形缓冲区（固定容量）
 *
 * 特点：
 * - 使用单调递增的无符号计数器（head_/tail_），避免取模分支
 * - 容量 N 必须为 2 的幂，利用位运算 (index & (N-1)) 实现快速取模
 * - 提供真正“逻辑顺序”的迭代器（按时间顺序遍历）
 */
template <typename T, std::size_t N> class CircularBuffer
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

    const_iterator(const CircularBuffer *buf, std::size_t pos) :
      buf_{buf}, pos_{pos}
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
    std::size_t pos_;
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
    const std::size_t index{tail_ & (N - 1)};
    const T old{data_[index]};
    data_[index] = value;

    // 计算是否满（tail - head == N）
    const std::size_t size{tail_ - head_};

    // mask = 0 或 ~0（全1），无分支选择
    const std::size_t mask{static_cast<std::size_t>(-(size == N))};

    // 若满，则 head_ += 1，否则 +=0（无分支）
    head_ += (mask & 1);

    ++tail_;

    return old;
  }

  /**
   * @brief 当前元素数量
   */
  std::size_t size() const
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
  std::size_t head_{0}; // 单调递增
  std::size_t tail_{0}; // 单调递增
};
