#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <meta>
#include <print>
#include <stdexcept>

// 辅助工具：判断一个反射类型是否是特定的 Eigen 类型
template <typename T>
consteval bool is_type_of(std::meta::info member_reflect)
{
  return std::meta::type_of(member_reflect) == ^^T;
}

// 插值查找时间戳最近的数据（C++26 反射全自动化版）
template <typename DataType>
static DataType Interpolate(const std::vector<DataType> &data,
                            const std::int64_t timestamp)
{
  if (data.empty())
  {
    throw std::runtime_error("数据为空!");
  }
  if (timestamp <= data.front().timestamp_)
  {
    return data.front();
  }
  if (timestamp >= data.back().timestamp_)
  {
    return data.back();
  }

  // 标准二分查找定位左右区间
  size_t left{0}, right{data.size() - 1};
  while (left + 1 < right)
  {
    size_t mid{(left + right) / 2};
    if (data[mid].timestamp_ < timestamp)
    {
      left = mid;
    }
    else
    {
      right = mid;
    }
  }

  const auto &datum0{data[left]};
  const auto &datum1{data[right]};
  const double t{static_cast<double>(timestamp)};
  const double t0{static_cast<double>(datum0.timestamp_)};
  const double t1{static_cast<double>(datum1.timestamp_)};
  const double alpha{(t1 > t0) ? std::clamp((t - t0) / (t1 - t0), 0.0, 1.0)
                               : 0.0};

  DataType interp;
  template for (constexpr auto /* std::meta::info */ member :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^DataType, std::meta::access_context::current()
                )))
  {
    // 如果成员是时间戳本身，直接赋值为目标时间戳
    if constexpr (std::meta::identifier_of(member) == "timestamp_")
    {
      interp.[:member:] = timestamp;
    }
    // 如果成员是 Eigen::Vector3f，执行线性插值 (LERP)
    else if constexpr (is_type_of<Eigen::Vector3f>(member))
    {
      interp.[:member:] = datum0.[:member:]
          * (1.0f - static_cast<float>(alpha)) + datum1.[:member:]
          * static_cast<float>(alpha);
    }
    // 如果成员是 Eigen::Vector3d，执行线性插值 (LERP)
    else if constexpr (is_type_of<Eigen::Vector3d>(member))
    {
      interp.[:member:]
          = datum0.[:member:] * (1.0 - alpha) + datum1.[:member:] * alpha;
    }
    // 如果成员是 Eigen::Quaternionf，执行球面线性插值 (SLERP)
    else if constexpr (is_type_of<Eigen::Quaternionf>(member))
    {
      interp.[:member:] = datum0.[:member:].slerp(static_cast<float>(alpha),
                                                  datum1.[:member:]);
    }
    // 如果成员是 Eigen::Quaterniond，执行球面线性插值 (SLERP)
    else if constexpr (is_type_of<Eigen::Quaterniond>(member))
    {
      interp.[:member:] = datum0.[:member:].slerp(alpha, datum1.[:member:]);
    }
  }

  return interp;
}
