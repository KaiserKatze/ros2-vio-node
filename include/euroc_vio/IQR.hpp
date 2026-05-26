#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

template <typename T /* Eigen::Vector */, typename U /* float or double */>
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
