#pragma once

#include <Eigen/Dense>

template <typename value_type> struct Rect
{
  const value_type src_u_;
  const value_type src_v_;
  const value_type dst_u_;
  const value_type dst_v_;

  static Rect FromOppositeCorners(value_type src_u, value_type src_v,
                                  value_type dst_u, value_type dst_v)
  {
    return {
        src_u,
        src_v,
        dst_u,
        dst_v,
    };
  }

  static Rect FromCornerAndVector(value_type src_u, value_type src_v,
                                  value_type delta_u, value_type delta_v)
  {
    return FromOppositeCorners(src_u, src_v, src_u + delta_u, src_v + delta_v);
  }

  bool Contains(const Eigen::Vector<value_type, 2> &point) const
  {
    const value_type px{point(0)};
    const value_type py{point(1)};
    return src_u_ <= px && px < dst_u_ && src_v_ <= py && py < dst_v_;
  }
};
