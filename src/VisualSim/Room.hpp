#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include <Eigen/Dense>

/**
 * @brief 按照指定参数 `(width_, depth_, height_)` 随机生成一个长方体房间，
          在地板、天花板和四周墙面上随机生成路标点
 */
template <typename value_type> struct Room
{
  using Point2        = Eigen::Vector<value_type, 2>;
  using Point3        = Eigen::Vector<value_type, 3>;
  using Transform2to3 = Eigen::Matrix<value_type, 3, 3>;
  using uniform_dist  = std::uniform_real_distribution<value_type>;

  const value_type width_{5.0};  // 开间
  const value_type depth_{5.0};  // 进深
  const value_type height_{3.0}; // 层高
  const Point3 center_{
      //房间的几何中心
      depth_ * static_cast<value_type>(0.5),
      width_ *static_cast<value_type>(0.5),
      height_ *static_cast<value_type>(0.5),
  };
  // 以路标点为列向量构成的 3xN 矩阵
  Eigen::Matrix<value_type, 3, Eigen::Dynamic> object_matrix;

  /**
   * @brief 按照网格生成路标点
   * @param step 网格间距，默认 0.5 米
   */
  Room(const value_type step = 0.5)
  {
    std::vector<Point3> object_points;

    //====================================
    // 路标点生成

    // Floor & Ceiling (XY planes)
    for (value_type x = 0; x <= depth_; x += step)
    {
      for (value_type y = 0; y <= width_; y += step)
      {
        object_points.emplace_back(x, y, 0);
        object_points.emplace_back(x, y, height_);
      }
    }

    // Walls (YZ planes at x=0 and x=depth)
    // 避免重复生成棱线上的点，调整 y 和 z 的起始范围（可选）
    for (value_type y = 0; y <= width_; y += step)
    {
      for (value_type z = 0; z <= height_; z += step)
      {
        object_points.emplace_back(0, y, z);
        object_points.emplace_back(depth_, y, z);
      }
    }

    // Walls (XZ planes at y=0 and y=width)
    for (value_type x = 0; x <= depth_; x += step)
    {
      for (value_type z = 0; z <= height_; z += step)
      {
        object_points.emplace_back(x, 0, z);
        object_points.emplace_back(x, width_, z);
      }
    }

    //====================================
    // 路标点去重

    // 保证列表已经排序
    std::sort(object_points.begin(), object_points.end(),
              [](const Point3 &p1, const Point3 &p2)
              {
                return std::tie(p1(0), p1(1), p1(2))
                       < std::tie(p2(0), p2(1), p2(2));
              });
    // 使用 std::unique 移动重复元素到末尾
    auto last = std::unique(object_points.begin(), object_points.end(),
                            [atol = 1e-4](const Point3 &p1, const Point3 &p2)
                            {
                              // 对于浮点数，建议使用一个极小的阈值 epsilon 判断相等
                              return (p1 - p2).norm() < atol;
                            });
    // 删除末尾的多余元素
    object_points.erase(last, object_points.end());

    // 加入房间几何中心
    object_points.push_back(center_);

    //====================================
    // 更新矩阵表示

    const size_t total = object_points.size();
    object_matrix.resize(3, total);
    for (size_t i = 0; i < total; ++i)
    {
      object_matrix.col(i) = object_points[i];
    }

    std::cerr << "网格点生成完毕，共计 " << total << " 个点。\n";
  }
};
