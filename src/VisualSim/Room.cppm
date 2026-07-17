module;

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

export module FastVIO.VisualSim:Room;

// import std;

namespace FastVIO::VisualSim
{

template <typename value_type>
struct Room
{
  using Point2  = Eigen::Vector<value_type, 2>;
  using Point3  = Eigen::Vector<value_type, 3>;
  using Point3i = std::array<int, 3>;

  static bool PointComparator(const Point3 &p1, const Point3 &p2)
  {
    return std::tie(p1(0), p1(1), p1(2)) < std::tie(p2(0), p2(1), p2(2));
  }

  const value_type width_{10.0}; // 开间
  const value_type depth_{10.0}; // 进深
  const value_type height_{3.0}; // 层高
  const Point3 center_{
      //房间的几何中心
      depth_ * static_cast<value_type>(0.5),
      width_ *static_cast<value_type>(0.5),
      height_ *static_cast<value_type>(0.5),
  };

  const int cnt_sep_depth_;
  const int cnt_sep_width_;
  const int cnt_sep_height_;
  const value_type step_d_;
  const value_type step_w_;
  const value_type step_h_;

  // 路标点在基 {(step_d_,0,0),(0,step_w_,0),(0,0,step_h_)} 下的坐标的列表
  std::vector<Point3i> object_points_;
  // 以路标点为列向量构成的 3xN 矩阵
  Eigen::Matrix<value_type, 3, Eigen::Dynamic> object_matrix_;

  /**
   * @brief 按照网格生成路标点
   * @note 网格间距，默认 0.5 米
   */
  Room(int cnt_sep_depth = 20, int cnt_sep_width = 20, int cnt_sep_height = 6) :
    cnt_sep_depth_{cnt_sep_depth}, cnt_sep_width_{cnt_sep_width},
    cnt_sep_height_{cnt_sep_height}, step_d_{depth_ / cnt_sep_depth_},
    step_w_{width_ / cnt_sep_width_}, step_h_{height_ / cnt_sep_height_}
  {
    if (cnt_sep_depth <= 0 || cnt_sep_width <= 0 || cnt_sep_height <= 0)
    {
      throw std::invalid_argument{"切分段数应该是正数!"};
    }
    if (cnt_sep_depth % 2 != 0 || cnt_sep_width % 2 != 0
        || cnt_sep_height % 2 != 0)
    {
      throw std::invalid_argument{"切分段数应该是偶数!"};
    }

    //====================================
    // 路标点生成

    // Floor & Ceiling (XY planes)
    for (int ix = 0; ix <= cnt_sep_depth_; ++ix)
    {
      for (int iy = 0; iy <= cnt_sep_width_; ++iy)
      {
        object_points_.push_back({ix, iy, 0});
        object_points_.push_back({ix, iy, cnt_sep_height_});
      }
    }

    // Walls (YZ planes at x=0 and x=depth)
    for (int iy = 0; iy <= cnt_sep_width_; ++iy)
    {
      for (int iz = 0; iz <= cnt_sep_height_; ++iz)
      {
        object_points_.push_back({0, iy, iz});
        object_points_.push_back({cnt_sep_depth_, iy, iz});
      }
    }

    // Walls (XZ planes at y=0 and y=width)
    for (int ix = 0; ix <= cnt_sep_depth_; ++ix)
    {
      for (int iz = 0; iz <= cnt_sep_height_; ++iz)
      {
        object_points_.push_back({ix, 0, iz});
        object_points_.push_back({ix, cnt_sep_width_, iz});
      }
    }

    //====================================
    // 路标点去重

    // 保证列表已经排序
    std::sort(object_points_.begin(), object_points_.end());
    // 使用 std::unique 移动重复元素到末尾
    auto last = std::unique(object_points_.begin(), object_points_.end());
    // 删除末尾的多余元素
    object_points_.erase(last, object_points_.end());

    //====================================
    // 更新矩阵表示

    const std::size_t total{object_points_.size()};
    object_matrix_.resize(3, total);
    for (std::size_t i = 0; i < total; ++i)
    {
      const auto point{object_points_[i]};
      object_matrix_.col(i) = Point3{
          point[0] * step_d_,
          point[1] * step_w_,
          point[2] * step_h_,
      };
    }

    std::print(stderr, "网格点生成完毕，共计 {} 个点.\n", total);
  }

  // 辅助函数：通过整数坐标查找其在矩阵中的索引
  auto GetIndex(const Point3i &p) const
  {
    // 利用已有序的 object_points_ 通过二分查找快速拿到全局索引，避开浮点比较
    auto it{std::lower_bound(object_points_.begin(), object_points_.end(), p)};
    if (it != object_points_.end() && *it == p)
    {
      return static_cast<std::size_t>(std::distance(object_points_.begin(), it));
    }
    std::stringstream ss;
    ss << "找不到点 [" << p[0] << ", " << p[1] << ", " << p[2]
       << "] 对应的路标点索引!";
    throw std::runtime_error(ss.str());
  }
};

} // namespace FastVIO::VisualSim
