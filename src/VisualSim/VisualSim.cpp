#include <functional>
#include <iostream>
#include <random>

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
  std::vector<Point3> object_points;

  Room(const size_t n_points = 1000)
  {
    std::vector<Point2> points;
    points.reserve(n_points);

    // 1. 在矩形 rect0 范围内，使用二维均匀分布，生成 `n_points` 个二维向量 `points`
    auto rect0 = Rect<value_type>::FromCornerAndVector(
        0, 0, 2 * (height_ + depth_), 2 * height_ + width_);
    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_dist dist_u(rect0.src_u_, rect0.dst_u_);
    uniform_dist dist_v(rect0.src_v_, rect0.dst_v_);
    for (size_t i = 0; i < n_points; ++i)
    {
      points.emplace_back(dist_u(gen), dist_v(gen));
    }

    // 2. 按照子区域 rect1 ~ rect6 的范围，从中取出点集
    auto rect1
        = Rect<value_type>::FromCornerAndVector(0, height_, height_, width_);
    auto rect2
        = Rect<value_type>::FromCornerAndVector(height_, 0, depth_, height_);
    auto rect3 = Rect<value_type>::FromCornerAndVector(height_, height_, depth_,
                                                       width_);
    auto rect4 = Rect<value_type>::FromCornerAndVector(
        height_, height_ + width_, depth_, height_);
    auto rect5 = Rect<value_type>::FromCornerAndVector(
        height_ + depth_, height_, height_, width_);
    auto rect6 = Rect<value_type>::FromCornerAndVector(2 * height_ + depth_,
                                                       height_, depth_, width_);
    std::vector<std::reference_wrapper<const Point2>> subset1;
    std::vector<std::reference_wrapper<const Point2>> subset2;
    std::vector<std::reference_wrapper<const Point2>> subset3;
    std::vector<std::reference_wrapper<const Point2>> subset4;
    std::vector<std::reference_wrapper<const Point2>> subset5;
    std::vector<std::reference_wrapper<const Point2>> subset6;
    for (const Point2 &point : points)
    {
      if (rect1.Contains(point))
      {
        subset1.push_back(std::cref(point));
      }
      else if (rect2.Contains(point))
      {
        subset2.push_back(std::cref(point));
      }
      else if (rect3.Contains(point))
      {
        subset3.push_back(std::cref(point));
      }
      else if (rect4.Contains(point))
      {
        subset4.push_back(std::cref(point));
      }
      else if (rect5.Contains(point))
      {
        subset5.push_back(std::cref(point));
      }
      else if (rect6.Contains(point))
      {
        subset6.push_back(std::cref(point));
      }
    }

    // 3. 对 subset1 ~ subset6 分别作线性映射，再放进 object_points 中
    const size_t count_points{
        subset1.size() + subset2.size() + subset3.size() + subset4.size()
            + subset5.size() + subset6.size(),
    };
    object_points.reserve(count_points);
    // (u, v) -> (0, v - height_, height_ - u)
    Transform2to3 tf1{
        {0.0, 0.0, 0.0},
        {0.0, 1.0, -height_},
        {-1.0, 0.0, height_},
    };
    for (const Point2 &point : subset1)
    {
      object_points.push_back(tf1 * Point3{point(0), point(1), 1.0});
    }
    // (u, v) -> (u - height_, 0, height_ - v)
    Transform2to3 tf2{
        {1.0, 0.0, -height_},
        {0.0, 0.0, 0.0},
        {0.0, -1.0, height_},
    };
    for (const Point2 &point : subset2)
    {
      object_points.push_back(tf2 * Point3{point(0), point(1), 1.0});
    }
    // (u, v) -> (u - height_, v - height_, 0)
    Transform2to3 tf3{
        {1.0, 0.0, -height_},
        {0.0, 1.0, -height_},
        {0.0, 0.0, 0.0},
    };
    for (const Point2 &point : subset3)
    {
      object_points.push_back(tf3 * Point3{point(0), point(1), 1.0});
    }
    // (u, v) -> (u - height_, width_, v - width_ - height_)
    Transform2to3 tf4{
        {1.0, 0.0, -height_},
        {0.0, 0.0, width_},
        {0.0, 1.0, -width_ - height_},
    };
    for (const Point2 &point : subset4)
    {
      object_points.push_back(tf4 * Point3{point(0), point(1), 1.0});
    }
    // (u, v) -> (depth_, v - height_, u - height_ - depth_)
    Transform2to3 tf5{
        {0.0, 0.0, depth_},
        {0.0, 1.0, -height_},
        {1.0, 0.0, -height_ - depth_},
    };
    for (const Point2 &point : subset5)
    {
      object_points.push_back(tf5 * Point3{point(0), point(1), 1.0});
    }
    // (u, v) -> (u - 2 * height_ - depth_, v - height_, height_)
    Transform2to3 tf6{
        {1.0, 0.0, -2 * height_ - depth_},
        {0.0, 1.0, -height_},
        {0.0, 0.0, height_},
    };
    for (const Point2 &point : subset6)
    {
      object_points.push_back(tf6 * Point3{point(0), point(1), 1.0});
    }
    std::cerr << "总共生成了 " << count_points << " 个有效点\n";
  }
};

int main()
{
  Room<float> room{};
}
