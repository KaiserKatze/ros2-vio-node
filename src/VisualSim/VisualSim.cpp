#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
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

/**
 * @brief 相机
 */
template <typename value_type> struct Camera
{
  Eigen::Matrix<value_type, 3, 3> intrinsic_;
  Eigen::Matrix<value_type, 3, 3> rotation_;
  Eigen::Vector<value_type, 3> translation_;
};

/**
 * @brief 双目相机
 */
template <typename value_type> struct StereoRig
{
  Camera<value_type> camera_left_;
  Camera<value_type> camera_right_;
};

/**
 * @brief 真实轨迹
 */
template <typename value_type> struct Path
{
  using Point2 = Eigen::Vector<value_type, 2>;
  using Point3 = Eigen::Vector<value_type, 3>;
  using Point4 = Eigen::Vector<value_type, 4>;

  static std::vector<std::pair<Point3, Point2>>
  GetCurrentImage(const Eigen::Matrix<value_type, 3, 3> &intrinsic_matrix,
                  const Eigen::Matrix<value_type, 3, 4> &extrinsic_matrix,
                  const int width, const int height,
                  const Room<value_type> &room)
  {
    // 将三维点的非齐次坐标转换为齐次坐标
    auto n_points{room.object_matrix.cols()};
    Eigen::Matrix<value_type, 4, Eigen::Dynamic> object_matrix_homo(4,
                                                                    n_points);
    object_matrix_homo(Eigen::seq(0, 2), Eigen::all) = room.object_matrix;
    object_matrix_homo.row(3).fill(1.0);

    // 投影到相机坐标系，得到齐次坐标
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_matrix_homo{
        intrinsic_matrix * extrinsic_matrix * object_matrix_homo,
    };

    // 检查三维点是否处于相机视域内
    std::vector<std::pair<Point3, Point2>> visible_points;
    visible_points.reserve(n_points);

    for (decltype(n_points) i = 0; i < n_points; ++i)
    {
      const Point3 point{pixel_matrix_homo.col(i)};
      const value_type w{point(2)};
      // 深度测试: 点必须在相机的前方
      if (w <= std::numeric_limits<value_type>::epsilon())
      {
        continue;
      }
      const value_type u{point(0) / w};
      const value_type v{point(1) / w};
      // 边界测试: 投影点在成像范围内
      if (0.0 < u && u < static_cast<value_type>(width) && 0.0 < v
          && v < static_cast<value_type>(height))
      {
        visible_points.emplace_back(point, Point2{u, v});
      }
    }

    return visible_points;
  }

  static std::tuple<std::vector<Point3>, std::vector<Point2>,
                    std::vector<Point2>>
  GetCurrentStereoImage(
      const StereoRig<value_type> &rig,
      const Eigen::Matrix<value_type, 4, 4> &rigid_transform_body,
      const int width, const int height, const Room<value_type> &room)
  {
    using Transform4 = Eigen::Matrix<value_type, 4, 4>;
    Transform4 rigid_transform_left{Transform4::Identity()};
    Transform4 rigid_transform_right{Transform4::Identity()};

    rigid_transform_left.template block<3, 3>(0, 0)
        = rig.camera_left_.rotation_;
    rigid_transform_left.template block<3, 1>(0, 3)
        = rig.camera_left_.translation_;
    rigid_transform_right.template block<3, 3>(0, 0)
        = rig.camera_right_.rotation_;
    rigid_transform_right.template block<3, 1>(0, 3)
        = rig.camera_right_.translation_;
    rigid_transform_left  = rigid_transform_body * rigid_transform_left;
    rigid_transform_right = rigid_transform_body * rigid_transform_right;

    std::vector<std::pair<Point3, Point2>> correspondence_left
        = GetCurrentImage(rig.camera_left_.intrinsic_,
                          rigid_transform_left.template block<3, 4>(0, 0),
                          width, height, room);
    std::vector<std::pair<Point3, Point2>> correspondence_right
        = GetCurrentImage(rig.camera_right_.intrinsic_,
                          rigid_transform_right.template block<3, 4>(0, 0),
                          width, height, room);

    // 左目、右目视图中可见三维点可能不一样，需要取交集
    std::vector<Point3> common_object_points;
    std::vector<Point2> common_image_left;
    std::vector<Point2> common_image_right;

    // 使用 Map 存储右目可见点，Key 是三维点的坐标（利用 Eigen 的比较逻辑）
    // 如果点数极多，可以考虑基于索引匹配，但这里基于坐标比对最通用
    auto comp = [](const Point3 &a, const Point3 &b)
    { return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2)); };
    std::map<Point3, Point2, decltype(comp)> right_lookup(comp);

    for (const auto &pair : correspondence_right)
    {
      right_lookup[pair.first] = pair.second;
    }

    // 遍历左目点，如果在右目也存在，则存入结果
    for (const auto &pair : correspondence_left)
    {
      const Point3 &p_world = pair.first;
      auto it               = right_lookup.find(p_world);
      if (it != right_lookup.end())
      {
        common_object_points.push_back(p_world);
        common_image_left.push_back(pair.second);
        common_image_right.push_back(it->second);
      }
    }

    return {common_object_points, common_image_left, common_image_right};
  }

  /**
   * @brief 让双目相机 rig 绕着房间的几何中心，在平行于地板的平面内，做匀速圆周运动
   */
  auto GetCurrentStereoImage(const StereoRig<value_type> &rig, value_type time,
                             const int width, const int height,
                             const Room<value_type> &room)
  {
    // 1. 计算房间几何中心
    const Point3 center(room.depth_ * 0.5, room.width_ * 0.5,
                        room.height_ * 0.5);

    // 2. 设置圆周运动参数
    const value_type radius
        = std::min(room.depth_, room.width_) * 0.4; // 运动半径（留在房间内）
    const value_type omega = 0.5;                   // 角速度 (rad/s)
    const value_type angle = omega * time;          // 当前角度

    // 3. 计算相机本体 (Body/Base) 在世界坐标系下的位置
    // 在平行于地板的平面内运动，Z 轴高度保持在房间中心高度
    Point3 body_pos(center.x() + radius * std::cos(angle),
                    center.y() + radius * std::sin(angle), center.z());

    // 4. 计算相机本体的朝向 (Rotation)
    // 习惯上：让相机 Z 轴指向房间中心 (Look-at)，X 轴水平
    // 这里构造一个简单的旋转矩阵：
    Point3 forward = (center - body_pos).normalized(); // 新的 Z 轴
    Point3 world_up(0, 0, 1);
    Point3 right = world_up.cross(forward).normalized(); // 新的 X 轴
    Point3 up    = forward.cross(right).normalized();    // 新的 Y 轴

    Eigen::Matrix<value_type, 3, 3> R_body;
    R_body.col(0) = right;
    R_body.col(1) = up;
    R_body.col(2) = forward;

    // 5. 构造 4x4 变换矩阵 T_world_body (Body -> World)
    // 注意：Path 类中 GetCurrentStereoImage 内部使用的是 rigid_transform_body
    // 该矩阵应为 Body 到 World 的坐标变换
    Eigen::Matrix<value_type, 4, 4> T_world_body
        = Eigen::Matrix<value_type, 4, 4>::Identity();
    T_world_body.template block<3, 3>(0, 0) = R_body;
    T_world_body.template block<3, 1>(0, 3) = body_pos;

    // 6. 调用已有的双目投影函数
    // 注意：原代码中的 GetCurrentStereoImage 内部会将该矩阵与相机外参相乘
    return GetCurrentStereoImage(rig, T_world_body, width, height, room);
  }
};

int main()
{
  Room<float> room{1.0};
  std::cerr << "Object Points =\n" << room.object_matrix.transpose() << "\n";
  StereoRig<float> rig{};
  auto &&[object_points, image_points_left, image_points_right]
      = Path<float>{}.GetCurrentStereoImage(rig, 0.0, 680, 480, room);
  std::cerr << "当前场景中，可见路标点有 " << object_points.size() << " 个.\n";
  return 0;
}
