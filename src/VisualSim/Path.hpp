#include <algorithm>
#include <print>

#include <Eigen/Dense>

#include "Room.hpp"
#include "StereoRig.hpp"

/**
 * @brief 真实轨迹
 */
template <typename value_type> struct Path
{
  using Point2   = Eigen::Vector<value_type, 2>;
  using Point3   = Eigen::Vector<value_type, 3>;
  using Attitude = Eigen::Matrix<value_type, 3, 3>;

  const value_type omega_{0.5}; // 角速度 (rad/s)
  bool print_debug_info_{true};

  /**
 * @brief 相机朝向模式
 */
  enum class OrientationMode
  {
    LookAtCenter, // 朝向圆心 (原逻辑)
    BackToCenter, // 背对圆心
    Tangent,      // 沿切线方向 (线速度方向)
    Upward,       // 朝向天花板 (垂直向上)
    StraightLine, // 直线段 (经过房间中心，平行于 x 轴)
  };

  std::pair<Point3, Attitude>
  GetPose(const Room<value_type> &room, value_type time,
          OrientationMode mode = OrientationMode::LookAtCenter) const
  {
    // 1. 计算房间几何中心
    const Point3 center{room.center_};

    // 2. 设置圆周运动参数
    const value_type radius{
        // 下面的系数必须小于 0.5
        static_cast<value_type>(mode == OrientationMode::LookAtCenter ? 0.4
                                                                      : 0.45)
            * std::min<value_type>(room.depth_, room.width_),
    }; // 运动半径（留在房间内）
    const value_type angle{omega_ * time}; // 当前角度

    // 3. 计算相机本体 (Body/Base) 在世界坐标系下的位置
    // 在平行于地板的平面内运动，Z 轴高度保持在房间中心高度
    Point3 pos_body{Point3::Zero()};

    pos_body.x() = center.x() + radius * std::cos(angle);
    pos_body.z() = center.z();
    if (mode == OrientationMode::StraightLine)
    {
      pos_body.y() = center.y();
    }
    else
    {
      pos_body.y() = center.y() + radius * std::sin(angle);
      if (mode != OrientationMode::LookAtCenter)
      {
        pos_body.z() -= static_cast<value_type>(0.3 * room.height_);
      }
    }

    // 初始位置: [cx+r,cy,cz]

    if (print_debug_info_)
    {
      std::print(stderr,
                 "[DEBUG] (匀速圆周运动) 运动参数:\n"
                 "\t圆心在世界坐标系下的坐标 = [{:.1f}, {:.1f}, {:.1f}]\n"
                 "\t角速率 = {:.1f}\n"
                 "\t轨迹半径 = {:.1f}\n",
                 center.x(), center.y(), center.z(), omega_, radius);
    }

    // 4. 计算相机本体的朝向 (Rotation)
    // 习惯上：让相机 Z 轴指向房间中心 (Look-at)，X 轴水平
    // 这里构造一个简单的旋转矩阵：
    // https://libeigen.gitlab.io/eigen/docs-nightly/group__Geometry__Module.html#gac32d7ca309f8c0ef9ae04172d49a88e6
    Point3 basis_x;
    Point3 basis_y;
    Point3 basis_z;

    switch (mode)
    {
    case OrientationMode::LookAtCenter:
    {
      basis_z = (center - pos_body).normalized();    // 朝向圆心
      basis_y = {0.0, 0.0, -1.0};                    // 朝向地心
      basis_x = basis_y.cross(basis_z).normalized(); // 朝向右侧
      break;
    }
    case OrientationMode::BackToCenter:
    {
      basis_z = (pos_body - center).normalized();    // 背对圆心
      basis_y = {0.0, 0.0, -1.0};                    // 朝向地心
      basis_x = basis_y.cross(basis_z).normalized(); // 朝向右侧
      break;
    }
    case OrientationMode::Tangent:
    {
      // 逆时针运动的切线方向
      basis_z = Point3{-std::sin(angle), std::cos(angle), 0.0};
      basis_y = {0.0, 0.0, -1.0};                    // 朝向地心
      basis_x = basis_y.cross(basis_z).normalized(); // 朝向右侧
      break;
    }
    case OrientationMode::Upward:
    {
      basis_z = {0.0, 0.0, 1.0};                     // 指向天花板
      basis_x = (center - pos_body).normalized();    // 朝向圆心
      basis_y = basis_z.cross(basis_x).normalized(); // 朝向运动的反方向
      break;
    }
    }

    Attitude att_body;
    // 体坐标系的三个基向量在世界坐标系下的坐标，组成了从世界坐标系到体坐标系的旋转变换在世界坐标系下的矩阵表示
    att_body.col(0) = basis_x;
    att_body.col(1) = basis_y;
    att_body.col(2) = basis_z;

    return {pos_body, att_body};
  }

  /**
   * @brief 让双目相机 rig 绕着房间的几何中心，在平行于地板的平面内，做匀速圆周运动
   */
  auto GetImage(const StereoRig<value_type> &rig, value_type time,
                const Room<value_type> &room,
                OrientationMode mode = OrientationMode::LookAtCenter) const
  {
    const Point3 center{room.center_};
    const auto &&[pos_body, att_body] = GetPose(room, time, mode);

    // if constexpr (false)
    // {
    //   auto center_body{att_body.transpose() * (center - pos_body)};
    //   auto center_pixel_left{
    //       rig.camera_left_.ProjectPoint(center, att_body.transpose(),
    //                                     -att_body.transpose() * pos_body),
    //   };
    //   auto center_pixel_right{
    //       rig.camera_right_.ProjectPoint(center, att_body.transpose(),
    //                                      -att_body.transpose() * pos_body),
    //   };
    //   // 当相机对准几何中心时，几何中心在体坐标系下的坐标应该恒等于 [0,0,radius]
    //   std::print(
    //       stderr,
    //       "\t房间几何中心在世界坐标系下的坐标 = [{:.1f}, {:.1f}, {:.1f}]\n"
    //       "\t房间几何中心在体坐标系下的坐标 = [{:.1f}, {:.1f}, {:.1f}]\n"
    //       "\t房间几何中心的左目投影坐标 = [{:.1f}, {:.1f}, {:.1f}]\n"
    //       "\t房间几何中心的右目投影坐标 = [{:.1f}, {:.1f}, {:.1f}]\n",
    //       center.x(), center.y(), center.z(),                //
    //       center_body.x(), center_body.y(), center_body.z(), //
    //       center_pixel_left.x(), center_pixel_left.y(),
    //       center_pixel_left.z(), //
    //       center_pixel_right.x(), center_pixel_right.y(),
    //       center_pixel_right.z());
    // }

    return rig.Project(room.object_matrix_, att_body.transpose(),
                       -att_body.transpose() * pos_body);
  }
};
