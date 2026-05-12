#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>

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

  std::pair<Point3, Attitude> GetPose(const Room<value_type> &room,
                                      value_type time) const
  {
    // 1. 计算房间几何中心
    const Point3 center{room.center_};

    // 2. 设置圆周运动参数
    const value_type radius{
        std::min<value_type>(room.depth_, room.width_)
            * static_cast<value_type>(0.4),
    }; // 运动半径（留在房间内）
    const value_type angle{omega_ * time}; // 当前角度

    // 3. 计算相机本体 (Body/Base) 在世界坐标系下的位置
    // 在平行于地板的平面内运动，Z 轴高度保持在房间中心高度
    Point3 pos_body{
        center.x() + radius * std::cos(angle),
        center.y() + radius * std::sin(angle),
        center.z(),
    };
    // 初始位置: [cx+r,cy,cz]

    // 4. 计算相机本体的朝向 (Rotation)
    // 习惯上：让相机 Z 轴指向房间中心 (Look-at)，X 轴水平
    // 这里构造一个简单的旋转矩阵：
    // https://libeigen.gitlab.io/eigen/docs-nightly/group__Geometry__Module.html#gac32d7ca309f8c0ef9ae04172d49a88e6
    Point3 basis_z{(center - pos_body).normalized()};    // Z 轴
    Point3 basis_y{0.0, 0.0, -1.0};                      // Y 轴
    Point3 basis_x{basis_y.cross(basis_z).normalized()}; // X 轴
    // std::cerr                                            //
    //     << std::fixed << std::setprecision(1)            //
    //     << "\tX: [" << basis_x.x() << ", " << basis_x.y() << ", " << basis_x.z()
    //     << "]; "
    //     << "\tY: [" << basis_y.x() << ", " << basis_y.y() << ", " << basis_y.z()
    //     << "]; "
    //     << "\tZ: [" << basis_z.x() << ", " << basis_z.y() << ", " << basis_z.z()
    //     << "]; "
    //     << "\n";

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
                const Room<value_type> &room) const
  {
    const Point3 center{room.center_};
    const auto &&[pos_body, att_body] = GetPose(room, time);

    // if constexpr (false)
    {
      auto center_body{att_body.transpose() * (center - pos_body)};
      // 几何中心在体坐标系下的坐标应该恒等于 [0,0,radius]
      std::cerr << std::fixed << std::setprecision(1)
                << "\t房间几何中心在世界坐标系下的坐标 = [" << center.x()
                << ", " << center.y() << ", " << center.z() << "]\n"
                << "\t房间几何中心在体坐标系下的坐标 = [" << center_body.x()
                << ", " << center_body.y() << ", " << center_body.z() << "]\n";
      auto center_pixel_left{
          rig.camera_left_.ProjectPoint(center, att_body.transpose(),
                                        -att_body.transpose() * pos_body),
      };
      std::cerr << std::fixed << std::setprecision(1)
                << "\t房间几何中心的左目投影坐标 = [" << center_pixel_left.x()
                << ", " << center_pixel_left.y() << ", "
                << center_pixel_left.z() << "]\n";
      auto center_pixel_right{
          rig.camera_right_.ProjectPoint(center, att_body.transpose(),
                                         -att_body.transpose() * pos_body),
      };
      std::cerr << std::fixed << std::setprecision(1)
                << "\t房间几何中心的右目投影坐标 = [" << center_pixel_right.x()
                << ", " << center_pixel_right.y() << ", "
                << center_pixel_right.z() << "]\n";
    }

    return rig.Project(room.object_matrix_, att_body.transpose(),
                       -att_body.transpose() * pos_body);
  }
};
