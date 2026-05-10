#include <iomanip>
#include <iostream>
#include <type_traits>

#include <Eigen/Dense>

#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"

int main()
{
  Room<float> room{1};

  // 打印路标点
  {
    std::cerr << "Object Points =[\n";
    const auto object_points{room.object_matrix};
    const auto len{room.object_matrix.cols()};
    for (std::remove_cv_t<decltype(len)> i = 0; i < len; ++i)
    {
      const auto &object_point{object_points.col(i)};
      std::cerr << "\t[" << object_point.x() << ", " << object_point.y() << ", "
                << object_point.z() << "];\n";
    }
    std::cerr << "];\n";
  }

  StereoRig<float> rig{};
  // 基线长度 (两个相机的光心的间距) 定为 0.1 米
  rig.camera_right_.translation_ = {-0.1, 0.0, 0.0};

  // std::cerr << "左目相机内参矩阵 =\n"
  //           << rig.camera_left_.intrinsic_ << "\n"
  //           << "左目相机旋转矩阵 =\n"
  //           << rig.camera_left_.rotation_ << "\n"
  //           << "左目相机平移向量 =\n"
  //           << rig.camera_left_.translation_ << "\n"
  //           << "右目相机内参矩阵 =\n"
  //           << rig.camera_right_.intrinsic_ << "\n"
  //           << "右目相机旋转矩阵 =\n"
  //           << rig.camera_right_.rotation_ << "\n"
  //           << "右目相机平移向量 =\n"
  //           << rig.camera_right_.translation_ << "\n";

  Path<float> path{};
  for (float time = 0.0; time < 5.0; time += 1.0)
  {
    std::cerr << "[INFO] 时间 = (" << std::fixed << std::setprecision(1) << time
              << ").\n";
    auto &&[object_points, image_points_left, image_points_right]
        = path.GetImage(rig, time, room);
    std::cerr << "\t当前场景中，双目可见路标点有 " << object_points.size()
              << " 个.\n";
  }
  return 0;
}
