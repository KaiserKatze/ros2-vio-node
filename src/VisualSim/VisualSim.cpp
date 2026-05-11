#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>

#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <sstream>
#include <stdexcept>

#include "MeshPlot.hpp"
#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"

template <typename value_type> struct VisualSim
{
  // 传入长宽高的划分段数
  Room<value_type> room_{10, 10, 6};
  // 初始化专属绘制器
  MeshPlot<value_type> mesh_plot_;
  // 仿真双目相机
  StereoRig<value_type> rig_{};
  // 仿真双目相机运动路径
  Path<value_type> path_{};

  VisualSim() : mesh_plot_{room_}
  {
    // 只修改双目相机的基线长度
    rig_.camera_right_.translation_ = {-0.1, 0.0, 0.0};
  }

  void Start()
  {
    for (value_type time = 0.0; time < 50.0; time += 0.1)
    {
      std::cerr << "[INFO] 时间 = (" << std::fixed << std::setprecision(1)
                << time << ").\n";

      // 直接拿到可见顶点的全局索引，而不是三维坐标系本身了
      auto &&[visible_object_indices, image_points_left, image_points_right]
          = path_.GetImage(rig_, time, room_);
      std::cerr << "\t当前场景中，双目可见路标点有 "
                << visible_object_indices.size() << " 个.\n";

      // 绘制相机图像
      {
        const cv::Scalar background_gray{128, 128, 128};
        cv::Mat cv_image_left{
            rig_.camera_left_.height_,
            rig_.camera_left_.width_,
            CV_8UC3,
            background_gray,
        };
        cv::Mat cv_image_right{
            rig_.camera_right_.height_,
            rig_.camera_right_.width_,
            CV_8UC3,
            background_gray,
        };

        // 核心绘制逻辑收口
        mesh_plot_.Draw(cv_image_left, cv_image_right, visible_object_indices,
                        image_points_left, image_points_right);

        if (mesh_plot_.Render(cv_image_left, cv_image_right))
        {
          break;
        }
      }
    }
  }
};

int main()
{
  try
  {
    VisualSim<float>{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::cerr << ex.what() << "\n";
  }
  return 0;
}
