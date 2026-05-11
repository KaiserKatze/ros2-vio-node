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
    cv::Mat cv_projection_matrix_left;
    cv::Mat cv_projection_matrix_right;

    {
      Eigen::Matrix<value_type, 3, 4> eigen_projection_matrix_left;
      Eigen::Matrix<value_type, 3, 4> eigen_projection_matrix_right;

      eigen_projection_matrix_left.template block<3, 3>(0, 0) //
          = rig_.camera_left_.rotation_;
      eigen_projection_matrix_left.template block<3, 1>(0, 3) //
          = rig_.camera_left_.translation_;
      eigen_projection_matrix_left
          = rig_.camera_left_.intrinsic_ * eigen_projection_matrix_left;

      eigen_projection_matrix_right.template block<3, 3>(0, 0) //
          = rig_.camera_right_.rotation_;
      eigen_projection_matrix_right.template block<3, 1>(0, 3) //
          = rig_.camera_right_.translation_;
      eigen_projection_matrix_right
          = rig_.camera_right_.intrinsic_ * eigen_projection_matrix_right;

      cv::eigen2cv(eigen_projection_matrix_left, cv_projection_matrix_left);
      cv::eigen2cv(eigen_projection_matrix_right, cv_projection_matrix_right);

      std::cerr << "左目相机投影矩阵 =\n"
                << cv_projection_matrix_left << "\n"
                << "右目相机投影矩阵 =\n"
                << cv_projection_matrix_right << "\n\n";
    }

    for (value_type time = 0.0; time < 50.0; time += 0.1)
    {
      std::cerr << "[INFO] 时间 = (" << std::fixed << std::setprecision(1)
                << time << ").\n";

      // 直接拿到可见顶点的全局索引，而不是三维坐标系本身了
      auto &&[visible_object_indices, image_points_left, image_points_right]
          = path_.GetImage(rig_, time, room_);
      std::cerr << "\t当前场景中，双目可见路标点有 "
                << visible_object_indices.size() << " 个.\n";

      // 利用像素点进行三角化
      {
        cv::Mat cv_image_points_left(
            2, static_cast<int>(image_points_left.size()), CV_32F);
        cv::Mat cv_image_points_right(
            2, static_cast<int>(image_points_right.size()), CV_32F);

        std::cerr << "cv_image_points_left = { .shape = ("
                  << cv_image_points_left.rows << ", "
                  << cv_image_points_left.cols << "), "
                  << ".type = " << cv::typeToString(cv_image_points_left.type())
                  << " }\n"
                  << "cv_image_points_right = { .shape = ("
                  << cv_image_points_right.rows << ", "
                  << cv_image_points_right.cols << "), "
                  << ".type = "
                  << cv::typeToString(cv_image_points_right.type()) << " }\n"
                  << "\n";

        for (size_t i = 0; i < image_points_left.size(); ++i)
        {
          const auto &image_point{image_points_left[i]};
          const value_type x{image_point.x()};
          const value_type y{image_point.y()};
          cv_image_points_left.at<float>(0, i) = x;
          cv_image_points_left.at<float>(1, i) = y;
        }

        for (size_t i = 0; i < image_points_right.size(); ++i)
        {
          const auto &image_point{image_points_right[i]};
          const value_type x{image_point.x()};
          const value_type y{image_point.y()};
          cv_image_points_right.at<float>(0, i) = x;
          cv_image_points_right.at<float>(1, i) = y;
        }

        std::cerr << "[INFO] 正在执行三角化 ...\n";

        std::cerr << "cv_projection_matrix_left = { .shape = ("
                  << cv_projection_matrix_left.rows << ", "
                  << cv_projection_matrix_left.cols << "), "
                  << ".type = "
                  << cv::typeToString(cv_projection_matrix_left.type())
                  << " }\n"
                  << "cv_projection_matrix_right = { .shape = ("
                  << cv_projection_matrix_right.rows << ", "
                  << cv_projection_matrix_right.cols << "), "
                  << ".type = "
                  << cv::typeToString(cv_projection_matrix_right.type())
                  << " }\n"
                  << "cv_image_points_left = { .shape = ("
                  << cv_image_points_left.rows << ", "
                  << cv_image_points_left.cols << "), "
                  << ".type = " << cv::typeToString(cv_image_points_left.type())
                  << " }\n"
                  << "cv_image_points_right = { .shape = ("
                  << cv_image_points_right.rows << ", "
                  << cv_image_points_right.cols << "), "
                  << ".type = "
                  << cv::typeToString(cv_image_points_right.type()) << " }\n"
                  << "\n";

        // 世界坐标系 (即以左目光心为原点的坐标系) 中路标点的齐次坐标
        cv::Mat landmarks_homo;
        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        cv::triangulatePoints(cv_projection_matrix_left,
                              cv_projection_matrix_right, cv_image_points_left,
                              cv_image_points_right, landmarks_homo);

        std::cerr << "[INFO] 三角化执行完毕 ...\n";

        if (visible_object_indices.size()
            != static_cast<size_t>(landmarks_homo.cols))
        {
          std::stringstream ss;
          ss << "Assertion Error: 三角化得到的路标点个数 ("
             << landmarks_homo.cols << ") 与预期不符!";
          throw std::runtime_error{ss.str()};
        }

        std::cerr << "[INFO] 正在将齐次坐标转为非齐次坐标 ...\n";

        // 世界坐标系中路标点的非齐次坐标
        cv::Mat landmarks_nonhomo;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gac42edda3a3a0f717979589fcd6ac0035
        cv::convertPointsFromHomogeneous(landmarks_homo.t(), landmarks_nonhomo);

        std::cerr << "[INFO] 坐标转换完毕 ...\n";

        std::cerr << "landmarks_nonhomo = { .shape = ("
                  << landmarks_nonhomo.rows << ", " << landmarks_nonhomo.cols
                  << "), "
                  << ".type = " << cv::typeToString(landmarks_nonhomo.type())
                  << " }\n"
                  << "\n";

        float error_sum{0.0f};
        float error_min{std::numeric_limits<float>::max()};
        float error_max{std::numeric_limits<float>::lowest()};
        for (size_t i = 0; i < visible_object_indices.size(); ++i)
        {
          const size_t object_point_index{visible_object_indices[i]};
          const Eigen::Vector<value_type, 3> object_point{
              room_.object_matrix_.col(object_point_index),
          };
          const cv::Point3f landmark{landmarks_nonhomo.at<cv::Point3f>(i, 0)};
          // 计算 L1 误差
          const float error{std::abs(object_point.x() - landmark.x)
                            + std::abs(object_point.y() - landmark.y)
                            + std::abs(object_point.z() - landmark.z)};
          error_sum += error;
          error_min = std::min(error_min, error);
          error_max = std::max(error_max, error);
        }

        std::cerr //
            << "\tAverage Error: " << (error_sum / landmarks_nonhomo.rows)
            << "\n"
            << "\tMinimal Error: " << error_min << "\n"
            << "\tMaximal Error: " << error_max << "\n";
      }

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
