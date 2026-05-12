#include <cstddef>
#include <exception>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <print>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include "MeshPlot.hpp"
#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"

#define START_VISUALIZATION 0

#if (!START_VISUALIZATION)
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "euroc_vio/main.h"
#endif

#if (!START_VISUALIZATION)
struct PathPublisher : public rclcpp::Node
{
private:
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_{
      create_publisher<nav_msgs::msg::Path>("/path_stereo_slam",
                                            rclcpp::QoS{10}),
  };
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image0_{
      create_publisher<sensor_msgs::msg::Image>("/cam0/image_raw",
                                                rclcpp::QoS{10}),
  };
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image1_{
      create_publisher<sensor_msgs::msg::Image>("/cam1/image_raw",
                                                rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_;

protected:
  PathPublisher() : Node("StereoSlam")
  {
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
  }

  template <typename value_type>
  void PublishPath(const Eigen::Quaternion<value_type> &attitude,
                   const Eigen::Vector<value_type, 3> &position)
  {
    rclcpp::Time now{this->get_clock()->now()};

    msg_path_.header.stamp = now;

    geometry_msgs::msg::PoseStamped msg_pose;
    msg_pose.header.stamp = now;

    msg_pose.pose.position.x = position.x();
    msg_pose.pose.position.y = position.y();
    msg_pose.pose.position.z = position.z();

    msg_pose.pose.orientation.w = attitude.w();
    msg_pose.pose.orientation.x = attitude.x();
    msg_pose.pose.orientation.y = attitude.y();
    msg_pose.pose.orientation.z = attitude.z();

    msg_path_.poses.push_back(msg_pose);
    publisher_path_->publish(msg_path_);
  }

  void PublishImage(const cv::Mat &image_left, const cv::Mat &image_right)
  {
    rclcpp::Time now{this->get_clock()->now()};

    cv_bridge::CvImage cv_image;
    cv_image.header.stamp    = now;
    cv_image.header.frame_id = DEFAULT_FRAME_ID;
    cv_image.encoding        = sensor_msgs::image_encodings::BGR8;

    cv_image.image = image_left;
    publisher_image0_->publish(*cv_image.toImageMsg());
    cv_image.image = image_right;
    publisher_image0_->publish(*cv_image.toImageMsg());
  }
};
#endif

// static void PrintCvMatInfo(const std::string &mat_name, const cv::Mat &mat)
// {
//   const auto mat_type{mat.type()};
//   const auto mat_depth{mat.depth()};
//   std::print("\t{} = {{ "
//              "data = {}, "
//              "total = {}, "
//              "dim = {}, "
//              "shape = ({}, {}), "
//              "continuous = {}, "
//              "channel = {}, "
//              "depth = {} ({}), "
//              "type = {} ({}) }}\n",
//              mat_name,                    //
//              static_cast<bool>(mat.data), //
//              mat.total(),                 //
//              mat.dims,                    //
//              mat.rows,
//              mat.cols,                               //
//              mat.isContinuous(),                     //
//              mat.channels(),                         //
//              cv::typeToString(mat_depth), mat_depth, //
//              cv::typeToString(mat_type), mat_type);
// }
// #define PrintInfo(mat) PrintCvMatInfo(#mat, mat)
#define PrintInfo(mat) /* do nothing */

template <typename value_type>
struct VisualSim
#if (!START_VISUALIZATION)
  : public PathPublisher
#endif
{
  // 传入长宽高的划分段数
  Room<value_type> room_{10, 10, 6};
  // 初始化专属绘制器
  MeshPlot<value_type> mesh_plot_{
      /* create_named_window */
      static_cast<bool>(START_VISUALIZATION),
  };
  // 仿真双目相机
  StereoRig<value_type> rig_{};
  // 仿真双目相机运动路径
  Path<value_type> path_{};

  using Point3     = Eigen::Vector<value_type, 3>;
  using Point2     = Eigen::Vector<value_type, 2>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Frame      = typename StereoRig<value_type>::Frame;

  VisualSim() : mesh_plot_{room_}
  {
    // 只修改双目相机的基线长度
    rig_.camera_right_.translation_ = {-0.1, 0.0, 0.0};
  }

  static cv::Mat eigen2cv(const std::vector<Point2> &image_points)
  {
    cv::Mat cv_image_points(2, static_cast<int>(image_points.size()), CV_32F);

    for (size_t i = 0; i < image_points.size(); ++i)
    {
      const auto &image_point{image_points[i]};
      const value_type x{image_point.x()};
      const value_type y{image_point.y()};
      cv_image_points.at<float>(0, i) = x;
      cv_image_points.at<float>(1, i) = y;
    }

    return cv_image_points;
  }

  void Start()
  {
    // 投影矩阵 P = K [R, t]
    cv::Mat cv_projection_matrix_left;
    cv::Mat cv_projection_matrix_right;

    // 全局状态
    Quaternion attitude{
        Quaternion::Identity(),
    };
    Point3 position{
        Point3::Zero(),
    };
    // 位姿初始化
    std::tie(position, attitude) = path_.GetPose(room_, 0.0);
    // 局部增量
    Quaternion delta_rotation{
        Quaternion::Identity(),
    };
    Point3 delta_position{
        Point3::Zero(),
    };

    // 将 Eigen 矩阵转换为 OpenCV 矩阵
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

    // 内参矩阵 K
    const auto cv_camera_matrix{
        cv_projection_matrix_left(cv::Range(0, 3), cv::Range(0, 3))};

    bool first_loop{true};
    Frame prev_frame;

    for (value_type time = 0.0; time < 200.0; time += 0.1)
    {
#if (!START_VISUALIZATION)
      // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
      if (!rclcpp::ok())
      {
        break;
      }
#endif

      std::cerr << "[INFO] 时间 = (" << std::fixed << std::setprecision(1)
                << time << ").\n";

      // 直接拿到可见顶点的全局索引，而不是三维坐标系本身了
      Frame frame{path_.GetImage(rig_, time, room_)};

      std::cerr << "\t当前场景中，双目可见路标点有 "
                << std::get<0>(frame).size() << " 个.\n";

      // 利用像素点进行三角化
      if (first_loop)
      {
        first_loop = false;
        std::print("\t初始化 ...\n");
#if (!START_VISUALIZATION)
        PublishPath(attitude, position);
#endif
      }
      else
      {
        // 必须复制一份
        // 如果不提前复制一份，就直接把 frame 用于对齐
        // 那么在执行 `prev_frame = std::move(frame)` 以后
        // prev_frame 中可用的像素点个数一定会逐渐减少
        // 直到像素点个数少于 4 个时触发 cv::solvePnPRansac 函数报错
        Frame current_frame{std::as_const(frame)};
        // 先将前后相邻两个图像帧中的像素点对齐，只保留交集部分
        StereoRig<value_type>::AlignFrames(prev_frame, current_frame);
        const std::vector<size_t> &visible_object_indices{
            std::get<0>(current_frame),
        };
        const std::vector<Point2> &image_points_left{
            std::get<1>(current_frame),
        };

        if (std::get<0>(prev_frame).size() != std::get<0>(current_frame).size())
        {
          std::stringstream ss;
          ss << "[ERROR] 前一帧路标点个数 (" << std::get<0>(prev_frame).size()
             << ") 与后一帧 (" << std::get<0>(current_frame).size()
             << ") 不符!\n";
          throw std::runtime_error{ss.str()};
        }

        // std::cerr << "\t正在执行三角化 ...\n";

        cv::Mat prev_cv_image_points_left{
            VisualSim::eigen2cv(std::get<1>(prev_frame)),
        };
        cv::Mat prev_cv_image_points_right{
            VisualSim::eigen2cv(std::get<1>(prev_frame)),
        };

        PrintInfo(cv_projection_matrix_left);
        PrintInfo(cv_projection_matrix_right);
        PrintInfo(prev_cv_image_points_left);
        PrintInfo(prev_cv_image_points_right);

        // 世界坐标系 (即以左目光心为原点的坐标系) 中路标点的齐次坐标
        cv::Mat landmarks_homo;
        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        cv::triangulatePoints(cv_projection_matrix_left,
                              cv_projection_matrix_right,
                              prev_cv_image_points_left,
                              prev_cv_image_points_right, landmarks_homo);

        PrintInfo(landmarks_homo);

        // std::cerr << "\t三角化执行完毕 ...\n";

        if (visible_object_indices.size()
            != static_cast<size_t>(landmarks_homo.cols))
        {
          std::stringstream ss;
          ss << "三角化得到的路标点个数 (" << landmarks_homo.cols
             << ") 与预期 (" << visible_object_indices.size() << ") 不符!\n";
          throw std::runtime_error{ss.str()};
        }

        // std::cerr << "\t正在将齐次坐标转为非齐次坐标 ...\n";

        // 世界坐标系中路标点的非齐次坐标
        cv::Mat landmarks_nonhomo;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gac42edda3a3a0f717979589fcd6ac0035
        cv::convertPointsFromHomogeneous(landmarks_homo.t(), landmarks_nonhomo);

        landmarks_nonhomo = landmarks_nonhomo.t();

        // std::cerr << "\t坐标转换完毕 ...\n";

        PrintInfo(landmarks_nonhomo);

        float error_sum{0.0f};
        float error_min{std::numeric_limits<float>::max()};
        float error_max{std::numeric_limits<float>::lowest()};
        for (size_t i = 0; i < visible_object_indices.size(); ++i)
        {
          const size_t object_point_index{visible_object_indices[i]};
          // Eigen::Vector<value_type, 3>
          const auto &object_point{
              room_.object_matrix_.col(object_point_index),
          };
          const cv::Point3f landmark{landmarks_nonhomo.at<cv::Point3f>(0, i)};
          // 计算 L1 误差
          const float error{std::abs(object_point.x() - landmark.x)
                            + std::abs(object_point.y() - landmark.y)
                            + std::abs(object_point.z() - landmark.z)};
          error_sum += error;
          error_min = std::min(error_min, error);
          error_max = std::max(error_max, error);
        }

        std::cerr //
            << "\t===== 路标点估计误差 =====\n"
            << "\tAverage Error: "
            << (error_sum / static_cast<float>(visible_object_indices.size()))
            << "\n"
            << "\tMinimal Error: " << error_min << "\n"
            << "\tMaximal Error: " << error_max << "\n";

        cv::Mat cv_image_points_left{
            VisualSim::eigen2cv(image_points_left),
        };
        // cv::Mat cv_image_points_right{
        //     VisualSim::eigen2cv(image_points_right),
        // };

        // 张量 cv_image_points_left 只有 1 个通道
        // 而 cv::solvePnPRansac 要求 imagePoints 参数必须是 Nx2 形状
        // 因此必须提前转置
        cv::Mat cv_image_points_next{cv_image_points_left.t()};
        PrintInfo(cv_image_points_next);

        // const int npoints_o{std::max(landmarks_nonhomo.checkVector(3, CV_32F),
        //                              landmarks_nonhomo.checkVector(3, CV_64F))};
        // std::print("\tlen(landmarks_nonhomo) = {}.\n", npoints_o);
        // const int npoints_i{
        //     std::max(cv_image_points_next.checkVector(2, CV_32F),
        //              cv_image_points_next.checkVector(2, CV_64F))};
        // std::print("\tlen(cv_image_points_next) = {}.\n", npoints_i);

        cv::Mat rVec, tVec;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        cv::solvePnPRansac(landmarks_nonhomo, cv_image_points_next,
                           cv_camera_matrix, cv::noArray(), rVec, tVec);

        PrintInfo(rVec);
        PrintInfo(tVec);

        cv::Mat cv_reproj_left;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        cv::projectPoints(landmarks_nonhomo, rVec, tVec, cv_camera_matrix,
                          cv::noArray(), cv_reproj_left);

        PrintInfo(cv_reproj_left);

        static_assert(
            std::is_same_v<typename std::remove_cvref_t<
                               decltype(image_points_left)>::value_type,
                           Point2>,
            "变量 `image_points_left` 类型错误!");
        if (cv_reproj_left.total() * cv_reproj_left.channels()
            != image_points_left.size() * Point2::RowsAtCompileTime)
        {
          throw std::runtime_error{"[ERROR] 重投影点个数与理论投影点个数不符!"};
        }

        float reproj_error_sum{0.0f};
        float reproj_error_min{std::numeric_limits<float>::max()};
        float reproj_error_max{std::numeric_limits<float>::lowest()};
        for (size_t i = 0; i < image_points_left.size(); ++i)
        {
          const Point2 &image_point{image_points_left[i]};
          const cv::Point2f cv_image{cv_reproj_left.at<cv::Point2f>(i, 0)};
          // 计算 L1 误差
          const float error{std::abs(image_point.x() - cv_image.x)
                            + std::abs(image_point.y() - cv_image.y)};
          reproj_error_sum += error;
          reproj_error_min = std::min(reproj_error_min, error);
          reproj_error_max = std::max(reproj_error_max, error);
        }

        std::cerr //
            << "\t===== 重投影误差 =====\n"
            << "\tAverage Error: "
            << (reproj_error_sum / static_cast<float>(image_points_left.size()))
            << "\n"
            << "\tMinimal Error: " << reproj_error_min << "\n"
            << "\tMaximal Error: " << reproj_error_max << "\n";

        // 估计轨迹
        {
          // 类型转换
          Point3 eigen_rVec;
          cv::cv2eigen(rVec, eigen_rVec);
          delta_rotation = Quaternion{Eigen::AngleAxis<value_type>{
              eigen_rVec.norm(),
              eigen_rVec.normalized(),
          }};
          cv::cv2eigen(tVec, delta_position);
          // 状态更新
          position = attitude * delta_position + position;
          attitude = (attitude * delta_rotation).normalized();
#if (!START_VISUALIZATION)
          PublishPath(attitude, position);
#endif
        }

        Point3 true_position{Point3::Zero()};
        Quaternion true_attitude{Quaternion::Identity()};
        std::tie(true_position, true_attitude) = path_.GetPose(room_, time);

        std::cerr //
            << "\t===== 位姿估计误差 =====\n"

            << "\t真实位置: ["             //
            << true_position.x() << ", "   //
            << true_position.y() << ", "   //
            << true_position.z() << "];\n" //

            << "\t真实朝向: ["           //
            << true_attitude.w() << ", " //
            << true_attitude.x() << ", " //
            << true_attitude.y() << ", " //
            << true_attitude.z()         //
            << "]\n"                     //

            << "\t估计位置: ["        //
            << position.x() << ", "   //
            << position.y() << ", "   //
            << position.z() << "];\n" //

            << "\t估计朝向: ["      //
            << attitude.w() << ", " //
            << attitude.x() << ", " //
            << attitude.y() << ", " //
            << attitude.z()         //
            << "]\n"                //

            << "\t位置误差: ["                                      //
            << std::abs(true_position.x() - position.x()) << ", "   //
            << std::abs(true_position.y() - position.y()) << ", "   //
            << std::abs(true_position.z() - position.z()) << "];\n" //

            << "\t朝向误差: ["                                    //
            << std::abs(true_attitude.w() - attitude.w()) << ", " //
            << std::abs(true_attitude.x() - attitude.x()) << ", " //
            << std::abs(true_attitude.y() - attitude.y()) << ", " //
            << std::abs(true_attitude.z() - attitude.z())         //
            << "]\n";
      }

      // 绘制相机图像
      {
        const cv::Scalar background_gray{128, 128, 128};
        cv::Mat cv_image_left(rig_.camera_left_.height_,
                              rig_.camera_left_.width_, CV_8UC3,
                              background_gray);
        cv::Mat cv_image_right(rig_.camera_right_.height_,
                               rig_.camera_right_.width_, CV_8UC3,
                               background_gray);

        std::print("\t绘制双目图像 ...\n");

        const std::vector<size_t> &visible_object_indices{std::get<0>(frame)};
        const std::vector<Point2> &image_points_left{std::get<1>(frame)};
        const std::vector<Point2> &image_points_right{std::get<2>(frame)};
        // 核心绘制逻辑收口
        mesh_plot_.Draw(cv_image_left, cv_image_right, visible_object_indices,
                        image_points_left, image_points_right);

#if (!START_VISUALIZATION)
        PublishImage(cv_image_left, cv_image_right);
#else
        if (mesh_plot_.Render(cv_image_left, cv_image_right, 1000))
        {
          break;
        }
#endif
      }

      prev_frame = std::move(frame);
    }
  }
};

int main(int argc, char *argv[])
{
#if (!START_VISUALIZATION)
  // 初始化 ROS 2
  rclcpp::init(argc, argv);
#else
  (void) argc;
  (void) argv;
#endif

  try
  {
    VisualSim<float>{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::cerr << ex.what() << "\n";
  }

#if (!START_VISUALIZATION)
  // 关闭 ROS 2 实例
  rclcpp::shutdown();
#endif
  return 0;
}
