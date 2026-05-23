#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include "MeshPlot.hpp"
#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"

using namespace std::chrono_literals;

#define START_VISUALIZATION 0
#define PUBLISH_GT_PATH 1
#define PUBLISH_EST_PATH 1
#define PUBLISH_IMAGE 0
#define OUTPUT_AS_INNOV 1
#define OUTPUT_AS_EUROC 0

#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "euroc_vio/main.h"
#endif

#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
struct PathPublisher : public rclcpp::Node
{
private:
#if (PUBLISH_GT_PATH)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_truth_{
      create_publisher<nav_msgs::msg::Path>("/path_truth", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_truth_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_truth_{
      create_publisher<nav_msgs::msg::Path>("/pose_truth", rclcpp::QoS{10}),
  };
#endif
#if (PUBLISH_EST_PATH)
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_est_{
      create_publisher<nav_msgs::msg::Path>("/path_est", rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_est_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_pose_est_{
      create_publisher<nav_msgs::msg::Path>("/pose_est", rclcpp::QoS{10}),
  };
#endif
#if (PUBLISH_IMAGE)
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image0_{
      create_publisher<sensor_msgs::msg::Image>("/cam0/image_raw",
                                                rclcpp::QoS{10}),
  };
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image1_{
      create_publisher<sensor_msgs::msg::Image>("/cam1/image_raw",
                                                rclcpp::QoS{10}),
  };
#endif

protected:
  PathPublisher() : Node("StereoSlam")
  {
#if (PUBLISH_GT_PATH)
    msg_path_truth_.header.frame_id = DEFAULT_FRAME_ID;
#endif
#if (PUBLISH_EST_PATH)
    msg_path_est_.header.frame_id = DEFAULT_FRAME_ID;
#endif
  }

  template <typename value_type>
  static geometry_msgs::msg::PoseStamped
  CreatePose(const rclcpp::Time &time,
             const Eigen::Quaternion<value_type> &attitude,
             const Eigen::Vector<value_type, 3> &position)
  {
    geometry_msgs::msg::PoseStamped msg_pose;
    msg_pose.header.stamp = time;

    msg_pose.pose.position.x = position.x();
    msg_pose.pose.position.y = position.y();
    msg_pose.pose.position.z = position.z();

    msg_pose.pose.orientation.w = attitude.w();
    msg_pose.pose.orientation.x = attitude.x();
    msg_pose.pose.orientation.y = attitude.y();
    msg_pose.pose.orientation.z = attitude.z();

    return msg_pose;
  }

  template <typename value_type>
  geometry_msgs::msg::PoseStamped
  PublishPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &publisher,
              nav_msgs::msg::Path &path,
              const Eigen::Quaternion<value_type> &attitude,
              const Eigen::Vector<value_type, 3> &position)
  {
    rclcpp::Time now{this->get_clock()->now()};
    path.header.stamp = now;
    const geometry_msgs::msg::PoseStamped pose{
        CreatePose(now, attitude, position),
    };
    path.poses.push_back(pose);
    publisher->publish(path);
    return pose;
  }

#if (PUBLISH_GT_PATH)
  template <typename value_type>
  void PublishGroundTruthPath(const Eigen::Quaternion<value_type> &attitude,
                              const Eigen::Vector<value_type, 3> &position)
  {
    const geometry_msgs::msg::PoseStamped pose{
        PublishPath(publisher_path_truth_,
                    msg_path_truth_, //
                    attitude, position),
    };
    nav_msgs::msg::Path path_pose;
    path_pose.header.frame_id = DEFAULT_FRAME_ID;
    path_pose.header.stamp    = pose.header.stamp;
    path_pose.poses.push_back(pose);
    publisher_pose_truth_->publish(path_pose);
  }
#endif

#if (PUBLISH_EST_PATH)
  template <typename value_type>
  void PublishEstimatedPath(const Eigen::Quaternion<value_type> &attitude,
                            const Eigen::Vector<value_type, 3> &position)
  {
    const geometry_msgs::msg::PoseStamped pose{
        PublishPath(publisher_path_est_,
                    msg_path_est_, //
                    attitude, position),
    };
    nav_msgs::msg::Path path_pose;
    path_pose.header.frame_id = DEFAULT_FRAME_ID;
    path_pose.header.stamp    = pose.header.stamp;
    path_pose.poses.push_back(pose);
    publisher_pose_est_->publish(path_pose);
  }
#endif

#if (PUBLISH_IMAGE)
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
    publisher_image1_->publish(*cv_image.toImageMsg());
  }
#endif
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
#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
  : public PathPublisher
#endif
{
  // 传入长宽高的划分段数
  Room<value_type> room_{};
  // 初始化专属绘制器
  MeshPlot<value_type> mesh_plot_{
      /* create_named_window */
      static_cast<bool>(START_VISUALIZATION),
  };
  // 仿真双目相机
  StereoRig<value_type> rig_{};
  // 仿真双目相机运动路径
  using OrientationMode = Path<value_type>::OrientationMode;
  OrientationMode orientation_mode_{OrientationMode::LookAtCenter};
  Path<value_type> path_{};
  value_type time_limit_simulation_{
      // 计算匀速圆周运动恰好旋转一周所需的时间
      std::round(1 + 2 * std::numbers::pi_v<value_type> / path_.omega_),
  };
  // 时间步长 (单位: 秒)
  value_type step_{static_cast<value_type>(0.05)};

  using Point3     = Eigen::Vector<value_type, 3>;
  using Point2     = Eigen::Vector<value_type, 2>;
  using Attitude   = Eigen::Matrix<value_type, 3, 3>;
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

  void WriteCameraConfig(const std::filesystem::path &path_cam,
                         const Camera<value_type> &camera) const
  {
    std::ofstream fout_cam(path_cam / "sensor.yaml");
    std::print(
        fout_cam,
        "sensor_type: camera\n\n"
        "T_BS:\n"
        "  cols: 4\n"
        "  rows: 4\n"
        "  data: [{:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
        "         {:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
        "         {:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
        "         0.0, 0.0, 0.0, 1.0]\n\n"
        "rate_hz: {}\n"
        "resolution: [{}, {}]\n"
        "camera_model: pinhole\n"
        "intrinsics: [{:.3f}, {:.3f}, {:.3f}, {:.3f}] # fu, fv, cu, cv\n"
        "distortion_model: radial-tangential\n"
        "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n",
        // 空间变换的齐次矩阵形式
        camera.rotation_(0, 0), camera.rotation_(0, 1), camera.rotation_(0, 2),
        camera.translation_(0), camera.rotation_(1, 0), camera.rotation_(1, 1),
        camera.rotation_(1, 2), camera.translation_(1), camera.rotation_(2, 0),
        camera.rotation_(2, 1), camera.rotation_(2, 2), camera.translation_(2),
        // 采样频率
        static_cast<value_type>(1.0) / step_,
        // 分辨率
        camera.width_, camera.height_,
        // 相机内参
        camera.intrinsic_(0, 0), camera.intrinsic_(1, 1),
        camera.intrinsic_(0, 2), camera.intrinsic_(1, 2));
  }

  void
  WriteGroundTruthConfig(const std::filesystem::path &path_groundtruth) const
  {
    std::ofstream fout(path_groundtruth / "sensor.yaml");
    std::print(fout, "sensor_type: visual-inertial\n\n"
                     "sensor_type: camera\n\n"
                     "T_BS:\n"
                     "  cols: 4\n"
                     "  rows: 4\n"
                     "  data: [1.0, 0.0, 0.0, 0.0,\n"
                     "         0.0, 1.0, 0.0, 0.0,\n"
                     "         0.0, 0.0, 1.0, 0.0,\n"
                     "         0.0, 0.0, 0.0, 1.0]\n");
  }

  std::pair<Point3, Attitude> GetPose(value_type time) const
  {
    return path_.GetPose(room_, time, orientation_mode_);
  }

  void Start()
  {
    const size_t min_count_common_landmarks{10};

    // 投影矩阵 P = K [R, t]
    cv::Mat cv_projection_matrix_left;
    cv::Mat cv_projection_matrix_right;

    // 全局状态
    Quaternion estimated_current_attitude{
        Quaternion::Identity(),
    };
    Point3 estimated_current_position{
        Point3::Zero(),
    };
    // 位姿初始化
    std::tie(estimated_current_position, estimated_current_attitude)
        = GetPose(static_cast<value_type>(0.0));
    path_.print_debug_info_ = false;
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
      // std::print("左目相机投影矩阵 =\n{}\n右目相机投影矩阵 =\n{}\n",
      //            cv_projection_matrix_left, cv_projection_matrix_right);
    }

#if (OUTPUT_AS_INNOV)
    // 创建文件结构
    std::error_code ec;
    std::filesystem::path path_fake{"fake"};
    std::filesystem::remove_all(path_fake, ec);
    if (std::filesystem::create_directories(path_fake, ec))
    {
      std::print(stderr, "[INFO] 工作目录创建成功: {}\n",
                 std::filesystem::absolute(path_fake).string());
    }
    else
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    std::ofstream fout_fake_data_in_camera_frame_csv(path_fake
                                                     / "data_camera.csv");
    std::print(fout_fake_data_in_camera_frame_csv,
               "#timestamp [ns], "
               "r_x [rad], r_y [rad], r_z [rad], "
               "t_x, t_y, t_z\n");
    std::ofstream fout_fake_data_in_world_frame_csv(path_fake
                                                    / "data_world.csv");
    std::print(fout_fake_data_in_world_frame_csv,
               "#timestamp [ns], "
               "r_x [rad], r_y [rad], r_z [rad], "
               "t_x, t_y, t_z\n");
#endif
#if (OUTPUT_AS_EUROC)
    // 创建 mav0 目录
    std::error_code ec;
    std::filesystem::path path_mav0{"mav0"};
    std::filesystem::remove_all(path_mav0, ec);
    if (std::filesystem::create_directories(path_mav0, ec))
    {
      std::print(stderr, "[INFO] 工作目录创建成功: {}\n",
                 std::filesystem::absolute(path_mav0).string());
    }
    else
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0 目录
    std::filesystem::path path_cam0{path_mav0 / "cam0"};
    if (!std::filesystem::create_directories(path_cam0, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0/data 目录
    std::filesystem::path path_cam0_data{path_mav0 / "cam0" / "data"};
    if (!std::filesystem::create_directories(path_cam0_data, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1 目录
    std::filesystem::path path_cam1{path_mav0 / "cam1"};
    if (!std::filesystem::create_directories(path_cam1, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1/data 目录
    std::filesystem::path path_cam1_data{path_mav0 / "cam1" / "data"};
    if (!std::filesystem::create_directories(path_cam1_data, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/state_groundtruth_estimate0 目录
    std::filesystem::path path_groundtruth{path_mav0
                                           / "state_groundtruth_estimate0"};
    if (!std::filesystem::create_directories(path_groundtruth, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }

    // 输出 mav0/cam0/sensor.yaml
    WriteCameraConfig(path_cam0, rig_.camera_left_);
    // 输出 mav0/cam1/sensor.yaml
    WriteCameraConfig(path_cam1, rig_.camera_right_);
    // 输出 mav0/state_groundtruth_estimate0/sensor.yaml
    WriteGroundTruthConfig(path_groundtruth);

    // 输出 mav0/cam0/data.csv 表头
    std::ofstream fout_cam0_data_csv(path_cam0 / "data.csv");
    std::print(fout_cam0_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/cam1/data.csv 表头
    std::ofstream fout_cam1_data_csv(path_cam1 / "data.csv");
    std::print(fout_cam1_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/state_groundtruth_estimate0/data.csv 表头
    std::ofstream fout_groundtruth_csv(path_groundtruth / "data.csv");
    std::print(fout_groundtruth_csv,
               "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
               "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n");
    std::print(fout_groundtruth_csv,
               // 时间戳
               "{:020d}, "
               // 位置
               "{:.18f}, {:.18f}, {:.18f}, "
               // 朝向
               "{:.18f}, {:.18f}, {:.18f}, {:.18f}\n",
               0,                                        //
               position.x(), position.y(), position.z(), //
               attitude.w(), attitude.x(), attitude.y(), attitude.z());
#endif

    // 内参矩阵 K
    const auto cv_camera_matrix{
        cv_projection_matrix_left(cv::Range(0, 3), cv::Range(0, 3))};

    bool first_loop{true};
    Frame prev_frame;

    for (value_type time = 0.0; time < time_limit_simulation_; time += step_)
    {
#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
      // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
      if (!rclcpp::ok())
      {
        break;
      }
#endif

      std::print(stderr, "[INFO] 时间 = ({:.1f}).\n", time);

#if (OUTPUT_AS_EUROC || OUTPUT_AS_INNOV)
      const auto timestamp_ns{static_cast<std::int64_t>(time * 1e9)};
#endif
#if (OUTPUT_AS_EUROC)
      std::print(fout_cam0_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
      std::print(fout_cam1_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
#endif

      // 直接拿到可见顶点的全局索引，而不是三维坐标系本身了
      const Frame frame{path_.GetImage(rig_, time, room_, orientation_mode_)};
      const size_t count_common_landmarks{std::get<0>(frame).size()};

      // std::print(stderr, "\t当前场景中，双目可见路标点有 {} 个.\n",
      //            count_common_landmarks);

      // 利用像素点进行三角化
      if (first_loop)
      {
        first_loop = false;
        std::print("\t初始化 ...\n");
      }
      else if (count_common_landmarks >= min_count_common_landmarks)
      {
        // 必须复制一份
        // 如果不提前复制一份，就直接把 frame 用于对齐
        // 那么在执行 `prev_frame = std::move(frame)` 以后
        // prev_frame 中可用的像素点个数一定会逐渐减少
        // 直到像素点个数少于 4 个时触发 cv::solvePnPRansac 函数报错
        Frame current_frame{frame};
        // 先将前后相邻两个图像帧中的像素点对齐，只保留交集部分
        StereoRig<value_type>::AlignFrames(prev_frame, current_frame);
        const std::vector<size_t> &visible_object_indices{
            std::get<0>(current_frame),
        };
        const std::vector<Point2> &image_points_left{
            std::get<1>(current_frame),
        };

        if (std::get<0>(prev_frame).size() != visible_object_indices.size())
        {
          std::stringstream ss;
          ss << "[ERROR] 前一帧路标点个数 (" << std::get<0>(prev_frame).size()
             << ") 与后一帧 (" << visible_object_indices.size() << ") 不符!\n";
          throw std::runtime_error{ss.str()};
        }

        cv::Mat prev_cv_image_points_left{
            VisualSim::eigen2cv(std::get<1>(prev_frame)),
        };
        cv::Mat prev_cv_image_points_right{
            VisualSim::eigen2cv(std::get<2>(prev_frame)),
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

        if (visible_object_indices.size()
            != static_cast<size_t>(landmarks_homo.cols))
        {
          std::stringstream ss;
          ss << "三角化得到的路标点个数 (" << landmarks_homo.cols
             << ") 与预期 (" << visible_object_indices.size() << ") 不符!\n";
          throw std::runtime_error{ss.str()};
        }

        // 世界坐标系中路标点的非齐次坐标
        cv::Mat landmarks_nonhomo;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gac42edda3a3a0f717979589fcd6ac0035
        cv::convertPointsFromHomogeneous(landmarks_homo.t(), landmarks_nonhomo);

        landmarks_nonhomo = landmarks_nonhomo.t();

        PrintInfo(landmarks_nonhomo);

        // float error_sum{0.0f};
        // float error_min{std::numeric_limits<float>::max()};
        // float error_max{std::numeric_limits<float>::lowest()};
        // for (size_t i = 0; i < visible_object_indices.size(); ++i)
        // {
        //   const size_t object_point_index{visible_object_indices[i]};
        //   // Eigen::Vector<value_type, 3>
        //   const auto &object_point{
        //       room_.object_matrix_.col(object_point_index),
        //   };
        //   const cv::Point3f landmark{landmarks_nonhomo.at<cv::Point3f>(0, i)};
        //   // 计算 L1 误差
        //   const float error{std::abs(object_point.x() - landmark.x)
        //                     + std::abs(object_point.y() - landmark.y)
        //                     + std::abs(object_point.z() - landmark.z)};
        //   error_sum += error;
        //   error_min = std::min(error_min, error);
        //   error_max = std::max(error_max, error);
        // }

        // std::print(
        //     stderr,
        //     "\t===== 路标点估计误差 =====\n"
        //     "\tAverage Error: {}\n"
        //     "\tMinimal Error: {}\n"
        //     "\tMaximal Error: {}\n",
        //     (error_sum / static_cast<float>(visible_object_indices.size())),
        //     error_min, error_max);

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

        // float reproj_error_sum{0.0f};
        // float reproj_error_min{std::numeric_limits<float>::max()};
        // float reproj_error_max{std::numeric_limits<float>::lowest()};
        // for (size_t i = 0; i < image_points_left.size(); ++i)
        // {
        //   const Point2 &image_point{image_points_left[i]};
        //   const cv::Point2f cv_image{cv_reproj_left.at<cv::Point2f>(i, 0)};
        //   // 计算 L1 误差
        //   const float error{std::abs(image_point.x() - cv_image.x)
        //                     + std::abs(image_point.y() - cv_image.y)};
        //   reproj_error_sum += error;
        //   reproj_error_min = std::min(reproj_error_min, error);
        //   reproj_error_max = std::max(reproj_error_max, error);
        // }

        // std::print(
        //     stderr,
        //     "\t===== 重投影误差 =====\n"
        //     "\tAverage Error: {}\n"
        //     "\tMinimal Error: {}\n"
        //     "\tMaximal Error: {}\n",
        //     (reproj_error_sum / static_cast<float>(image_points_left.size())),
        //     reproj_error_min, reproj_error_max);

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
          estimated_current_attitude
              = (estimated_current_attitude * delta_rotation.conjugate())
                    .normalized();
          estimated_current_position
              = estimated_current_position
                - estimated_current_attitude * delta_position;
        }

        Point3 true_current_position{Point3::Zero()};
        Quaternion true_current_attitude{Quaternion::Identity()};
        std::tie(true_current_position, true_current_attitude) = GetPose(time);

#if (OUTPUT_AS_INNOV)
        // 计算真值的旋转角度、单位化平移向量
        Point3 true_prev_position{Point3::Zero()};
        Quaternion true_prev_attitude{Quaternion::Identity()};
        std::tie(true_prev_position, true_prev_attitude)
            = GetPose(time - step_);
        Point3 true_delta_position{
            true_current_position - true_prev_position,
        };
        Quaternion true_delta_attitude{
            // C_21 = F_2 F_1.transpose
            true_current_attitude * true_prev_attitude.conjugate(),
        };
        Eigen::AngleAxis<value_type> true_rotation_angle_axis{
            true_delta_attitude,
        };
        Point3 true_rotation_vector{
            true_rotation_angle_axis.angle() * true_rotation_angle_axis.axis(),
        };
        // 写入数据文件
        std::print(fout_fake_data_in_world_frame_csv,
                   // 时间戳
                   "{:020d}, "
                   // 旋转向量
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 平移方向
                   "{:.18f}, {:.18f}, {:.18f}\n",
                   timestamp_ns, //
                   true_rotation_vector.x(), true_rotation_vector.y(),
                   true_rotation_vector.z(), //
                   true_delta_position.x(), true_delta_position.y(),
                   true_delta_position.z());
        // 转换坐标系：从世界坐标系转为相机坐标系
        true_delta_position
            = true_prev_attitude.conjugate() * true_delta_position;
        true_rotation_vector
            = true_prev_attitude.conjugate() * true_rotation_vector;
        // 写入数据文件
        std::print(fout_fake_data_in_camera_frame_csv,
                   // 时间戳
                   "{:020d}, "
                   // 旋转向量
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 平移方向
                   "{:.18f}, {:.18f}, {:.18f}\n",
                   timestamp_ns, //
                   true_rotation_vector.x(), true_rotation_vector.y(),
                   true_rotation_vector.z(), //
                   true_delta_position.x(), true_delta_position.y(),
                   true_delta_position.z());
#endif
#if (OUTPUT_AS_EUROC)
        std::print(fout_groundtruth_csv,
                   // 时间戳
                   "{:020d}, "
                   // 位置
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 朝向
                   "{:.18f}, {:.18f}, {:.18f}, {:.18f}\n",
                   timestamp_ns,                                            //
                   true_position.x(), true_position.y(), true_position.z(), //
                   true_attitude.w(), true_attitude.x(), true_attitude.y(),
                   true_attitude.z());
#endif

        // std::print(stderr,
        //            "\t===== 位姿估计误差 =====\n"
        //            "\t真实位置: [{:.3f}, {:.3f}, {:.3f}];\n"
        //            "\t真实朝向: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]\n"
        //            "\t估计位置: [{:.3f}, {:.3f}, {:.3f}];\n"
        //            "\t估计朝向: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]\n"
        //            "\t位置误差: [{:.3f}, {:.3f}, {:.3f}];\n"
        //            "\t朝向误差: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]\n",
        //            true_position.x(), true_position.y(), true_position.z(),
        //            true_attitude.w(), true_attitude.x(), true_attitude.y(),
        //            true_attitude.z(), position.x(), position.y(), position.z(),
        //            attitude.w(), attitude.x(), attitude.y(), attitude.z(),
        //            std::abs(true_position.x() - position.x()),
        //            std::abs(true_position.y() - position.y()),
        //            std::abs(true_position.z() - position.z()),
        //            std::abs(true_attitude.w() - attitude.w()),
        //            std::abs(true_attitude.x() - attitude.x()),
        //            std::abs(true_attitude.y() - attitude.y()),
        //            std::abs(true_attitude.z() - attitude.z()));
      }

#if (START_VISUALIZATION || PUBLISH_IMAGE || OUTPUT_AS_EUROC)
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

        // 核心绘制逻辑收口
        mesh_plot_.Draw(cv_image_left, cv_image_right, frame);

#if (PUBLISH_IMAGE)
        PublishImage(cv_image_left, cv_image_right);
#endif
#if (START_VISUALIZATION)
        if (mesh_plot_.Render(cv_image_left, cv_image_right, 1000))
        {
          break;
        }
#endif
#if (OUTPUT_AS_EUROC)
        // 时间戳作为图片文件名称
        const std::string image_file_name{
            std::format("{:020d}.png", timestamp_ns),
        };
        cv::imwrite(std::filesystem::absolute(path_cam0_data / image_file_name),
                    cv_image_left);
        cv::imwrite(std::filesystem::absolute(path_cam1_data / image_file_name),
                    cv_image_right);
#endif
      }
#endif

#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
      // std::print(stderr, "[INFO] 尝试发布位姿 ...\n");
#if (PUBLISH_GT_PATH)
      {
        Point3 true_position{Point3::Zero()};
        Quaternion true_attitude{Quaternion::Identity()};
        std::tie(true_position, true_attitude) = GetPose(time);
        PublishGroundTruthPath(true_attitude, true_position);
      }
#endif
#if (PUBLISH_EST_PATH)
      PublishEstimatedPath(attitude, position);
#endif
#endif

      // 双目可见路标点数量过少!
      if (count_common_landmarks < min_count_common_landmarks)
      {
        first_loop = true;
      }
      else
      {
        prev_frame = std::move(frame);
      }

      std::this_thread::sleep_for(50ms);
    }

    if (first_loop)
    {
      throw std::runtime_error{"直到仿真结束，双目可见路标点数量还是太少!"};
    }
  }
};

int main(int argc, char *argv[])
{
#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
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
    std::println(stderr, "{}", ex.what());
  }

#if ((PUBLISH_GT_PATH || PUBLISH_EST_PATH || PUBLISH_IMAGE)                    \
     && !(OUTPUT_AS_EUROC || OUTPUT_AS_INNOV))
  // 关闭 ROS 2 实例
  rclcpp::shutdown();
#endif
  return 0;
}
