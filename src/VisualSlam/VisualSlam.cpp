#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

// #include "EKF.hpp"
#include "EightPointAlgorithm.hpp"
#include "EuRoC.hpp"
#include "FastDetector.hpp"
#include "ImageDataLoader.hpp"

#define START_VISUALIZATION 1

#if (!START_VISUALIZATION)
struct StereoSlamPublisher : public rclcpp::Node
{
private:
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_{
      create_publisher<nav_msgs::msg::Path>("/path_stereo_slam",
                                            rclcpp::QoS{10}),
  };
  nav_msgs::msg::Path msg_path_;

protected:
  StereoSlamPublisher() : Node("StereoSlam")
  {
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
  }

  template <typename value_type>
  void Publish(const double timestamp,
               const Eigen::Quaternion<value_type> &attitude,
               const Eigen::Vector<value_type, 3> &position)
  {
    geometry_msgs::msg::PoseStamped msg_pose;

    rclcpp::Time pose_timestamp{
        static_cast<std::int64_t>(timestamp * 1e9),
    };
    msg_path_.header.stamp = pose_timestamp;
    msg_pose.header.stamp  = pose_timestamp;

    msg_pose.pose.position.x = position.x();
    msg_pose.pose.position.y = position.y();
    msg_pose.pose.position.z = position.z();

    msg_pose.pose.orientation.w = attitude.w();
    msg_pose.pose.orientation.x = attitude.x();
    msg_pose.pose.orientation.y = attitude.y();
    msg_pose.pose.orientation.z = attitude.z();

    msg_path_.poses.push_back(msg_pose);
    publisher_->publish(msg_path_);
  }
};
#endif

template <typename PointType = cv::Point2f>
struct StereoSlam
#if (!START_VISUALIZATION)
  : public StereoSlamPublisher
#endif
{
public:
  static constexpr double threshold_for_pose_recovery_{20.0};
  const std::string loopback_window_name_{"VisualSlam"};
  const std::string disparity_window_name_{"Disparity"};
  const std::string depth_window_name_{"Depth Map"};
  const EuRoC::EuRoC euroc_{};

  using value_type = typename PointType::value_type;
  using EPA        = EightPointAlgorithm<value_type>;
  using Landmark   = Eigen::Vector<value_type, 4>;

private:
  std::ofstream data_output_{"StereoSlam.csv", std::ios::out | std::ios::trunc};
  ImageDataLoader loader_{};
  CornerDetection::FastDetector<PointType> detector_{};
  size_t frame_index_{};
  // EKF<PointType> ekf_{};
  const bool visualize_{true};
  const bool plot_disparity_and_depth_{false};

public:
  StereoSlam(bool visualize = true, bool plot_disparity_and_depth = false) :
    visualize_{visualize}, plot_disparity_and_depth_{plot_disparity_and_depth}
  {
    if (visualize_)
    {
      cv::namedWindow(loopback_window_name_, cv::WINDOW_NORMAL);
      if (plot_disparity_and_depth_)
      {
        cv::namedWindow(disparity_window_name_, cv::WINDOW_NORMAL);
        cv::namedWindow(depth_window_name_, cv::WINDOW_NORMAL);
      }
    }
  }

  ~StereoSlam()
  {
    cv::destroyAllWindows();
  }

private:
  enum class KeyEvent
  {
    EXIT,
    NEXT,
    PREV,
    NOOP
  };

  /**
   * @brief 处理键盘事件
   * @param delay 延时 (单位：毫秒)
   */
  KeyEvent InterpretKeyEvent(int delay = 10)
  {
    size_t digit{0};

    while (true)
    {
      // https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#gafa15c0501e0ddd90918f17aa071d3dd0
      const auto key{cv::waitKey(delay) & 0xFF};
      if ('0' <= key && key <= '9')
      {
        digit = 10 * digit + (key - '0');
        continue;
      }

      // 处理回车键
      if (key == 13)
      {
        frame_index_ = loader_.Rewind(digit);
        return KeyEvent::NOOP;
      }
      else if (key == 's' || key == 'S')
      {
        // SaveStereoFrame(frame);
        return KeyEvent::NOOP;
      }
      else if (key == 27 || key == 'q' || key == 'Q')
      {
        return KeyEvent::EXIT;
      }
      else if (key == 'd' || key == 'D')
      {
        return KeyEvent::NEXT;
      }
      else if (key == 'a' || key == 'A')
      {
        return KeyEvent::PREV;
      }

      break;
    }

    return KeyEvent::NOOP;
  }

  static std::vector<cv::Scalar> GenerateRandomColors(int count_colors)
  {
    cv::Mat m{count_colors, 1, CV_8UC3};
    // https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0
    cv::randu(m, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    std::vector<cv::Scalar> result;
    result.reserve(count_colors);
    for (int i = 0; i < count_colors; ++i)
    {
      auto v = m.at<cv::Vec3b>(i);
      result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
  }

  const size_t count_colors_{255};
  const std::vector<cv::Scalar> colors_{GenerateRandomColors(count_colors_)};

  void PlotFlow(cv::Mat &flow, std::vector<PointType> const &pts0,
                std::vector<PointType> const &pts1, cv::Size offset0,
                cv::Size offset1) const
  {
    for (size_t index = 0; index < pts0.size(); ++index)
    {
      PointType pt0{pts0[index]};
      PointType pt1{pts1[index]};
      pt0.x += offset0.width;
      pt0.y += offset0.height;
      pt1.x += offset1.width;
      pt1.y += offset1.height;
      const cv::Scalar lineColor{colors_[index % count_colors_]};
      const int lineThickness{2};
      cv::line(flow, pt0, pt1, lineColor, lineThickness);
    }
  }

  void SaveStereoFrame(const StereoFrame<cv::Mat> &frame) const
  {
    std::stringstream ss_file_name;
    ss_file_name << "frame_" << std::dec << std::setw(4) << std::setfill('0')
                 << frame_index_;
    std::string file_name{ss_file_name.str()};

    // https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    cv::imwrite(file_name + "_left.png", frame.image_left_);
    cv::imwrite(file_name + "_right.png", frame.image_right_);
  }

  template <typename T> static T centroid(const std::vector<T> &pts)
  {
    if (pts.empty())
    {
      return T(0.0f, 0.0f); // 或抛出异常
    }
    // 用 (0,0) 作为初始累加值，每次将点坐标加到累加器上
    auto sum = std::ranges::fold_left(pts, T(0.0f, 0.0f), [](T acc, const T &p)
                                      { return T(acc.x + p.x, acc.y + p.y); });
    float n  = static_cast<float>(pts.size());
    return T(sum.x / n, sum.y / n);
  }

public:
  void StartOdometer()
  {
    bool init{false};
    // === 初始化 CLAHE 实例 ===
    // clipLimit: 对比度限制阈值，一般取 2.0 到 4.0 之间。值越大，对比度增强越强，但也可能引入更多噪声。
    // tileGridSize: 图像划分的网格大小，通常为 8x8。
    cv::Ptr<cv::CLAHE> clahe{
        cv::createCLAHE(3.0, cv::Size(8, 8)),
    };

    StereoFrame<cv::Mat> prev_frame;
    std::vector<PointType> corners_prev_left;
    std::vector<PointType> corners_prev_right;
    std::vector<PointType> corners_next_left;
    std::vector<PointType> corners_next_right;

    data_output_ << "timestamp, "
                    "dqw, dqx, dqy, dqz, "
                    "dpx, dpy, dpz, "
                    "qw, qx, qy, qz, "
                    "px, py, pz, "
                    "\n";

    Eigen::Quaternion<value_type> attitude{
        Eigen::Quaternion<value_type>::Identity(),
    };
    Eigen::Vector<value_type, 3> position{
        Eigen::Vector<value_type, 3>::Zero(),
    };
    Eigen::Quaternion<value_type> delta_rotation{
        Eigen::Quaternion<value_type>::Identity(),
    };
    Eigen::Vector<value_type, 3> delta_position{
        Eigen::Vector<value_type, 3>::Zero(),
    };

    while (loader_)
    {
#if (!START_VISUALIZATION)
      // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
      if (!rclcpp::ok())
      {
        break;
      }
#endif

      StereoFrame<cv::Mat> frame{loader_()};

      std::cerr << "[INFO] 正在处理第 " << frame_index_++ << " 张图片 ...\n";

      if (!init)
      {
        init       = true;
        prev_frame = std::move(frame);
        ++loader_;
        continue;
      }

      auto [image_prev_left_rectified, image_prev_right_rectified]
          = euroc_.remap(prev_frame.image_left_, prev_frame.image_right_);
      // TODO 在计算灰度图像之前，是否应该利用传统的图像增强技术（例如滤波）进行图像预处理？
      auto [image_prev_left_grayscale, image_prev_right_grayscale]
          = euroc_.grayscale(image_prev_left_rectified,
                             image_prev_right_rectified);

      auto [image_next_left_rectified, image_next_right_rectified]
          = euroc_.remap(frame.image_left_, frame.image_right_);
      auto [image_next_left_grayscale, image_next_right_grayscale]
          = euroc_.grayscale(image_next_left_rectified,
                             image_next_right_rectified);

      // === 应用 CLAHE 图像增强 ===
      // 直接在原灰度图变量上进行原地操作（或者覆写），保证后续特征提取和视差计算全部基于增强后的图像
      clahe->apply(image_prev_left_grayscale, image_prev_left_grayscale);
      clahe->apply(image_prev_right_grayscale, image_prev_right_grayscale);
      clahe->apply(image_next_left_grayscale, image_next_left_grayscale);
      clahe->apply(image_next_right_grayscale, image_next_right_grayscale);

      const bool found_corners{
          detector_.FindCorners(
              image_prev_left_grayscale, image_prev_right_grayscale,
              image_next_left_grayscale, image_next_right_grayscale,
              corners_prev_left, corners_prev_right, corners_next_left,
              corners_next_right),
      };
      if (found_corners)
      {
        // 当视图之间的旋转、平移未知（例如从上一帧右目到下一帧右目，从上一帧左目到下一帧左目）时：
        // 1. 八点法求解基础矩阵 F
        // 2. 求解本质矩阵 E = K_right^T * F * K_left
        // 3. 分解本质矩阵 E = T_antisym * R
        // 4. 三角化
        std::cerr
            << "\t数据关联特征点个数 = " //
            << corners_prev_left.size() << "," << corners_prev_right.size()
            << "," << corners_next_right.size() << ","
            << corners_next_left.size() << ","
            << "\n"
            // 计算点集重心，用来检测是否出现死循环（发生死循环时，点集重心保持不变）
            << "\t点集重心 = " << centroid(corners_prev_left) << ","
            << centroid(corners_prev_right) << ","
            << centroid(corners_next_right) << ","
            << centroid(corners_next_left) << ","
            << "\n";

        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
        const cv::Mat fundamental_matrix{
            cv::findFundamentalMat(corners_prev_left, corners_next_left),
        };

        cv::Mat camera_matrix;
        cv::eigen2cv(euroc_.mat_cam_intrinsic_rectified_, camera_matrix);
        const cv::Mat essential_matrix{
            cv::findEssentialMat(corners_prev_left, corners_next_left,
                                 camera_matrix),
        };

        cv::Mat matR, vecT, triangulatedPoints;
        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga2ee9f187170acece29c5172c2175e7ae
        const int number_inliers_pass_chirality_check{
            // 通过“手性检查”的特征点个数
            cv::recoverPose(essential_matrix, corners_prev_left,
                            corners_next_left, camera_matrix, matR, vecT,
                            threshold_for_pose_recovery_, cv::noArray(),
                            triangulatedPoints),
        };

        delta_rotation = Eigen::Quaternion<value_type>{
            Eigen::Matrix<value_type, 3, 3>{
                {matR.template at<value_type>(0, 0),
                 matR.template at<value_type>(0, 1),
                 matR.template at<value_type>(0, 2)},
                {matR.template at<value_type>(1, 0),
                 matR.template at<value_type>(1, 1),
                 matR.template at<value_type>(1, 2)},
                {matR.template at<value_type>(2, 0),
                 matR.template at<value_type>(2, 1),
                 matR.template at<value_type>(2, 2)},
            },
        };
        cv::cv2eigen(vecT, delta_position);

        // 打印旋转轴、旋转角点、平移距离、平移方向
        {
          Eigen::AngleAxis<value_type> delta_rotation_vector{delta_rotation};
          auto delta_rotation_vector_angle{delta_rotation_vector.angle()};
          auto delta_rotation_vector_axis{delta_rotation_vector.axis()};
          if (std::abs(delta_rotation_vector_axis.x()) < 1e-8)
          {
            delta_rotation_vector_axis.x() = 0.0;
          }
          if (std::abs(delta_rotation_vector_axis.y()) < 1e-8)
          {
            delta_rotation_vector_axis.y() = 0.0;
          }
          if (std::abs(delta_rotation_vector_axis.z()) < 1e-8)
          {
            delta_rotation_vector_axis.z() = 0.0;
          }
          auto delta_position_norm{delta_position.norm()};
          auto delta_position_direction{delta_position.normalized()};
          if (std::abs(delta_position_direction.x()) < 1e-8)
          {
            delta_position_direction.x() = 0.0;
          }
          if (std::abs(delta_position_direction.y()) < 1e-8)
          {
            delta_position_direction.y() = 0.0;
          }
          if (std::abs(delta_position_direction.z()) < 1e-8)
          {
            delta_position_direction.z() = 0.0;
          }

          std::cerr << "\t旋转角度 = " //
                    << delta_rotation_vector_angle << " rad\n"
                    << "\t旋转轴 = "                          //
                    << "["                                    //
                    << delta_rotation_vector_axis.x() << ", " //
                    << delta_rotation_vector_axis.y() << ", " //
                    << delta_rotation_vector_axis.z()         //
                    << "]\n"
                    << "\t平移距离 = " //
                    << delta_position_norm << "\n"
                    << "\t平移方向 = "                      //
                    << "["                                  //
                    << delta_position_direction.x() << ", " //
                    << delta_position_direction.y() << ", " //
                    << delta_position_direction.z()         //
                    << "]\n";
        }

        if (number_inliers_pass_chirality_check < 8)
        {
          std::cerr << "\t通过“手性检查”的特征点个数 ("
                    << number_inliers_pass_chirality_check
                    << ") 过少! 当前图像帧无法用于姿态估计!\n";
        }
        else
        {

          // double sampsonDistance_cv{NAN};
          // try
          // {
          //   auto pt1 = [&]
          //   {
          //     cv::Mat pts;
          //     cv::Mat(corners_prev_left)
          //         .reshape(1, static_cast<int>(corners_prev_left.size()))
          //         .convertTo(pts, CV_64F); // 转换为 double
          //     return pts;
          //   }();
          //   auto pt2 = [&]
          //   {
          //     cv::Mat pts;
          //     cv::Mat(corners_next_left)
          //         .reshape(1, static_cast<int>(corners_next_left.size()))
          //         .convertTo(pts, CV_64F); // 转换为 double
          //     return pts;
          //   }();
          //   auto F = [&]
          //   {
          //     cv::Mat res;
          //     fundamental_matrix_cv.convertTo(res, CV_64F);
          //     return res;
          //   }();
          //   std::cerr << "pt1.type() -> " << pt1.type() << "\n" // CV_64FC2
          //             << "pt2.type() -> " << pt2.type() << "\n" // CV_64FC2
          //             << "F.type() -> " << F.type() << "\n";    // CV_64F
          //   sampsonDistance_cv = cv::sampsonDistance(pt1, pt2, F);
          // }
          // catch (const std::exception &e)
          // {
          //   std::cerr << "Error @ `sampsonDistance_cv`: " << e.what() << "\n";
          //   throw e;
          // }

          // auto pixel_left{
          //     Util::ConvertToMatrix3X_EigenMap(corners_prev_left),
          // };
          // auto pixel_right{
          //     Util::ConvertToMatrix3X_EigenMap(corners_next_left),
          // };

          // typename EPA::TriangulationConfig tri_config{
          //     euroc_.mat_cam_intrinsic_rectified_.template cast<value_type>(),
          //     euroc_.mat_cam_intrinsic_rectified_.template cast<value_type>(),
          //     Eigen::Matrix<value_type, 3, 3>::Identity(),
          //     Eigen::Vector<value_type, 3>::Zero(),
          //     pixel_left,
          //     pixel_right,
          // };
          // auto fundamental_matrix{
          //     EPA::EstimateFundamentalMatrix(tri_config),
          // };

          // double sampsonDistance_handmade{NAN};
          // try
          // {
          //   auto pt1 = [&]
          //   {
          //     cv::Mat pts;
          //     cv::Mat(corners_prev_left)
          //         .reshape(1, static_cast<int>(corners_prev_left.size()))
          //         .convertTo(pts, CV_64F); // 转换为 double
          //     return pts;
          //   }();
          //   auto pt2 = [&]
          //   {
          //     cv::Mat pts;
          //     cv::Mat(corners_next_left)
          //         .reshape(1, static_cast<int>(corners_next_left.size()))
          //         .convertTo(pts, CV_64F); // 转换为 double
          //     return pts;
          //   }();
          //   auto F = [&]
          //   {
          //     cv::Mat matF_cv;
          //     cv::eigen2cv(
          //         Eigen::Matrix3d{fundamental_matrix.template cast<double>()},
          //         matF_cv);
          //     return matF_cv;
          //   }();
          //   std::cerr << "pt1.type() -> " << pt1.type() << "\n"
          //             << "pt2.type() -> " << pt2.type() << "\n"
          //             << "F.type() -> " << F.type() << "\n";
          //   sampsonDistance_handmade = cv::sampsonDistance(pt1, pt2, F);
          // }
          // catch (const std::exception &e)
          // {
          //   std::cerr << "Error @ `sampsonDistance_handmade`: " << e.what()
          //             << "\n";
          //   throw e;
          // }

          // std::cerr << "\tFundamental Matrix (handmade) = " << fundamental_matrix
          //           << "\n"
          //           << "\t\tSampson Distance = " << sampsonDistance_handmade
          //           << "\n"
          //           << "\tFundamental Matrix (OpenCV 2) = "
          //           << fundamental_matrix_cv << "\n"
          //           << "\t\tSampson Distance = " << sampsonDistance_cv << "\n";
          // 得出结论：OpenCV 算出的基础矩阵的误差比我的算法的误差要小很多

          // auto essential_matrix{
          //     tri_config.ComputeEssentialMatrix(fundamental_matrix),
          // };
          // auto &&[matR1, matR2, vecT]
          //     = EPA::DecomposeEssentialMatrix(essential_matrix);
          // // 第 1 种组合方式 (matR1 + vecT)
          // tri_config.rotation_    = matR1;
          // tri_config.translation_ = vecT;
          // auto model_points1{
          //     EPA::Triangulate(tri_config),
          // };
          // auto model_points1_count_positive_z{
          //     (model_points1.row(2).array() > 0.0).count(),
          // };
          // // 第 2 种组合方式 (matR1 + -vecT)
          // tri_config.translation_ = -vecT;
          // auto model_points2{
          //     EPA::Triangulate(tri_config),
          // };
          // auto model_points2_count_positive_z{
          //     (model_points2.row(2).array() > 0.0).count(),
          // };
          // // 第 3 种组合方式 (matR2 + -vecT)
          // tri_config.rotation_ = matR2;
          // auto model_points3{
          //     EPA::Triangulate(tri_config),
          // };
          // auto model_points3_count_positive_z{
          //     (model_points3.row(2).array() > 0.0).count(),
          // };
          // // 第 4 种组合方式 (matR2 + vecT)
          // tri_config.translation_ = vecT;
          // auto model_points4{
          //     EPA::Triangulate(tri_config),
          // };
          // auto model_points4_count_positive_z{
          //     (model_points4.row(2).array() > 0.0).count(),
          // };

          // // 找出 Z 分量大于零的列向量个数最多的组合方式
          // decltype(matR1) best_matR;
          // decltype(vecT) best_vecT;
          // decltype(model_points1) best_model_points;
          // decltype(model_points1_count_positive_z) max_positive_z_count{0};

          // if (model_points1_count_positive_z > max_positive_z_count)
          // {
          //   max_positive_z_count = model_points1_count_positive_z;
          //   best_matR            = matR1;
          //   best_vecT            = vecT;
          //   best_model_points    = std::move(model_points1);
          // }
          // if (model_points2_count_positive_z > max_positive_z_count)
          // {
          //   max_positive_z_count = model_points2_count_positive_z;
          //   best_matR            = matR1;
          //   best_vecT            = -vecT;
          //   best_model_points    = std::move(model_points2);
          // }
          // if (model_points3_count_positive_z > max_positive_z_count)
          // {
          //   max_positive_z_count = model_points3_count_positive_z;
          //   best_matR            = matR2;
          //   best_vecT            = -vecT;
          //   best_model_points    = std::move(model_points3);
          // }
          // if (model_points4_count_positive_z > max_positive_z_count)
          // {
          //   max_positive_z_count = model_points4_count_positive_z;
          //   best_matR            = matR2;
          //   best_vecT            = vecT;
          //   best_model_points    = std::move(model_points4);
          // }

          // Eigen::Quaternion<value_type> delta_rotation{best_matR};

          // 更新状态
          position = attitude * delta_position + position;
          attitude = attitude * delta_rotation;

          data_output_ << std::fixed                 //
                       << std::setprecision(18)      //
                       << frame.timestamp_ << ", "   //
                       << delta_rotation.w() << ", " //
                       << delta_rotation.x() << ", " //
                       << delta_rotation.y() << ", " //
                       << delta_rotation.z() << ", " //
                       << delta_position.x() << ", " //
                       << delta_position.y() << ", " //
                       << delta_position.z() << ", " //
                       << attitude.w() << ", "       //
                       << attitude.x() << ", "       //
                       << attitude.y() << ", "       //
                       << attitude.z() << ", "       //
                       << position.x() << ", "       //
                       << position.y() << ", "       //
                       << position.z()               //
                       << "\n";

          // 路标点在世界坐标系中的齐次坐标 ( 变量类型实际上是 `std::vector<Eigen::Vector4d>` )
          // std::vector<Landmark> landmarks;
          // landmarks.reserve(corners_prev_left.size());
          // 利用 EKF 解算姿态
          // ekf_.Update(corners_prev_left, corners_prev_right, corners_next_left,
          //             corners_next_right, landmarks);
          // TODO 将计算得到的路标点，与之前记录的路标点进行比较，检测回环（即是否回到起点或其附近位置）
        }

#if (!START_VISUALIZATION)
        Publish(frame.timestamp_, attitude, position);
#endif
      }

      // std::cerr << "\t最终检测到 " << corners_prev_left.size()
      //           << " 个角点 ...\n";

      // 可视化
      if (visualize_)
      {
        // 拼接图像，绘制角点连线，显示前后相邻两帧的双目视图
        {
          cv::Mat vis_top, vis_bottom, vis;

          // https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gaab5ceee39e0580f879df645a872c6bf7
          cv::hconcat(image_prev_left_rectified, image_prev_right_rectified,
                      vis_top);
          cv::hconcat(image_next_left_rectified, image_next_right_rectified,
                      vis_bottom);
          cv::vconcat(vis_top, vis_bottom, vis);

          // 绘制角点连线
          if (found_corners)
          {
            const cv::Size maskSize{vis.size()};
            cv::Mat mask{
                cv::Mat::zeros(maskSize, image_prev_left_rectified.type())};
            const cv::Size imageSize{image_prev_left_rectified.size()};
            PlotFlow(mask, corners_prev_left, corners_prev_right,
                     cv::Size{0, 0}, cv::Size{imageSize.width, 0});
            PlotFlow(mask, corners_prev_right, corners_next_right,
                     cv::Size{imageSize.width, 0}, imageSize);
            PlotFlow(mask, corners_next_right, corners_next_left, imageSize,
                     cv::Size{0, imageSize.height});
            PlotFlow(mask, corners_next_left, corners_prev_left,
                     cv::Size{0, imageSize.height}, cv::Size{0, 0});
            cv::add(mask, vis, vis);
          }

          cv::imshow(loopback_window_name_, vis);
        }

        // 修改窗口名称
        {
          std::stringstream ss_window_title;
          ss_window_title << "Image Frame [#" << std::setw(4)
                          << std::setfill('0') << frame_index_ << "]";
          cv::setWindowTitle(loopback_window_name_, ss_window_title.str());
        }

        // 显示视差图、深度图
        if (plot_disparity_and_depth_)
        {
          // 绘制视差图

          cv::Ptr<cv::StereoSGBM> sgbm{cv::StereoSGBM::create( //
              0,                                               //
              96,                                              //
              9,                                               //
              8 * 9 * 9,                                       //
              32 * 9 * 9,                                      //
              1,                                               //
              63,                                              //
              10,                                              //
              100,                                             //
              32                                               //
              )};
          cv::Mat disparity_sgbm, disparity;
          sgbm->compute(image_next_left_grayscale, image_next_right_grayscale,
                        disparity_sgbm);
          disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0);
          cv::imshow(disparity_window_name_, disparity / 96.0);

          // 绘制深度图 (手动计算)

          // const value_type f{static_cast<value_type>(euroc_.focal_length_rectified_)};
          // const value_type b{static_cast<value_type>(euroc_.baseline_length_)};
          // const value_type fb{f * b};
          // cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32F);
          // // 避免除以零（视差为0表示无穷远或无效点）
          // cv::Mat mask = disparity > 0;
          // cv::divide(fb, disparity, depth);
          // depth.setTo(0, ~mask);
          // // 显示深度图（注意：深度图通常值域很大，直接 imshow 会全白，需要归一化或伪彩色渲染）
          // cv::Mat depth_display;
          // // 过滤掉极远的点以便观察，假设我们只关心 0.1m 到 10m
          // cv::threshold(depth, depth_display, 10.0, 10.0, cv::THRESH_TRUNC);
          // cv::normalize(depth_display, depth_display, 0, 255, cv::NORM_MINMAX,
          //               CV_8U);
          // cv::applyColorMap(depth_display, depth_display, cv::COLORMAP_JET);
          // cv::imshow(depth_window_name_, depth_display);

          // 绘制深度图 (内置函数)

          cv::Mat xyz;
          // Q 矩阵包含了 f 和 b 的关系
          // xyz 是一个三通道图像，xyz.at<cv::Vec3f>(y, x)[2] 就是该像素的深度 z
          cv::reprojectImageTo3D(disparity, xyz, euroc_.Q);
          // 提取深度
          cv::Mat depth_clean;
          cv::extractChannel(xyz, depth_clean, 2);
          // 物理范围过滤 (只保留 0.1m 到 10m)
          cv::Mat mask{(depth_clean > 0.1) & (depth_clean < 10.0)};
          cv::Mat filtered_depth{cv::Mat::zeros(depth_clean.size(), CV_32F)};
          depth_clean.copyTo(filtered_depth, mask);
          // 转换为 8 位图像以便显示
          cv::Mat display_map;
          // 注意：不要直接 normalize，先根据你感兴趣的最大距离缩放
          // 比如 10米 对应 255
          filtered_depth.convertTo(display_map, CV_8U, 255.0 / 10.0);
          // 应用伪彩色
          cv::applyColorMap(display_map, display_map, cv::COLORMAP_JET);
          // 将无效区域（原来是0的点）染黑，防止被 ColorMap 染成深蓝色
          display_map.setTo(cv::Scalar(0, 0, 0), ~mask);
          cv::imshow(depth_window_name_, display_map);
        }

        // 处理键盘事件
        switch (InterpretKeyEvent(0))
        {
        case KeyEvent::EXIT:
          return;
        default:
          break;
        }
      }

      prev_frame = std::move(frame);
      ++loader_;
      corners_prev_left  = std::move(corners_next_left);
      corners_prev_right = std::move(corners_next_right);
    }
  }
};

#if (!START_VISUALIZATION)
void StartVisualSlamAsRosNode(int argc, char *argv[])
{
  // 初始化 ROS 2
  rclcpp::init(argc, argv);

  try
  {
    // 通过共享指针以符合 rclcpp 标准的方式实例化 Node
    auto slam_node = std::make_shared<StereoSlam<cv::Point2f>>(false);

    // 注意：当前实现完全基于数据集驱动并且没有用到其他 Subscription。
    // 使用阻塞式调用 StartOdometer() 不会影响消息发布机制。
    // 如果后续你需要加入 callback(例如接收真正的相机传感器 topic)，
    // 你需要将 StartOdometer 放到 std::thread 中并在 main 线程调用 rclcpp::spin(slam_node);
    slam_node->StartOdometer();
  }
  catch (const std::exception &e)
  {
    std::cerr << "ROS 2 Node Error: " << e.what() << "\n";
  }

  // 关闭 ROS 2 实例
  rclcpp::shutdown();
}
#endif

int main(int argc, char *argv[])
{
#if START_VISUALIZATION
  (void) argc;
  (void) argv;
  StereoSlam<cv::Point2f>(true).StartOdometer();
#else
  StartVisualSlamAsRosNode(argc, argv);
#endif
  return 0;
}
