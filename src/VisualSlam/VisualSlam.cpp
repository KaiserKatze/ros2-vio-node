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
#include "Util.hpp"
#include "euroc_vio/main.h"

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

template <typename PointType = cv::Point2f>
struct StereoSlam : public StereoSlamPublisher
{
public:
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
  bool visualize_{true};
  bool plot_disparity_and_depth_{false};

public:
  StereoSlam(bool visualize = true, bool plot_disparity_and_depth = false) :
    visualize_{visualize}, plot_disparity_and_depth_{plot_disparity_and_depth}
  {
    if (visualize_)
    {
      cv::namedWindow(loopback_window_name_, cv::WINDOW_NORMAL);
      if (plot_disparity_and_depth)
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

public:
  void StartOdometer()
  {
    bool init{false};
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
        Eigen::Quaternion<value_type>::Identity()};
    Eigen::Vector<value_type, 3> position{Eigen::Vector<value_type, 3>::Zero()};

    // 引入 rclcpp::ok() 以响应 ROS 2 节点的关闭信号 (如 Ctrl+C)
    while (rclcpp::ok() && loader_)
    {
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
      auto [image_prev_left_grayscale, image_prev_right_grayscale]
          = euroc_.grayscale(image_prev_left_rectified,
                             image_prev_right_rectified);

      auto [image_next_left_rectified, image_next_right_rectified]
          = euroc_.remap(frame.image_left_, frame.image_right_);
      auto [image_next_left_grayscale, image_next_right_grayscale]
          = euroc_.grayscale(image_next_left_rectified,
                             image_next_right_rectified);

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

        const cv::Mat fundamental_matrix_cv{
            cv::findFundamentalMat(corners_prev_left, corners_next_left),
        };

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

        auto pixel_left{
            Util::ConvertToMatrix3X_EigenMap(corners_prev_left),
        };
        auto pixel_right{
            Util::ConvertToMatrix3X_EigenMap(corners_next_left),
        };

        typename EPA::TriangulationConfig tri_config{
            euroc_.mat_cam_intrinsic_rectified_.template cast<value_type>(),
            euroc_.mat_cam_intrinsic_rectified_.template cast<value_type>(),
            Eigen::Matrix<value_type, 3, 3>::Identity(),
            Eigen::Vector<value_type, 3>::Zero(),
            pixel_left,
            pixel_right,
        };
        auto fundamental_matrix{
            EPA::EstimateFundamentalMatrix(tri_config),
        };

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

        auto essential_matrix{
            tri_config.ComputeEssentialMatrix(fundamental_matrix),
        };
        auto &&[matR1, matR2, vecT]
            = EPA::DecomposeEssentialMatrix(essential_matrix);
        // 第 1 种组合方式 (matR1 + vecT)
        tri_config.rotation_    = matR1;
        tri_config.translation_ = vecT;
        auto model_points1{
            EPA::Triangulate(tri_config),
        };
        auto model_points1_count_positive_z{
            (model_points1.row(2).array() > 0.0).count(),
        };
        // 第 2 种组合方式 (matR1 + -vecT)
        tri_config.translation_ = -vecT;
        auto model_points2{
            EPA::Triangulate(tri_config),
        };
        auto model_points2_count_positive_z{
            (model_points2.row(2).array() > 0.0).count(),
        };
        // 第 3 种组合方式 (matR2 + -vecT)
        tri_config.rotation_ = matR2;
        auto model_points3{
            EPA::Triangulate(tri_config),
        };
        auto model_points3_count_positive_z{
            (model_points3.row(2).array() > 0.0).count(),
        };
        // 第 4 种组合方式 (matR2 + vecT)
        tri_config.translation_ = vecT;
        auto model_points4{
            EPA::Triangulate(tri_config),
        };
        auto model_points4_count_positive_z{
            (model_points4.row(2).array() > 0.0).count(),
        };

        // 找出 Z 分量大于零的列向量个数最多的组合方式
        decltype(matR1) best_matR;
        decltype(vecT) best_vecT;
        decltype(model_points1) best_model_points;
        decltype(model_points1_count_positive_z) max_positive_z_count{0};

        if (model_points1_count_positive_z > max_positive_z_count)
        {
          max_positive_z_count = model_points1_count_positive_z;
          best_matR            = matR1;
          best_vecT            = vecT;
          best_model_points    = std::move(model_points1);
        }
        if (model_points2_count_positive_z > max_positive_z_count)
        {
          max_positive_z_count = model_points2_count_positive_z;
          best_matR            = matR1;
          best_vecT            = -vecT;
          best_model_points    = std::move(model_points2);
        }
        if (model_points3_count_positive_z > max_positive_z_count)
        {
          max_positive_z_count = model_points3_count_positive_z;
          best_matR            = matR2;
          best_vecT            = -vecT;
          best_model_points    = std::move(model_points3);
        }
        if (model_points4_count_positive_z > max_positive_z_count)
        {
          max_positive_z_count = model_points4_count_positive_z;
          best_matR            = matR2;
          best_vecT            = vecT;
          best_model_points    = std::move(model_points4);
        }

        Eigen::Quaternion<value_type> quad_rotation{best_matR};

        // 更新状态
        position = attitude * best_vecT + position;
        attitude = attitude * quad_rotation;

        data_output_ << std::fixed                //
                     << std::setprecision(18)     //
                     << frame.timestamp_ << ", "  //
                     << quad_rotation.w() << ", " //
                     << quad_rotation.x() << ", " //
                     << quad_rotation.y() << ", " //
                     << quad_rotation.z() << ", " //
                     << best_vecT.x() << ", "     //
                     << best_vecT.y() << ", "     //
                     << best_vecT.z() << ", "     //
                     << attitude.w() << ", "      //
                     << attitude.x() << ", "      //
                     << attitude.y() << ", "      //
                     << attitude.z() << ", "      //
                     << position.x() << ", "      //
                     << position.y() << ", "      //
                     << position.z()              //
                     << "\n";

        // 路标点在世界坐标系中的齐次坐标 ( 变量类型实际上是 `std::vector<Eigen::Vector4d>` )
        // std::vector<Landmark> landmarks;
        // landmarks.reserve(corners_prev_left.size());
        // 利用 EKF 解算姿态
        // ekf_.Update(corners_prev_left, corners_prev_right, corners_next_left,
        //             corners_next_right, landmarks);
        // TODO 将计算得到的路标点，与之前记录的路标点进行比较，检测回环（即是否回到起点或其附近位置）

        Publish(frame.timestamp_, attitude, position);
      }

      // std::cerr << "\t最终检测到 " << corners_prev_left.size()
      //           << " 个角点 ...\n";

      // 可视化
      if (visualize_)
      {
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

        {
          std::stringstream ss_window_title;
          ss_window_title << "Image Frame [#" << std::setw(4)
                          << std::setfill('0') << frame_index_ << "]";
          cv::setWindowTitle(loopback_window_name_, ss_window_title.str());
        }

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

int main(int argc, char *argv[])
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
  return 0;
}
