#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
#include "EuRoC.hpp"
#include "FastDetector.hpp"
#include "ImageDataLoader.hpp"

template <typename PointType = cv::Point2f> struct StereoSlam
{
public:
  const std::string loopback_window_name_{"VisualSlam"};
  const std::string disparity_window_name_{"Disparity"};
  const std::string depth_window_name_{"Depth Map"};
  const EuRoC::EuRoC euroc_{};

  using Landmark = Eigen::Vector<typename PointType::value_type, 4>;

private:
  ImageDataLoader loader_{};
  CornerDetection::FastDetector<> detector_{};
  size_t frame_index_{};
  // EKF<> ekf_{};

public:
  StereoSlam()
  {
    cv::namedWindow(loopback_window_name_, cv::WINDOW_NORMAL);
    cv::namedWindow(disparity_window_name_, cv::WINDOW_NORMAL);
    cv::namedWindow(depth_window_name_, cv::WINDOW_NORMAL);
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
      cv::Point2f pt0{pts0[index]};
      cv::Point2f pt1{pts1[index]};
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
  void StartOdometer(bool visualize                = true,
                     bool plot_disparity_and_depth = false)
  {
    bool init{false};
    StereoFrame<cv::Mat> prev_frame;
    std::vector<cv::Point2f> corners_prev_left;
    std::vector<cv::Point2f> corners_prev_right;
    std::vector<cv::Point2f> corners_next_left;
    std::vector<cv::Point2f> corners_next_right;

    while (loader_)
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

      const bool found_corners{detector_.FindCorners(
          image_prev_left_grayscale, image_prev_right_grayscale,
          image_next_left_grayscale, image_next_right_grayscale,
          corners_prev_left, corners_prev_right, corners_next_left,
          corners_next_right)};
      if (found_corners)
      {
        // 当视图之间的旋转、平移未知（例如从上一帧右目到下一帧右目，从上一帧左目到下一帧左目）时：
        // 1. 八点法求解基础矩阵 F
        // 2. 求解本质矩阵 E = K_right^T * F * K_left
        // 3. 分解本质矩阵 E = T_antisym * R
        // 4. 三角化

        // 路标点在世界坐标系中的齐次坐标 ( 变量类型实际上是 `std::vector<Eigen::Vector4d>` )
        // std::vector<Landmark> landmarks;
        // landmarks.reserve(corners_prev_left.size());
        // 利用 EKF 解算姿态
        // ekf_.Update(corners_prev_left, corners_prev_right, corners_next_left,
        //             corners_next_right, landmarks);
        // TODO 将计算得到的路标点，与之前记录的路标点进行比较，检测回环（即是否回到起点或其附近位置）
      }

      std::cerr << "\t最终检测到 " << corners_prev_left.size()
                << " 个角点 ...\n";

      // 可视化
      if (visualize)
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

        if (plot_disparity_and_depth)
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

          // const float f{static_cast<float>(euroc_.focal_length_rectified_)};
          // const float b{static_cast<float>(euroc_.baseline_length_)};
          // const float fb{f * b};
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

int main()
{
  try
  {
    StereoSlam slam{};
    slam.StartOdometer();
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << "\n";
  }
  return 0;
}
