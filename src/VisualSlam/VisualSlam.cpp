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

#include "EuRoC.hpp"
#include "FastDetector.hpp"
#include "ImageDataLoader.hpp"

struct StereoSlam
{
public:
  const std::string loopback_window_name_{"VisualSlam"};
  const std::string disparity_window_name_{"Disparity"};
  const EuRoC::EuRoC euroc_{};

private:
  ImageDataLoader loader_{};
  CornerDetection::FastDetector<> detector_{};
  size_t frame_index_{};

public:
  StereoSlam()
  {
    cv::namedWindow(loopback_window_name_, cv::WINDOW_NORMAL);
    cv::namedWindow(disparity_window_name_, cv::WINDOW_NORMAL);
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

  template <typename PointType = cv::Point2f>
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
  void StartOdometer(bool visualize = true)
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
        // TODO 找到足够角点用于解算姿态
      }

      std::cerr << "\t最终检测到 " << corners_prev_left.size()
                << " 个角点 ...\n";

      // 可视化
      if (visualize)
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
          PlotFlow(mask, corners_prev_left, corners_prev_right, cv::Size{0, 0},
                   cv::Size{imageSize.width, 0});
          PlotFlow(mask, corners_prev_right, corners_next_right,
                   cv::Size{imageSize.width, 0}, imageSize);
          PlotFlow(mask, corners_next_right, corners_next_left, imageSize,
                   cv::Size{0, imageSize.height});
          PlotFlow(mask, corners_next_left, corners_prev_left,
                   cv::Size{0, imageSize.height}, cv::Size{0, 0});
          cv::add(mask, vis, vis);
        }

        cv::imshow(loopback_window_name_, vis);

        {
          std::stringstream ss_window_title;
          ss_window_title << "Image Frame [#" << std::setw(4)
                          << std::setfill('0') << frame_index_ << "]";
          cv::setWindowTitle(loopback_window_name_, ss_window_title.str());
        }

        // 绘制视差图
        {
          cv::Ptr<cv::StereoSGBM> sgbm{cv::StereoSGBM::create(
              0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32)};
          cv::Mat disparity_sgbm, disparity;
          sgbm->compute(image_next_left_grayscale, image_next_right_grayscale,
                        disparity_sgbm);
          disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0);
          cv::imshow(disparity_window_name_, disparity / 96.0);
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
