#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <ios>
#include <print>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <sophus/so3.hpp>

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

template <typename PointType = cv::Point2f>
struct StereoSlam
{
public:
  using value_type = typename PointType::value_type;
  using Vector3    = Eigen::Vector<value_type, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Attitude   = Sophus::SO3<value_type>;

  const EuRoC::EuRoC euroc_{};

private:
  std::ofstream file_fast_{"estimated_motion.csv",
                           std::ios::out | std::ios::trunc};
  std::ofstream file_traj_{"estimated_trajectory.csv",
                           std::ios::out | std::ios::trunc};
  ImageDataLoader loader_;
  CornerDetection::FastDetector<PointType> detector_{};

public:
  StereoSlam() = delete;

  StereoSlam(const StereoSlam &) = delete;

  StereoSlam(StereoSlam &&) = delete;

  StereoSlam(const std::filesystem::path &path_mav0) : loader_{path_mav0} {}

  ~StereoSlam() {}

private:
  static cv::Point2f centroid(const std::vector<cv::Point2f> &pts)
  {
    if (pts.empty())
    {
      return cv::Point2f(0.0f, 0.0f); // 或抛出异常
    }
    // 用 (0,0) 作为初始累加值，每次将点坐标加到累加器上
    auto sum{std::ranges::fold_left(
        pts, cv::Point2f(0.0f, 0.0f), [](cv::Point2f acc, const cv::Point2f &p)
        { return cv::Point2f(acc.x + p.x, acc.y + p.y); }
    )};
    auto n{static_cast<float>(pts.size())};
    return cv::Point2f(sum.x / n, sum.y / n);
  }

public:
  void StartOdometer()
  {
    bool init{false};
    // 初始化 CLAHE 实例
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

    // 打印表头
    std::print(file_traj_, "#timestamp [ns],"
                           "p_x [m],p_y [m],p_z [m],"
                           "q_w [],q_x [],q_y [],q_z []\n");
    std::print(file_fast_, "#timestamp [ns],"
                           "r_x [rad],r_y [rad],r_z [rad],"
                           "t_x [],t_y [],t_z []\n");

    Attitude attitude{};
    Vector3 position{Vector3::Zero()};
    Attitude delta_rotation{};
    Vector3 delta_position{Vector3::Zero()};
    Quaternion attitude_quat{Quaternion::Identity()};

    // 打印位姿
    std::print(file_traj_,
               // 时间戳
               "{:020d},"
               // 位置
               "{:.18f},{:.18f},{:.18f},"
               // 朝向
               "{:.18f},{:.18f},{:.18f},{:.18f}\n",
               prev_frame.timestamp_, position.x(), position.y(), position.z(),
               attitude_quat.w(), attitude_quat.x(), attitude_quat.y(),
               attitude_quat.z());

    while (loader_)
    {
      StereoFrame<cv::Mat> frame{loader_()};
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

      // 应用 CLAHE 图像增强
      // 直接在原灰度图变量上进行原地操作（或者覆写），保证后续特征提取和视差计算全部基于增强后的图像
      clahe->apply(image_prev_left_grayscale, image_prev_left_grayscale);
      clahe->apply(image_prev_right_grayscale, image_prev_right_grayscale);
      clahe->apply(image_next_left_grayscale, image_next_left_grayscale);
      clahe->apply(image_next_right_grayscale, image_next_right_grayscale);

      const bool found_corners{
          detector_.FindCorners(image_prev_left_grayscale,
                                image_prev_right_grayscale,
                                image_next_left_grayscale,
                                image_next_right_grayscale, corners_prev_left,
                                corners_prev_right, corners_next_left,
                                corners_next_right),
      };
      if (found_corners)
      {
        // 当视图之间的旋转、平移未知（例如从上一帧右目到下一帧右目，从上一帧左目到下一帧左目）时：
        // 1. 八点法求解基础矩阵 F
        // 2. 求解本质矩阵 E = K_right^T * F * K_left
        // 3. 分解本质矩阵 E = T_antisym * R
        // 4. 三角化
        // 5. 作为 PnP 问题，解出帧间旋转和平移
        // 6. 离散时间积分，计算实时位姿

        // 世界坐标系 (即以左目光心为原点的坐标系) 中路标点的齐次坐标
        cv::Mat landmarks_homo;

        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        cv::triangulatePoints(euroc_.P0, euroc_.P1, corners_prev_left,
                              corners_prev_right, landmarks_homo);

        if (landmarks_homo.cols == 0)
        {
          goto continue_loop;
        }

        // 世界坐标系中路标点的非齐次坐标
        cv::Mat landmarks_nonhomo;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gac42edda3a3a0f717979589fcd6ac0035
        cv::convertPointsFromHomogeneous(landmarks_homo.t(), landmarks_nonhomo);

        // 相机内参矩阵
        cv::Mat camera_matrix;
        cv::eigen2cv(euroc_.mat_cam_intrinsic_rectified_, camera_matrix);

        // 旋转向量与平移向量
        cv::Mat rVec_cv, tVec_cv;
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        cv::solvePnPRansac(landmarks_nonhomo, corners_next_left, camera_matrix,
                           cv::noArray(), rVec_cv, tVec_cv);

        // 数据类型转换
        Vector3 rVec_eigen;
        cv::cv2eigen(rVec_cv, rVec_eigen);
        rVec_eigen     = -rVec_eigen;
        delta_rotation = Attitude::exp(rVec_eigen);
        cv::cv2eigen(tVec_cv, delta_position);
        delta_position = -(delta_rotation * delta_position);

        Vector3 normalized_translation{delta_position.normalized()};

        // 打印角位移和平移方向，用来喂给 ESKF
        std::print(file_fast_,
                   // 时间戳
                   "{:020d},"
                   // 角位移
                   "{:.18f},{:.18f},{:.18f},"
                   // 单位化平移向量
                   "{:.18f},{:.18f},{:.18f}\n",
                   prev_frame.timestamp_, rVec_eigen.x(), rVec_eigen.y(),
                   rVec_eigen.z(), normalized_translation.x(),
                   normalized_translation.y(), normalized_translation.z());

        // 更新状态
        position = position + attitude * delta_position;
        attitude = attitude * delta_rotation;

        // 打印位姿
        attitude_quat = attitude.unit_quaternion();
        std::print(file_traj_,
                   // 时间戳
                   "{:020d},"
                   // 位置
                   "{:.18f},{:.18f},{:.18f},"
                   // 朝向
                   "{:.18f},{:.18f},{:.18f},{:.18f}\n",
                   frame.timestamp_, position.x(), position.y(), position.z(),
                   attitude_quat.w(), attitude_quat.x(), attitude_quat.y(),
                   attitude_quat.z());
      }

    continue_loop:
      prev_frame = std::move(frame);
      ++loader_;
      corners_prev_left  = std::move(corners_next_left);
      corners_prev_right = std::move(corners_next_right);
    }
  }
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::print(stderr, "Usage: {} <path_mav0>\n", argv[0]);
    return 1;
  }
  std::print(stderr, "OpenCV Version: {}\n", cv::getVersionString());
  StereoSlam<cv::Point2f>{std::filesystem::path{argv[1]}}.StartOdometer();
  return 0;
}
