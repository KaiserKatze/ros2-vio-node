#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
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
#include <string_view>
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
#include "euroc_vio/Integrator.hpp"

struct StereoSlam : public VisualIntegrator
{
public:
  using PointType  = cv::Point2f;
  using value_type = double;
  using Vector3    = Eigen::Vector<value_type, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Attitude   = Sophus::SO3<value_type>;

  const EuRoC::EuRoC euroc_{};

private:
  std::ofstream file_traj_{"estimated_trajectory.csv",
                           std::ios::out | std::ios::trunc};
  ImageDataLoader loader_;
  CornerDetection::FastDetector detector_{};
  bool do_visualization_{false};
  const std::string window_name_{"Stereo Visual SLAM"};

public:
  StereoSlam() = delete;

  StereoSlam(const StereoSlam &) = delete;

  StereoSlam(StereoSlam &&) = delete;

  StereoSlam(const std::filesystem::path &path_mav0, bool do_visualization) :
    loader_{path_mav0}, do_visualization_{do_visualization}
  {
    if (do_visualization_)
    {
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    }
  }

  ~StereoSlam() {}

private:
  /**
   * @brief 打印表头和初始位姿
   */
  void WriteDataHeader()
  {
    // 打印表头
    std::print(file_traj_, "#timestamp [ns],"
                           "p_x [m],p_y [m],p_z [m],"
                           "q_w [],q_x [],q_y [],q_z []\n");
    // 打印初始位姿
    WriteDataContent(0);
  }

  /**
   * @brief 打印位姿
   */
  void WriteDataContent(std::int64_t timestamp)
  {
    auto pos{this->VisualIntegrator::pose_.translation()};
    auto att{this->VisualIntegrator::pose_.so3().unit_quaternion()};
    std::print(file_traj_,
               // 时间戳
               "{:020d},"
               // 位置
               "{:.18f},{:.18f},{:.18f},"
               // 朝向
               "{:.18f},{:.18f},{:.18f},{:.18f}\n",
               timestamp,                 //
               pos.x(), pos.y(), pos.z(), //
               att.w(), att.x(), att.y(), att.z());
  }

  // 辅助函数：将灰度图转为彩色（BGR）
  static cv::Mat toColor(const cv::Mat &img) noexcept
  {
    if (img.channels() == 1)
    {
      cv::Mat color;
      cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
      return color;
    }
    return img.clone(); // 已是彩色则复制一份
  }

  static cv::Mat
  stitchImages(const cv::Mat &image_prev_left_rectified,
               const cv::Mat &image_prev_right_rectified,
               const cv::Mat &image_prev_left_grayscale,
               const cv::Mat &image_prev_right_grayscale,
               const cv::Mat &image_next_left_rectified,
               const cv::Mat &image_next_right_rectified,
               const cv::Mat &image_next_left_grayscale,
               const cv::Mat &image_next_right_grayscale) noexcept
  {
    // 转换所有图像为彩色
    cv::Mat img1{toColor(image_prev_left_rectified)};
    cv::Mat img2{toColor(image_prev_right_rectified)};
    cv::Mat img3{toColor(image_prev_left_grayscale)};
    cv::Mat img4{toColor(image_prev_right_grayscale)};
    cv::Mat img5{toColor(image_next_left_rectified)};
    cv::Mat img6{toColor(image_next_right_rectified)};
    cv::Mat img7{toColor(image_next_left_grayscale)};
    cv::Mat img8{toColor(image_next_right_grayscale)};

    // 第一行：前4张
    cv::Mat row1;
    cv::hconcat(std::vector<cv::Mat>{img1, img2, img3, img4}, row1);

    // 第二行：后4张
    cv::Mat row2;
    cv::hconcat(std::vector<cv::Mat>{img5, img6, img7, img8}, row2);

    // 垂直拼接两行
    cv::Mat result;
    cv::vconcat(std::vector<cv::Mat>{row1, row2}, result);

    return result;
  }

public:
  void StartOdometer()
  {
    WriteDataHeader();

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

    // 世界坐标系 (即以左目光心为原点的坐标系) 中路标点的齐次坐标
    cv::Mat landmarks_homo;
    // 世界坐标系中路标点的非齐次坐标
    cv::Mat landmarks_nonhomo;

    // 相机内参矩阵
    cv::Mat camera_matrix;
    cv::eigen2cv(euroc_.mat_cam_intrinsic_rectified_, camera_matrix);

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

      // 预测路标点在下一帧的投影
      if (landmarks_homo.cols > 0) // 确认是否存在路标点
      {
        // (左目相机相对于世界坐标系的姿态 = 载具相对于世界坐标系的姿态)取逆
        // 得到(从世界坐标系到左目相机坐标系的变换)
        auto inv_pose_left{this->VisualIntegrator::pose_.inverse()};
        cv::Mat rVec_left;
        cv::Mat tVec_left;
        cv::eigen2cv(inv_pose_left.so3().log(), rVec_left);
        cv::eigen2cv(inv_pose_left.translation(), tVec_left);
        // (右目相机相对于世界坐标系的姿态)取逆
        // 得到(从世界坐标系到右目相机坐标系的变换)
        auto inv_pose_right{inv_pose_left};
        // 下面相当于令 (inv_pose_right := T_C1C0 * inv_pose_left)
        inv_pose_right.translation().x() -= euroc_.baseline_length_;
        cv::Mat rVec_right;
        cv::Mat tVec_right;
        cv::eigen2cv(inv_pose_right.so3().log(), rVec_right);
        cv::eigen2cv(inv_pose_right.translation(), tVec_right);
        // 准备存储空间
        size_t capacity{std::max(corners_next_left.size(),
                                 corners_next_right.size())};
        capacity = std::max(static_cast<size_t>(landmarks_homo.cols), capacity);
        corners_next_left.reserve(capacity);
        corners_next_right.reserve(capacity);
        // https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        // 由于图像经过了立体矫正，所以畸变系数全为零
        cv::projectPoints(landmarks_nonhomo, rVec_left, tVec_left,
                          camera_matrix, cv::noArray(), corners_next_left,
                          cv::noArray());
        cv::projectPoints(landmarks_nonhomo, rVec_right, tVec_right,
                          camera_matrix, cv::noArray(), corners_next_right,
                          cv::noArray());
      }

      // 将前一帧、后一帧的左目、右目的原始图像、增强后的图像展示出来
      cv::Mat vis;
      if (do_visualization_)
      {
        vis = stitchImages(
            image_prev_left_rectified, image_prev_right_rectified,
            image_prev_left_grayscale, image_prev_right_grayscale,
            image_next_left_rectified, image_next_right_rectified,
            image_next_left_grayscale, image_next_right_grayscale
        );
        cv::imshow(window_name_, vis);
        cv::waitKey(5);
      }

      const bool found_corners{
          detector_.FindCorners(image_prev_left_grayscale,
                                image_prev_right_grayscale,
                                image_next_left_grayscale,
                                image_next_right_grayscale, corners_prev_left,
                                corners_prev_right, corners_next_left,
                                corners_next_right, true),
      };
      assert(corners_prev_left.size() == corners_prev_right.size()
             && corners_prev_left.size() == corners_next_left.size()
             && corners_prev_left.size() == corners_next_right.size()
             && "Inconsistent corner array sizes");
      if (found_corners)
      {
        // 当视图之间的旋转、平移未知（例如从上一帧右目到下一帧右目，从上一帧左目到下一帧左目）时：
        // 1. 八点法求解基础矩阵 F
        // 2. 求解本质矩阵 E = K_right^T * F * K_left
        // 3. 分解本质矩阵 E = T_antisym * R
        // 4. 三角化
        // 5. 作为 PnP 问题，解出帧间旋转和平移
        // 6. 离散时间积分，计算实时位姿

        // https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        cv::triangulatePoints(euroc_.P0, euroc_.P1, corners_prev_left,
                              corners_prev_right, landmarks_homo);

        if (landmarks_homo.cols > 0)
        {
          // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gac42edda3a3a0f717979589fcd6ac0035
          cv::convertPointsFromHomogeneous(landmarks_homo.t(),
                                           landmarks_nonhomo);

          // 旋转向量与平移向量
          cv::Mat rVec_cv, tVec_cv;
          // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
          cv::solvePnPRansac(landmarks_nonhomo, corners_next_left,
                             camera_matrix, cv::noArray(), rVec_cv, tVec_cv);

          // 数据类型转换
          Vector3 rVec_eigen;
          cv::cv2eigen(rVec_cv, rVec_eigen);
          rVec_eigen = -rVec_eigen;
          Attitude delta_rotation{Attitude::exp(rVec_eigen)};
          Vector3 delta_position{Vector3::Zero()};
          cv::cv2eigen(tVec_cv, delta_position);
          delta_position = -(delta_rotation * delta_position);

          // 更新状态
          this->VisualIntegrator::Update(delta_rotation, delta_position);

          // 打印位姿
          WriteDataContent(prev_frame.timestamp_);
        }

        // 只有在追踪成功时，才将本帧的有效特征点保存为下一帧的“上一帧点”
        corners_prev_left  = std::move(corners_next_left);
        corners_prev_right = std::move(corners_next_right);
      }
      else
      {
        // 追踪失败时，彻底清空状态，下一帧将重新全图检测角点
        corners_prev_left.clear();
        corners_prev_right.clear();
      }

      prev_frame = std::move(frame);
      ++loader_;
    }
  }
};

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    std::print(stderr, "Usage: {} [--visualize] <path_mav0>\n", argv[0]);
    return 1;
  }
  std::print(stderr, "OpenCV Version: {}\n", cv::getVersionString());

  bool do_visualization{false};
  auto path_mav0{std::filesystem::path{argv[argc - 1]}};

  // 检测是否存在的 --visualize 选项
  // 如果存在该选项则将 do_visualization 赋值为 true
  if (argc == 3 && std::string_view{argv[1]} == "--visualize")
  {
    do_visualization = true;
    std::print(stderr, "Visualization enabled.\n");
  }

  StereoSlam inst{path_mav0, do_visualization};
  inst.StartOdometer();

  return 0;
}
