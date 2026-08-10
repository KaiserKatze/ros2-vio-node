#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <ios>
#include <limits>
#include <numeric>
#include <print>
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

#include "FastDetector.hpp"
#include "ImageDataLoader.hpp"
#include "euroc_vio/DatumImu.hpp"
#include "euroc_vio/ErrorStateKalmanFilter.hpp"
#include "euroc_vio/EuRoC.hpp"
#include "euroc_vio/Integrator.hpp"
#include "euroc_vio/SensorYaml.hpp"
#include "euroc_vio/StereoObservation.hpp"
#include "euroc_vio/TrackingConfig.hpp"
#include "euroc_vio/util.hpp"

// OpenCV 提取角点时只提供 cv::Point2f 类型
using PointType = cv::Point2f;

template <typename T>
static T GetMatValue(const cv::Mat &mat, int row, int col)
{
  if (row < 0 || row >= mat.rows)
  {
    throw std::out_of_range(
        std::format("Row index out of range: row={} rows={} {}", row, mat.rows,
                    Util::FormatCvMatInfo("mat", mat))
    );
  }

  const int scalar_cols_per_row{mat.cols * mat.channels()};
  if (col < 0 || col >= scalar_cols_per_row)
  {
    throw std::out_of_range(std::format(
        "Column/channel index out of range: col={} scalar_cols_per_row={} {}",
        col, scalar_cols_per_row, Util::FormatCvMatInfo("mat", mat)
    ));
  }

  if (mat.depth() == CV_32F)
  {
    return static_cast<T>(mat.ptr<float>(row)[col]);
  }
  else if (mat.depth() == CV_64F)
  {
    return static_cast<T>(mat.ptr<double>(row)[col]);
  }

  throw std::runtime_error(std::format("Unsupported OpenCV matrix depth: {}",
                                       Util::FormatCvMatInfo("mat", mat)));
}

namespace FastVIO
{

template <typename value_type>
static std::vector<StereoObservation<value_type>>
CreateStereoObservationSet(const std::vector<PointType> &pts_left,
                           const std::vector<PointType> &pts_right, //
                           const cv::Mat &landmarks_nonhomo,
                           const std::vector<std::uint32_t> &feature_ids)
{
  assert(pts_left.size() == pts_right.size()
         && pts_left.size() == feature_ids.size());
  assert(landmarks_nonhomo.depth() == CV_32F
         || landmarks_nonhomo.depth() == CV_64F);
  auto len{pts_left.size()};
  using len_t = decltype(len);
  assert(static_cast<len_t>(landmarks_nonhomo.rows) == len);
  assert(landmarks_nonhomo.cols * landmarks_nonhomo.channels() == 3);
  std::vector<StereoObservation<value_type>> result;
  result.reserve(len);
  for (len_t i = 0; i < len; ++i)
  {
    auto pt_left{pts_left[i]};
    auto pt_right{pts_right[i]};
    result.emplace_back(feature_ids[i],
                        Eigen::Vector<value_type, 2>{
                            pt_left.x,
                            pt_left.y,
                        },
                        Eigen::Vector<value_type, 2>{
                            pt_right.x,
                            pt_right.y,
                        },
                        Eigen::Vector<value_type, 3>{
                            GetMatValue<value_type>(landmarks_nonhomo, i, 0),
                            GetMatValue<value_type>(landmarks_nonhomo, i, 1),
                            GetMatValue<value_type>(landmarks_nonhomo, i, 2),
                        });
  }
  return result;
}

struct SlamConfig
{
  bool do_visualization_;
};

struct CornerTrackingStats
{
  using Clock = std::chrono::steady_clock;

  std::size_t total_frame_count_{0};
  std::size_t success_frame_count_{0};
  std::size_t failed_frame_count_{0};
  std::size_t visual_task_count_{0};
  double visual_task_total_ms_{0.0};
  double visual_task_min_ms_{std::numeric_limits<double>::max()};
  double visual_task_max_ms_{0.0};
  decltype(Clock::now()) visual_task_begin_;

  [[nodiscard]]
  double GetSuccessRate() const noexcept
  {
    if (total_frame_count_ == 0)
    {
      return 0.0;
    }
    return 100.0 * static_cast<double>(success_frame_count_)
           / static_cast<double>(total_frame_count_);
  }

  void NextFrame() noexcept
  {
    ++total_frame_count_;
  }

  auto GetFrameId() const noexcept
  {
    return total_frame_count_;
  }

  void PrintFrameBegin(bool use_hint, std::size_t input_corner_count) const
  {
    std::println(stderr,
                 "\n[FindCorners] ===== 开始特征跟踪 =====\n"
                 "\t当前帧编号={} | 是否启用先验信息={} | 输入角点数={} | "
                 "最少所需角点数={} | 最多所需角点数={}",
                 total_frame_count_, use_hint, input_corner_count,
                 CornerDetection::AbstractDetector::minCorners,
                 CornerDetection::AbstractDetector::maxCorners);
  }

  void
  RecordFrameResult(bool success,
                    const std::vector<PointType> &corners_prev_left,
                    const std::vector<PointType> &corners_prev_right,
                    const std::vector<PointType> &corners_next_left,
                    const std::vector<PointType> &corners_next_right) noexcept
  {
    if (success)
    {
      ++success_frame_count_;
    }
    else
    {
      ++failed_frame_count_;
    }

    CornerDetection::AbstractDetector::PrintCornerSetSizes(
        "\t角点个数: ", corners_prev_left, corners_prev_right,
        corners_next_left, corners_next_right
    );
  }

  void PrintSummary() const noexcept
  {
    std::println(stderr,
                 "\n[FindCorners] ===== 总统计 =====\n"
                 "成功帧数={} 失败帧数={} 总帧数={} 成功率={:.2f}%\n"
                 "平均耗时={}ms 最小耗时={}ms 最大耗时={}ms",
                 success_frame_count_, failed_frame_count_, total_frame_count_,
                 GetSuccessRate(), GetAverageElapsedTime(),
                 GetMinimalElapsedTime(), GetMaximalElapsedTime());
  }

  void StartTimer() noexcept
  {
    visual_task_begin_ = Clock::now();
  }

  [[nodiscard]]
  auto EndTimer() noexcept
  {
    ++visual_task_count_;
    const auto visual_task_end{Clock::now()};
    const double visual_task_elapsed_ms{
        std::chrono::duration<double, std::milli>{visual_task_end
                                                  - visual_task_begin_}
            .count(),
    };

    visual_task_total_ms_ += visual_task_elapsed_ms;
    visual_task_min_ms_ = std::min(visual_task_min_ms_, visual_task_elapsed_ms);
    visual_task_max_ms_ = std::max(visual_task_max_ms_, visual_task_elapsed_ms);
    return visual_task_elapsed_ms;
  }

  void RecordElapsedTime(std::int64_t timestamp, double elapsed_ms) const
  {
    const auto frame_id{GetFrameId()};
    static std::ofstream file_{"CornerTrackingStats.csv"};
    if (frame_id <= 1)
    {
      std::print(file_, "#timestamp [ns],"
                        "elapsed time [ms]\n");
    }
    std::print(file_, "{},{}\n", timestamp, elapsed_ms);
  }

  [[nodiscard]]
  double GetAverageElapsedTime() const noexcept
  {
    if (visual_task_count_ == 0)
    {
      return 0.0;
    }
    return visual_task_total_ms_ / static_cast<double>(visual_task_count_);
  }

  [[nodiscard]]
  double GetMinimalElapsedTime() const noexcept
  {
    return visual_task_min_ms_;
  }

  [[nodiscard]]
  double GetMaximalElapsedTime() const noexcept
  {
    return visual_task_max_ms_;
  }
};

class TrajectoryWriter
{
protected:
  /**
   * @brief 打印表头和初始位姿
   */
  template <typename SophusSE3>
  void WriteDataHeader(const SophusSE3 &pose)
  {
    // 打印表头
    std::print(file_traj_, "#timestamp [ns],"
                           "p_x [m],p_y [m],p_z [m],"
                           "q_w [],q_x [],q_y [],q_z []\n");
    // 打印初始位姿
    WriteDataContent(0, pose);
  }

  /**
   * @brief 打印位姿
   */
  template <typename SophusSE3>
  void WriteDataContent(std::int64_t timestamp, const SophusSE3 &pose)
  {
    auto pos{pose.translation()};
    auto att{pose.so3().unit_quaternion()};
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

private:
  std::ofstream file_traj_{"estimated_trajectory.csv",
                           std::ios::out | std::ios::trunc};
};

struct StereoSlam : public VisualIntegrator, private TrajectoryWriter
{
public:
  using value_type = double;
  using Vector3    = Eigen::Vector<value_type, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Attitude   = Sophus::SO3<value_type>;
  using ESKF       = ErrorStateKalmanFilter<value_type>;

  // PnP RANSAC 参数: 输入图像已完成矫正与去畸变, 内点重投影残差应在 1~2px 量级,
  // OpenCV 默认的 8px 阈值会把错误的 3D-2D 对应也接纳为内点
  static constexpr float kPnpReprojectionErrorPixels{2.0F};
  static constexpr int kPnpRansacIterations{200};
  static constexpr double kPnpConfidence{0.99};
  // 内点低于此数时相对位姿不可信, 该帧不参与积分
  static constexpr int kMinPnpInliers{12};
  // 静止初始化所用的 IMU 样本数 (EuRoC 开头有静止段, 200Hz 下约 1 秒)
  static constexpr std::size_t kStaticInitSampleCount{200};

  // 标定参数由 mav0/cam0|cam1/sensor.yaml 加载, 必须先于依赖它的成员构造
  const EuRoC::EuRoC euroc_;

private:
  const std::string window_name_{"Stereo Visual SLAM"};

  ImageDataLoader loader_;
  SlamConfig config_;

  // IMU 采样序列 (体坐标系) 与其消费游标: ESKF 的预测步依赖它
  std::vector<DatumImu> imu_data_;
  std::size_t imu_cursor_{0};

  CornerDetection::FastDetector detector_{};
  // 初始化 CLAHE 实例
  // clipLimit: 对比度限制阈值，一般取 2.0 到 4.0 之间。值越大，对比度增强越强，但也可能引入更多噪声。
  // tileGridSize: 图像划分的网格大小，通常为 8x8。
  cv::Ptr<cv::CLAHE> clahe_{
      cv::createCLAHE(3.0, cv::Size(8, 8)),
  };
  ESKF eskf_;

#pragma region CONSTRUCTORS

public:
  StereoSlam() = delete;

  StereoSlam(const StereoSlam &) = delete;

  StereoSlam(StereoSlam &&) = delete;

  StereoSlam(const std::filesystem::path &path_mav0, const SlamConfig &config) :
    euroc_{path_mav0}, loader_{path_mav0}, config_{config}
  {
    if (config_.do_visualization_)
    {
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    }

    typename ESKF::Config eskf_config;

    cv::cv2eigen(euroc_.P0, eskf_config.stereo_camera_model_.proj_left_);
    cv::cv2eigen(euroc_.P1, eskf_config.stereo_camera_model_.proj_right_);
    // 没有这个外参, ESKF 会把矫正后左目系当成体坐标系, 路标点与雅可比全部错位
    // SVD 投影确保旋转块严格正交, 满足 Sophus::SE3 的前置条件
    const Eigen::Matrix3d rmat_B_rectC0{
        euroc_.T_B_rectifiedC0.block<3, 3>(0, 0),
    };
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd{
        rmat_B_rectC0,
        Eigen::ComputeFullU | Eigen::ComputeFullV,
    };
    eskf_config.stereo_camera_model_.transform_cam0_ = typename ESKF::Pose{
        Eigen::Matrix3d{svd.matrixU() * svd.matrixV().transpose()},
        Eigen::Vector3d{euroc_.T_B_rectifiedC0.block<3, 1>(0, 3)},
    };

    auto path_imu0_yaml{path_mav0 / "imu0" / "sensor.yaml"};
    auto opt_sensor_config_imu0{SensorYaml::ReadSensorYaml(path_imu0_yaml)};
    if (!opt_sensor_config_imu0.has_value())
    {
      throw std::runtime_error{std::format(
          "Failed to parse IMU config yaml '{}'.", path_imu0_yaml.c_str()
      )};
    }
    const auto sensor_config_imu0{opt_sensor_config_imu0.value()};
    eskf_config.imu_rate_ = sensor_config_imu0.rate_hz_;

    eskf_ = ESKF{eskf_config};

    // EuRoC 的 imu0 已在体坐标系下表达, 无需再做旋转
    imu_data_ = DatumImu::Load((path_mav0 / "imu0" / "data.csv").string(),
                               Sophus::SO3d{});
    if (imu_data_.empty())
    {
      throw std::runtime_error{std::format(
          "IMU data is empty: '{}'.", (path_mav0 / "imu0" / "data.csv").c_str()
      )};
    }

    // 把 YAML 中的物理噪声参数交给 ESKF, 用于过程协方差传播
    eskf_.SetGyroscopeNoiseDensity(sensor_config_imu0.gyroscope_noise_density_);
    eskf_.SetGyroscopeRandomWalk(sensor_config_imu0.gyroscope_random_walk_);
    eskf_.SetAccelerometerNoiseDensity(
        sensor_config_imu0.accelerometer_noise_density_
    );
    eskf_.SetAccelerometerRandomWalk(
        sensor_config_imu0.accelerometer_random_walk_
    );

    InitializeNominalStateFromStaticImu();
  }

#pragma endregion

  ~StereoSlam()
  {
    if (config_.do_visualization_)
    {
      cv::destroyAllWindows();
    }
  }

private:
  // 用开头的静止段估计初始姿态: 静止时比力的均值方向即重力反方向。
  // 按 SetNominalState 的约定, 把姿态设为单位阵、让体坐标系下的重力向量
  // 吸收初始朝向的全部不确定性 (线性化效果优于反过来假设重力已知)
  void InitializeNominalStateFromStaticImu()
  {
    const std::size_t sample_count{
        std::min(kStaticInitSampleCount, imu_data_.size()),
    };
    Vector3 accel_mean{Vector3::Zero()};
    Vector3 gyro_mean{Vector3::Zero()};
    for (std::size_t i = 0; i < sample_count; ++i)
    {
      accel_mean += imu_data_[i].linear_acceleration_;
      gyro_mean += imu_data_[i].angular_velocity_;
    }
    accel_mean /= static_cast<value_type>(sample_count);
    gyro_mean /= static_cast<value_type>(sample_count);

    typename ESKF::NominalStateVariable init_state;
    // 姿态取单位阵, 于是体坐标系与世界系重合, 重力直接取 -比力
    init_state.pose_    = typename ESKF::Pose{};
    init_state.gravity_ = -accel_mean;
    eskf_.SetNominalState(init_state);

    std::println(stderr,
                 "[INIT] 静止段样本={}个 比力均值=[{:.4f} {:.4f} {:.4f}] "
                 "(模长 {:.4f} m/s^2) 陀螺均值=[{:.5f} {:.5f} {:.5f}]",
                 sample_count, accel_mean.x(), accel_mean.y(), accel_mean.z(),
                 accel_mean.norm(), gyro_mean.x(), gyro_mean.y(),
                 gyro_mean.z());
  }

  // 把截止到 timestamp 的所有 IMU 采样送进 ESKF 完成名义状态前推与协方差传播
  void PropagateImuUntil(std::int64_t timestamp)
  {
    while (imu_cursor_ < imu_data_.size()
           && imu_data_[imu_cursor_].timestamp_ <= timestamp)
    {
      eskf_.ImuUpdate(&imu_data_[imu_cursor_++]);
    }
  }

  // 辅助函数：将灰度图转为彩色（BGR）
  static cv::Mat ConvertGrayToBGR(const cv::Mat &img) noexcept
  {
    cv::Mat color;
    cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
    return color;
  }

  static cv::Mat stitchImages(const cv::Mat &image_prev_left_rectified,
                              const cv::Mat &image_prev_right_rectified,
                              cv::Mat image_prev_left_grayscale,
                              cv::Mat image_prev_right_grayscale,
                              const cv::Mat &image_next_left_rectified,
                              const cv::Mat &image_next_right_rectified,
                              cv::Mat image_next_left_grayscale,
                              cv::Mat image_next_right_grayscale) noexcept
  {
    image_prev_left_grayscale  = ConvertGrayToBGR(image_prev_left_grayscale);
    image_prev_right_grayscale = ConvertGrayToBGR(image_prev_right_grayscale);
    image_next_left_grayscale  = ConvertGrayToBGR(image_next_left_grayscale);
    image_next_right_grayscale = ConvertGrayToBGR(image_next_right_grayscale);

    // 第一行：前4张
    cv::Mat row1;
    cv::hconcat(
        std::vector<cv::Mat>{
            image_prev_left_rectified,
            image_prev_right_rectified,
            image_prev_left_grayscale,
            image_prev_right_grayscale,
        },
        row1
    );

    // 第二行：后4张
    cv::Mat row2;
    cv::hconcat(
        std::vector<cv::Mat>{
            image_next_left_rectified,
            image_next_right_rectified,
            image_next_left_grayscale,
            image_next_right_grayscale,
        },
        row2
    );

    // 垂直拼接两行
    cv::Mat result;
    cv::vconcat(std::vector<cv::Mat>{row1, row2}, result);

    return result;
  }

  void HandleFrame(const StereoFrame<cv::Mat> &frame, cv::Mat &left_rectified,
                   cv::Mat &right_rectified, cv::Mat &left_grayscale,
                   cv::Mat &right_grayscale) const noexcept
  {
    std::tie(left_rectified, right_rectified)
        = euroc_.remap(frame.image_left_, frame.image_right_);
    std::tie(left_grayscale, right_grayscale)
        = euroc_.grayscale(left_rectified, right_rectified);

    clahe_->apply(left_grayscale, left_grayscale);
    clahe_->apply(right_grayscale, right_grayscale);
  }

public:
  void StartOdometer()
  {
    WriteDataHeader(eskf_.GetNominalState().pose_);

    bool init_frame{false};
    bool init_landmarks{false};

    CornerTrackingStats corner_tracking_stats{};

    std::int64_t timestamp;
    cv::Mat image_prev_left_rectified;
    cv::Mat image_prev_right_rectified;
    cv::Mat image_prev_left_grayscale;
    cv::Mat image_prev_right_grayscale;
    cv::Mat image_next_left_rectified;
    cv::Mat image_next_right_rectified;
    cv::Mat image_next_left_grayscale;
    cv::Mat image_next_right_grayscale;

    std::vector<PointType> corners_prev_left;
    std::vector<PointType> corners_prev_right;
    std::vector<PointType> corners_next_left;
    std::vector<PointType> corners_next_right;
    std::vector<std::uint32_t> feature_ids;

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
      corner_tracking_stats.StartTimer();
      if (!init_frame)
      {
        init_frame = true;

        timestamp = frame.timestamp_;
        // 首帧之前的 IMU 只用于静止初始化, 从首帧成像时刻起才开始前推
        PropagateImuUntil(frame.timestamp_);
        HandleFrame(frame, image_prev_left_rectified,
                    image_prev_right_rectified, image_prev_left_grayscale,
                    image_prev_right_grayscale);

        ++loader_;
        continue;
      }

      // 预测步: 把两帧之间的高频 IMU 采样全部积分到当前成像时刻,
      // 名义状态与误差协方差同步传播 (过程噪声在此注入)
      PropagateImuUntil(frame.timestamp_);

      HandleFrame(frame, image_next_left_rectified, image_next_right_rectified,
                  image_next_left_grayscale, image_next_right_grayscale);

      // 预测路标点在下一帧的投影
      if (!eskf_.GetLandmarks().empty()) // 确认是否存在路标点
      {
        std::tie(corners_next_left, corners_next_right, feature_ids)
            = eskf_.PredictNextCorners<cv::Point2f, int>(euroc_.image_width,
                                                         euroc_.image_height);
      }

      // 将前一帧、后一帧的左目、右目的原始图像、增强后的图像展示出来
      cv::Mat vis;
      if (config_.do_visualization_)
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

      const bool use_hint{landmarks_homo.cols > 0};
      corner_tracking_stats.NextFrame();
      corner_tracking_stats.PrintFrameBegin(use_hint, corners_prev_left.size());
      const bool found_corners{
          detector_.FindCorners(image_prev_left_grayscale,  //
                                image_prev_right_grayscale, //
                                image_next_left_grayscale,  //
                                image_next_right_grayscale, //
                                corners_prev_left,          //
                                corners_prev_right,         //
                                corners_next_left,          //
                                corners_next_right,         //
                                feature_ids,                //
                                use_hint),
      };
      corner_tracking_stats.RecordFrameResult(found_corners, corners_prev_left,
                                              corners_prev_right,
                                              corners_next_left,
                                              corners_next_right);
      assert(corners_prev_left.size() == corners_prev_right.size()
             && corners_prev_left.size() == corners_next_left.size()
             && corners_prev_left.size() == corners_next_right.size()
             && corners_prev_left.size() == feature_ids.size());

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

          if (!init_landmarks)
          {
            init_landmarks = true;
            auto corner_set_prev{CreateStereoObservationSet<value_type>(
                corners_prev_left, corners_prev_right, landmarks_nonhomo,
                feature_ids
            )};
            eskf_.StereoUpdate(timestamp, corner_set_prev);
          }

          // 当前帧的观测必须配当前帧三角化出的路标点: 沿用上一帧的三角化结果
          // 会把 t-1 时刻相机系下的坐标当成 t 时刻的, 使路标数据库混入不同历元
          cv::Mat landmarks_homo_next;
          cv::Mat landmarks_nonhomo_next;
          cv::triangulatePoints(euroc_.P0, euroc_.P1, corners_next_left,
                                corners_next_right, landmarks_homo_next);
          if (landmarks_homo_next.cols > 0)
          {
            cv::convertPointsFromHomogeneous(landmarks_homo_next.t(),
                                             landmarks_nonhomo_next);
            auto corner_set_next{CreateStereoObservationSet<value_type>(
                corners_next_left, corners_next_right, landmarks_nonhomo_next,
                feature_ids
            )};
            // 观测更新: 视觉残差修正 IMU 预测出的名义状态与零偏
            eskf_.StereoUpdate(frame.timestamp_, corner_set_next);
          }

          // 旋转向量与平移向量
          cv::Mat rVec_cv, tVec_cv;
          // 显式指定 RANSAC 参数: 图像已矫正去畸变, 默认 8px 重投影阈值过宽,
          // 会把错误对应当作内点; 收回 inliers 以便判定本次求解是否可信
          cv::Mat pnp_inliers;
          const bool pnp_ok{
              cv::solvePnPRansac(landmarks_nonhomo, corners_next_left,
                                 camera_matrix, cv::noArray(), rVec_cv, tVec_cv,
                                 false, kPnpRansacIterations,
                                 kPnpReprojectionErrorPixels, kPnpConfidence,
                                 pnp_inliers, cv::SOLVEPNP_ITERATIVE),
          };
          const int pnp_inlier_count{pnp_ok ? pnp_inliers.rows : 0};

          // 内点过少时本帧的相对位姿不可信, 宁可跳过也不要把野值积分进轨迹
          if (pnp_ok && pnp_inlier_count >= kMinPnpInliers)
          {
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
          }
          else
          {
            std::println(stderr,
                         "\tPnP 求解不可信 (成功={} 内点={}个 < {}个)，"
                         "本帧不更新位姿",
                         pnp_ok, pnp_inlier_count, kMinPnpInliers);
          }

          // 打印位姿: 输出 ESKF 融合后的名义状态 (体坐标系位姿),
          // 而非纯视觉开环积分的结果 —— 后者没有任何漂移修正,
          // 单帧的错误对应会被永久累积
          // 位姿已包含到当前帧的运动, 必须标注当前帧时间戳
          // (此处的 timestamp 仍是上一帧的值, 帧末才推进)
          WriteDataContent(frame.timestamp_, eskf_.GetNominalState().pose_);
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

      const auto visual_task_elapsed_ms{corner_tracking_stats.EndTimer()};
      std::print(stderr,
                 "[VisualTask] 时间戳={} 帧编号={}\n"
                 "\t当前帧是否成功跟踪路标点={}\n"
                 "\t所有路标点个数 (包括新建、活跃的)={}\n"
                 "\t当前帧耗时={:.3f}\n",
                 frame.timestamp_, corner_tracking_stats.GetFrameId(),
                 (found_corners ? "成功" : "失败"), feature_ids.size(),
                 visual_task_elapsed_ms);
      corner_tracking_stats.RecordElapsedTime(timestamp,
                                              visual_task_elapsed_ms);

      timestamp                  = frame.timestamp_;
      image_prev_left_rectified  = std::move(image_next_left_rectified);
      image_prev_right_rectified = std::move(image_next_right_rectified);
      image_prev_left_grayscale  = std::move(image_next_left_grayscale);
      image_prev_right_grayscale = std::move(image_next_right_grayscale);

      ++loader_;
    }

    corner_tracking_stats.PrintSummary();
  }
};

} // namespace FastVIO

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

  FastVIO::SlamConfig config;
  config.do_visualization_ = do_visualization;

  FastVIO::StereoSlam inst{path_mav0, config};
  inst.StartOdometer();

  return 0;
}
