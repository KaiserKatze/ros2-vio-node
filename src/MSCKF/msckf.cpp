// msckf.cpp —— 基于 MSCKF 的双目视觉 + IMU 紧耦合里程计 (EuRoC MAV V2_01_easy)
// 依赖: OpenCV 4 / Eigen 3.4 / Sophus / Ceres Solver / yaml-cpp, 标准: C++26
// 标定与噪声参数在运行时用 yaml-cpp 从数据集各 sensor.yaml 读取, 不做硬编码
// 延迟补偿: IMU 插值精确传播到成像时刻, 相机-IMU 时间偏移 t_d 作为状态在线估计
// 重力向量 (gx, gy, gz) 加入误差状态在线估计, 初始化时由静止比力或先验给出
// ZUPT: IMU 方差静止检测触发零速伪量测; 滑窗消费的特征 id 回馈前端删除, 避免重复关联
// OC-EKF 一致性修正: 对 Φ 与 H 施加不可观子空间约束 (全局平移 + 绕重力偏航), 防伪可观
// 前端: 滤波姿态(陀螺积分)辅助光流初值预测 + buildOpticalFlowPyramid 金字塔复用
// 边缘化: 低运动冗余克隆优先剔除, 仅吸收被删克隆上的观测、轨迹保活;
//         长轨迹经延迟初始化升级为 SLAM 特征进入状态向量 [IMU | SLAM特征 | 克隆]
// MonocularUpdate: 融合外部单目算法输出的帧间相对旋转(高精度)与平移方向(无尺度、低精度)
//
// 编译 (或直接使用配套 CMakeLists.txt):
//   g++ -std=c++2c -O3 -march=native msckf.cpp -o msckf $(pkg-config --cflags --libs opencv4 eigen3 yaml-cpp) -lceres -lglog -pthread
// 运行:
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0                      # 静止 IMU 初始化 (默认)
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --init groundtruth   # 真值姿态初始化
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --time-offset 0.005  # t_d 初值(秒), 在线精化
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --output /tmp/v2_01.tum  # 指定轨迹输出路径
// 输出:
//   TUM 格式轨迹 (time px py pz qx qy qz qw), 默认写入当前目录 trajectory_tum.txt

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <print>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ============================ 命令行参数 ============================
enum class InitializationMode
{
  kStaticImu,
  kGroundTruth
};

InitializationMode ParseInitializationValue(std::string_view value)
{
  if (value == "imu")
  {
    return InitializationMode::kStaticImu;
  }
  if (value == "groundtruth" || value == "gt")
  {
    return InitializationMode::kGroundTruth;
  }
  throw std::runtime_error("未知的初始化方式 (可选: imu | groundtruth): "
                           + std::string(value));
}

struct CommandLineOptions
{
  fs::path dataset_root = fs::path{std::getenv("HOME")} / "EuRoC_MAV_Datasets"
                          / "V2_01_easy" / "mav0";
  InitializationMode initialization_mode = InitializationMode::kStaticImu;
  double initial_time_offset      = 0; // 图像时刻 + t_d = 对应的 IMU 时刻 (秒)
  fs::path output_trajectory_path = "trajectory_tum.txt";
};

CommandLineOptions ParseCommandLine(int argc, char **argv)
{
  CommandLineOptions options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string_view argument = argv[i];
    if (argument.starts_with("--init="))
    {
      options.initialization_mode
          = ParseInitializationValue(argument.substr(7));
    }
    else if (argument == "--init")
    {
      if (++i >= argc)
      {
        throw std::runtime_error("--init 缺少取值 (imu | groundtruth)");
      }
      options.initialization_mode = ParseInitializationValue(argv[i]);
    }
    else if (argument.starts_with("--time-offset="))
    {
      options.initial_time_offset = std::stod(std::string(argument.substr(14)));
    }
    else if (argument == "--time-offset")
    {
      if (++i >= argc)
      {
        throw std::runtime_error("--time-offset 缺少取值 (秒)");
      }
      options.initial_time_offset = std::stod(argv[i]);
    }
    else if (argument.starts_with("--output="))
    {
      options.output_trajectory_path = fs::path(argument.substr(9));
    }
    else if (argument == "--output")
    {
      if (++i >= argc)
      {
        throw std::runtime_error("--output 缺少取值 (轨迹文件路径)");
      }
      options.output_trajectory_path = fs::path(argv[i]);
    }
    else
    {
      options.dataset_root = fs::path(argument);
    }
  }
  return options;
}

// ================ 标定参数读取 (yaml-cpp 解析 EuRoC sensor.yaml) ================
constexpr double kGravity
    = 9.81; // 重力先验值, 实际重力向量在误差状态中在线估计

struct CameraCalibration
{
  cv::Mat camera_matrix;
  cv::Mat distortion;
  cv::Size image_size;
  Sophus::SE3d body_from_camera;
};

struct ImuNoiseParameters
{
  double gyro_noise_density  = 0;
  double gyro_random_walk    = 0;
  double accel_noise_density = 0;
  double accel_random_walk   = 0;
};

Sophus::SE3d ReadBodyFromSensorPose(const YAML::Node &sensor_node)
{
  const YAML::Node pose_data = sensor_node["T_BS"]["data"];
  if (!pose_data || pose_data.size() != 16)
  {
    throw std::runtime_error("sensor.yaml 缺少 4x4 的 T_BS.data");
  }
  Eigen::Matrix4d matrix;
  for (int i = 0; i < 16; ++i)
  {
    matrix(i / 4, i % 4) = pose_data[i].as<double>();
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      matrix.topLeftCorner<3, 3>(), Eigen::ComputeFullU | Eigen::ComputeFullV
  );
  const Eigen::Matrix3d orthonormal_rotation
      = svd.matrixU() * svd.matrixV().transpose();
  return {Sophus::SO3d(orthonormal_rotation), matrix.topRightCorner<3, 1>()};
}

CameraCalibration LoadCameraCalibration(const fs::path &sensor_yaml_path)
{
  const YAML::Node node       = YAML::LoadFile(sensor_yaml_path.string());
  const YAML::Node intrinsics = node["intrinsics"]; // [fu, fv, cu, cv]
  const YAML::Node resolution = node["resolution"];
  const YAML::Node distortion_coefficients = node["distortion_coefficients"];
  if (!intrinsics || intrinsics.size() != 4 || !resolution
      || !distortion_coefficients)
  {
    throw std::runtime_error("相机标定字段缺失: " + sensor_yaml_path.string());
  }

  CameraCalibration calibration;
  calibration.camera_matrix
      = cv::Mat_<double>({3, 3}, {intrinsics[0].as<double>(), 0.0,
                                  intrinsics[2].as<double>(), 0.0,
                                  intrinsics[1].as<double>(),
                                  intrinsics[3].as<double>(), 0.0, 0.0, 1.0});
  calibration.distortion
      = cv::Mat::zeros(1, static_cast<int>(distortion_coefficients.size()),
                       CV_64F);
  for (int i = 0; i < calibration.distortion.cols; ++i)
  {
    calibration.distortion.at<double>(0, i)
        = distortion_coefficients[i].as<double>();
  }
  calibration.image_size = {resolution[0].as<int>(), resolution[1].as<int>()};
  calibration.body_from_camera = ReadBodyFromSensorPose(node);
  return calibration;
}

ImuNoiseParameters LoadImuNoiseParameters(const fs::path &sensor_yaml_path)
{
  const YAML::Node node = YAML::LoadFile(sensor_yaml_path.string());
  return {.gyro_noise_density = node["gyroscope_noise_density"].as<double>(),
          .gyro_random_walk   = node["gyroscope_random_walk"].as<double>(),
          .accel_noise_density
          = node["accelerometer_noise_density"].as<double>(),
          .accel_random_walk = node["accelerometer_random_walk"].as<double>()};
}

// ============================ 数据集读取 ============================
struct ImuSample
{
  double time                         = 0;
  Eigen::Vector3d angular_velocity    = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
};

ImuSample InterpolateImuSample(const ImuSample &before, const ImuSample &after,
                               double time)
{
  const double interval = after.time - before.time;
  const double ratio
      = interval > 1e-9 ? std::clamp((time - before.time) / interval, 0.0, 1.0)
                        : 0.0;
  return {.time = time,
          .angular_velocity
          = before.angular_velocity
            + ratio * (after.angular_velocity - before.angular_velocity),
          .linear_acceleration
          = before.linear_acceleration
            + ratio * (after.linear_acceleration - before.linear_acceleration)};
}

struct StereoFrame
{
  double time = 0;
  fs::path left_image_path;
  fs::path right_image_path;
};

std::vector<ImuSample> LoadImuSamples(const fs::path &dataset_root)
{
  std::ifstream file(dataset_root / "imu0" / "data.csv");
  if (!file)
  {
    throw std::runtime_error("无法打开 "
                             + (dataset_root / "imu0" / "data.csv").string());
  }
  std::vector<ImuSample> samples;
  std::string line;
  while (std::getline(file, line))
  {
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#')
    {
      continue;
    }
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    long long timestamp_ns = 0;
    ImuSample sample;
    if (stream >> timestamp_ns >> sample.angular_velocity.x()
        >> sample.angular_velocity.y() >> sample.angular_velocity.z()
        >> sample.linear_acceleration.x() >> sample.linear_acceleration.y()
        >> sample.linear_acceleration.z())
    {
      sample.time = static_cast<double>(timestamp_ns) * 1e-9;
      samples.push_back(sample);
    }
  }
  return samples;
}

std::vector<StereoFrame> LoadStereoFrames(const fs::path &dataset_root)
{
  std::ifstream file(dataset_root / "cam0" / "data.csv");
  if (!file)
  {
    throw std::runtime_error("无法打开 "
                             + (dataset_root / "cam0" / "data.csv").string());
  }
  std::vector<StereoFrame> frames;
  std::string line;
  while (std::getline(file, line))
  {
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#')
    {
      continue;
    }
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    long long timestamp_ns = 0;
    std::string file_name;
    if (!(stream >> timestamp_ns >> file_name))
    {
      continue;
    }
    StereoFrame frame{.time = static_cast<double>(timestamp_ns) * 1e-9,
                      .left_image_path
                      = dataset_root / "cam0" / "data" / file_name,
                      .right_image_path
                      = dataset_root / "cam1" / "data" / file_name};
    if (fs::exists(frame.left_image_path) && fs::exists(frame.right_image_path))
    {
      frames.push_back(std::move(frame));
    }
  }
  return frames;
}

struct GroundTruthState
{
  double time              = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond world_from_body_orientation
      = Eigen::Quaterniond::Identity();
  Eigen::Vector3d velocity   = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias  = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
};

double LoadGroundTruthStartTime(const fs::path &dataset_root)
{
  const fs::path csv_path
      = dataset_root / "state_groundtruth_estimate0" / "data.csv";
  std::ifstream file(csv_path);
  if (!file)
  {
    throw std::runtime_error("无法打开 " + csv_path.string());
  }
  std::string line;
  while (std::getline(file, line))
  {
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#')
    {
      continue;
    }
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    long long timestamp_ns = 0;
    if (stream >> timestamp_ns)
    {
      return static_cast<double>(timestamp_ns) * 1e-9;
    }
  }
  throw std::runtime_error("groundtruth 文件中没有有效记录: "
                           + csv_path.string());
}

GroundTruthState LoadClosestGroundTruthState(const fs::path &dataset_root,
                                             double target_time)
{
  const fs::path csv_path
      = dataset_root / "state_groundtruth_estimate0" / "data.csv";
  std::ifstream file(csv_path);
  if (!file)
  {
    throw std::runtime_error("无法打开 " + csv_path.string());
  }
  GroundTruthState closest_state;
  double closest_time_gap = std::numeric_limits<double>::max();
  std::string line;
  while (std::getline(file, line))
  {
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#')
    {
      continue;
    }
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    long long timestamp_ns = 0;
    GroundTruthState state;
    Eigen::Quaterniond &orientation = state.world_from_body_orientation;
    if (!(stream >> timestamp_ns >> state.position.x() >> state.position.y()
          >> state.position.z() >> orientation.w() >> orientation.x()
          >> orientation.y() >> orientation.z() >> state.velocity.x()
          >> state.velocity.y() >> state.velocity.z() >> state.gyro_bias.x()
          >> state.gyro_bias.y() >> state.gyro_bias.z() >> state.accel_bias.x()
          >> state.accel_bias.y() >> state.accel_bias.z()))
    {
      continue;
    }
    state.time            = static_cast<double>(timestamp_ns) * 1e-9;
    const double time_gap = std::abs(state.time - target_time);
    if (time_gap < closest_time_gap)
    {
      closest_time_gap = time_gap;
      closest_state    = state;
    }
    if (state.time > target_time + 1.0)
    {
      break; // csv 按时间递增, 已越过目标时刻
    }
  }
  if (closest_time_gap > 0.1)
  {
    throw std::runtime_error("groundtruth 中没有接近首帧时刻 (0.1s 内) 的记录");
  }
  return closest_state;
}

// ==================== 静止检测 (ZUPT 触发条件, 独立于滤波器) ====================
class StationaryDetector
{
public:
  void AddImuSample(const ImuSample &sample)
  {
    window_.push_back(sample);
    while (!window_.empty()
           && window_.front().time < sample.time - kWindowDuration)
    {
      window_.pop_front();
    }
  }

  bool IsStationary() const
  {
    if (window_.size() < kMinSampleCount)
    {
      return false;
    }
    Eigen::Vector3d gyro_mean  = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_mean = Eigen::Vector3d::Zero();
    for (const ImuSample &sample : window_)
    {
      gyro_mean += sample.angular_velocity;
      accel_mean += sample.linear_acceleration;
    }
    const double count = static_cast<double>(window_.size());
    gyro_mean /= count;
    accel_mean /= count;

    double gyro_variance_sum  = 0;
    double accel_variance_sum = 0;
    for (const ImuSample &sample : window_)
    {
      gyro_variance_sum += (sample.angular_velocity - gyro_mean).squaredNorm();
      accel_variance_sum
          += (sample.linear_acceleration - accel_mean).squaredNorm();
    }
    return std::sqrt(gyro_variance_sum / count) < kGyroStdThreshold
           && std::sqrt(accel_variance_sum / count) < kAccelStdThreshold;
  }

private:
  static constexpr double kWindowDuration    = 0.3; // 秒
  static constexpr size_t kMinSampleCount    = 20;
  static constexpr double kGyroStdThreshold  = 0.015; // rad/s, 静止段约为其 1/5
  static constexpr double kAccelStdThreshold = 0.1;   // m/s^2, 飞行振动远超此值

  std::deque<ImuSample> window_;
};

// ============================ 立体矫正 ============================
class StereoRectifier
{
public:
  StereoRectifier(const CameraCalibration &left,
                  const CameraCalibration &right) : image_size_(left.image_size)
  {
    const Sophus::SE3d right_from_left
        = right.body_from_camera.inverse() * left.body_from_camera;
    cv::Mat rotation_left_to_right, translation_left_to_right;
    cv::eigen2cv(right_from_left.rotationMatrix(), rotation_left_to_right);
    cv::eigen2cv(Eigen::Vector3d(right_from_left.translation()),
                 translation_left_to_right);

    cv::Mat left_rect_rotation, right_rect_rotation, disparity_to_depth;
    cv::stereoRectify(left.camera_matrix, left.distortion, right.camera_matrix,
                      right.distortion, image_size_, rotation_left_to_right,
                      translation_left_to_right, left_rect_rotation,
                      right_rect_rotation, left_projection_, right_projection_,
                      disparity_to_depth, cv::CALIB_ZERO_DISPARITY, 0);

    cv::initUndistortRectifyMap(left.camera_matrix, left.distortion,
                                left_rect_rotation, left_projection_,
                                image_size_, CV_32FC1, left_map_x_,
                                left_map_y_);
    cv::initUndistortRectifyMap(right.camera_matrix, right.distortion,
                                right_rect_rotation, right_projection_,
                                image_size_, CV_32FC1, right_map_x_,
                                right_map_y_);

    focal_length_          = left_projection_.at<double>(0, 0);
    left_principal_point_  = {left_projection_.at<double>(0, 2),
                              left_projection_.at<double>(1, 2)};
    right_principal_point_ = {right_projection_.at<double>(0, 2),
                              right_projection_.at<double>(1, 2)};
    baseline_              = -right_projection_.at<double>(0, 3)
                             / right_projection_.at<double>(0, 0);

    Eigen::Matrix3d left_rect_rotation_eigen;
    cv::cv2eigen(left_rect_rotation, left_rect_rotation_eigen);
    body_from_rectified_left_
        = left.body_from_camera
          * Sophus::SE3d(Sophus::SO3d(left_rect_rotation_eigen.transpose()),
                         Eigen::Vector3d::Zero());
  }

  void Rectify(const cv::Mat &raw_left, const cv::Mat &raw_right,
               cv::Mat &rectified_left, cv::Mat &rectified_right) const
  {
    cv::remap(raw_left, rectified_left, left_map_x_, left_map_y_,
              cv::INTER_LINEAR);
    cv::remap(raw_right, rectified_right, right_map_x_, right_map_y_,
              cv::INTER_LINEAR);
  }

  Eigen::Vector2d LeftPixelToNormalized(const cv::Point2f &pixel) const
  {
    return {(pixel.x - left_principal_point_.x()) / focal_length_,
            (pixel.y - left_principal_point_.y()) / focal_length_};
  }

  Eigen::Vector2d RightPixelToNormalized(const cv::Point2f &pixel) const
  {
    return {(pixel.x - right_principal_point_.x()) / focal_length_,
            (pixel.y - right_principal_point_.y()) / focal_length_};
  }

  cv::Point2f LeftNormalizedToPixel(const Eigen::Vector2d &normalized) const
  {
    return {static_cast<float>(normalized.x() * focal_length_
                               + left_principal_point_.x()),
            static_cast<float>(normalized.y() * focal_length_
                               + left_principal_point_.y())};
  }

  double focal_length() const
  {
    return focal_length_;
  }
  double baseline() const
  {
    return baseline_;
  }
  const cv::Size &image_size() const
  {
    return image_size_;
  }
  const Sophus::SE3d &body_from_rectified_left() const
  {
    return body_from_rectified_left_;
  }

private:
  cv::Size image_size_;
  cv::Mat left_projection_, right_projection_;
  cv::Mat left_map_x_, left_map_y_, right_map_x_, right_map_y_;
  double focal_length_                   = 0;
  double baseline_                       = 0;
  Eigen::Vector2d left_principal_point_  = Eigen::Vector2d::Zero();
  Eigen::Vector2d right_principal_point_ = Eigen::Vector2d::Zero();
  Sophus::SE3d body_from_rectified_left_;
};

// ==================== CLAHE 图像增强 (矫正后、提点前) ====================
class ClaheEnhancer
{
public:
  ClaheEnhancer() : clahe_(cv::createCLAHE(kClipLimit, kTileGrid)) {}

  cv::Mat Enhance(const cv::Mat &image) const
  {
    cv::Mat enhanced;
    clahe_->apply(image, enhanced);
    return enhanced;
  }

private:
  static constexpr double kClipLimit
      = 3.0; // 限幅: 增强弱纹理同时抑制噪声过度放大
  inline static const cv::Size kTileGrid{8, 8}; // 局部直方图均衡网格
  cv::Ptr<cv::CLAHE> clahe_;
};

// ==================== FAST 角点提取 + 时序/双目数据关联 ====================
struct StereoObservation
{
  Eigen::Vector2d left_normalized  = Eigen::Vector2d::Zero();
  Eigen::Vector2d right_normalized = Eigen::Vector2d::Zero();
};

class FeatureTracker
{
public:
  struct TrackedFeature
  {
    long feature_id = -1;
    StereoObservation observation;
    cv::Point2f left_pixel;
  };

  explicit FeatureTracker(const StereoRectifier &rectifier) :
    rectifier_(rectifier)
  {
  }

  // current_camera_from_previous_camera: 由外部姿态估计(陀螺积分)给出的两帧间相机旋转
  std::vector<TrackedFeature>
  Track(const cv::Mat &rectified_left, const cv::Mat &rectified_right,
        const Sophus::SO3d &current_camera_from_previous_camera)
  {
    std::vector<cv::Mat> left_pyramid, right_pyramid;
    cv::buildOpticalFlowPyramid(rectified_left, left_pyramid, kLkWindowSize,
                                kPyramidLevels);
    cv::buildOpticalFlowPyramid(rectified_right, right_pyramid, kLkWindowSize,
                                kPyramidLevels);

    std::vector<cv::Point2f> left_points;
    std::vector<long> feature_ids;
    TrackFromPreviousFrame(left_pyramid, current_camera_from_previous_camera,
                           left_points, feature_ids);
    DetectNewFastCorners(rectified_left, left_points, feature_ids);

    std::vector<TrackedFeature> result;
    std::vector<cv::Point2f> kept_points;
    std::vector<long> kept_ids;
    if (!left_points.empty())
    {
      std::vector<cv::Point2f> right_points = left_points;
      std::vector<uchar> status;
      std::vector<float> error;
      cv::calcOpticalFlowPyrLK(
          left_pyramid, right_pyramid, left_points, right_points, status, error,
          kLkWindowSize, kPyramidLevels,
          {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01},
          cv::OPTFLOW_USE_INITIAL_FLOW
      );
      for (size_t i = 0; i < left_points.size(); ++i)
      {
        const double vertical_error
            = std::abs(left_points[i].y - right_points[i].y);
        const double disparity = left_points[i].x - right_points[i].x;
        if (!status[i] || !InsideImage(right_points[i])
            || vertical_error > kMaxEpipolarError || disparity < kMinDisparity)
        {
          continue;
        }
        result.push_back({feature_ids[i],
                          {rectifier_.LeftPixelToNormalized(left_points[i]),
                           rectifier_.RightPixelToNormalized(right_points[i])},
                          left_points[i]});
        kept_points.push_back(left_points[i]);
        kept_ids.push_back(feature_ids[i]);
      }
    }
    previous_left_points_ = std::move(kept_points);
    previous_feature_ids_ = std::move(kept_ids);
    previous_left_pyramid_
        = std::move(left_pyramid); // 复用: 下一帧的"上一帧金字塔"
    return result;
  }

  // 删除滤波器已消费的特征: 停止跟踪旧 id, 其占据的网格下一帧可重新提取为新特征
  void DropFeatures(std::span<const long> feature_ids)
  {
    if (feature_ids.empty())
    {
      return;
    }
    const std::set<long> ids_to_drop(feature_ids.begin(), feature_ids.end());
    std::vector<cv::Point2f> kept_points;
    std::vector<long> kept_ids;
    for (size_t i = 0; i < previous_feature_ids_.size(); ++i)
    {
      if (ids_to_drop.contains(previous_feature_ids_[i]))
      {
        continue;
      }
      kept_points.push_back(previous_left_points_[i]);
      kept_ids.push_back(previous_feature_ids_[i]);
    }
    previous_left_points_ = std::move(kept_points);
    previous_feature_ids_ = std::move(kept_ids);
  }

private:
  static constexpr size_t kMaxFeatureCount    = 200;
  static constexpr int kFastThreshold         = 20;
  static constexpr double kMinFeatureDistance = 25.0;
  static constexpr double kMaxEpipolarError   = 3.0;
  static constexpr double kMinDisparity       = 0.5;
  static constexpr int kImageBorder           = 5;
  static constexpr int kPyramidLevels         = 3;
  inline static const cv::Size kLkWindowSize{21, 21};

  bool InsideImage(const cv::Point2f &point) const
  {
    const cv::Size &image_size = rectifier_.image_size();
    return point.x >= kImageBorder && point.y >= kImageBorder
           && point.x < image_size.width - kImageBorder
           && point.y < image_size.height - kImageBorder;
  }

  // 纯旋转预测: 把上一帧归一化方向旋到当前帧, 作为光流初值 (快速转动时大幅提高存活率)
  cv::Point2f PredictPixelWithRotation(
      const cv::Point2f &pixel,
      const Sophus::SO3d &current_camera_from_previous_camera
  ) const
  {
    const Eigen::Vector2d normalized = rectifier_.LeftPixelToNormalized(pixel);
    const Eigen::Vector3d rotated_direction
        = current_camera_from_previous_camera
          * Eigen::Vector3d(normalized.x(), normalized.y(), 1.0);
    if (rotated_direction.z() < 0.1)
    {
      return pixel;
    }
    return rectifier_.LeftNormalizedToPixel(
        {rotated_direction.x() / rotated_direction.z(),
         rotated_direction.y() / rotated_direction.z()}
    );
  }

  void TrackFromPreviousFrame(
      const std::vector<cv::Mat> &current_left_pyramid,
      const Sophus::SO3d &current_camera_from_previous_camera,
      std::vector<cv::Point2f> &left_points, std::vector<long> &feature_ids
  )
  {
    if (previous_left_pyramid_.empty() || previous_left_points_.empty())
    {
      return;
    }
    std::vector<cv::Point2f> current_points(previous_left_points_.size());
    for (size_t i = 0; i < previous_left_points_.size(); ++i)
    {
      current_points[i]
          = PredictPixelWithRotation(previous_left_points_[i],
                                     current_camera_from_previous_camera);
    }
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(
        previous_left_pyramid_, current_left_pyramid, previous_left_points_,
        current_points, status, error, kLkWindowSize, kPyramidLevels,
        {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01},
        cv::OPTFLOW_USE_INITIAL_FLOW
    );

    std::vector<cv::Point2f> matched_previous, matched_current;
    std::vector<long> matched_ids;
    for (size_t i = 0; i < previous_left_points_.size(); ++i)
    {
      if (!status[i] || !InsideImage(current_points[i]))
      {
        continue;
      }
      matched_previous.push_back(previous_left_points_[i]);
      matched_current.push_back(current_points[i]);
      matched_ids.push_back(previous_feature_ids_[i]);
    }

    std::vector<uchar> inlier_mask(matched_current.size(), 1);
    if (matched_current.size() >= 8)
    {
      cv::findFundamentalMat(matched_previous, matched_current, cv::FM_RANSAC,
                             1.0, 0.99, inlier_mask);
    }
    for (size_t i = 0; i < matched_current.size(); ++i)
    {
      if (!inlier_mask[i])
      {
        continue;
      }
      left_points.push_back(matched_current[i]);
      feature_ids.push_back(matched_ids[i]);
    }
  }

  void DetectNewFastCorners(const cv::Mat &image,
                            std::vector<cv::Point2f> &left_points,
                            std::vector<long> &feature_ids)
  {
    if (left_points.size() >= kMaxFeatureCount)
    {
      return;
    }
    cv::Mat occupancy_mask(image.size(), CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f &point : left_points)
    {
      cv::circle(occupancy_mask, point, static_cast<int>(kMinFeatureDistance),
                 cv::Scalar(0), cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, kFastThreshold, true);
    std::ranges::sort(keypoints, [](const auto &a, const auto &b)
                      { return a.response > b.response; });

    for (const cv::KeyPoint &keypoint : keypoints)
    {
      if (left_points.size() >= kMaxFeatureCount)
      {
        break;
      }
      const cv::Point2f point = keypoint.pt;
      if (!InsideImage(point)
          || occupancy_mask.at<uchar>(cv::Point(point)) == 0)
      {
        continue;
      }
      cv::circle(occupancy_mask, point, static_cast<int>(kMinFeatureDistance),
                 cv::Scalar(0), cv::FILLED);
      left_points.push_back(point);
      feature_ids.push_back(next_feature_id_++);
    }
  }

  const StereoRectifier &rectifier_;
  std::vector<cv::Mat> previous_left_pyramid_;
  std::vector<cv::Point2f> previous_left_points_;
  std::vector<long> previous_feature_ids_;
  long next_feature_id_ = 0;
};

// ==================== 特征三角化 (Ceres 双目重投影) ====================
struct StereoReprojectionCost
{
  StereoReprojectionCost(const Sophus::SE3d &camera_from_world,
                         const StereoObservation &observation,
                         double stereo_baseline) :
    rotation(camera_from_world.rotationMatrix()),
    translation(camera_from_world.translation()), measurement(observation),
    baseline(stereo_baseline)
  {
  }

  template <typename T>
  bool operator()(const T *const world_point, T *residual) const
  {
    const Eigen::Matrix<T, 3, 1> point(world_point[0], world_point[1],
                                       world_point[2]);
    const Eigen::Matrix<T, 3, 1> point_in_camera
        = rotation.cast<T>() * point + translation.cast<T>();
    if (point_in_camera.z() < T(1e-3))
    {
      return false;
    }
    residual[0] = point_in_camera.x() / point_in_camera.z()
                  - T(measurement.left_normalized.x());
    residual[1] = point_in_camera.y() / point_in_camera.z()
                  - T(measurement.left_normalized.y());
    residual[2] = (point_in_camera.x() - T(baseline)) / point_in_camera.z()
                  - T(measurement.right_normalized.x());
    residual[3] = point_in_camera.y() / point_in_camera.z()
                  - T(measurement.right_normalized.y());
    return true;
  }

  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  StereoObservation measurement;
  double baseline;
};

// ============================ MSCKF 滤波器 ============================
struct FeatureTrack
{
  long feature_id = -1;
  std::map<long, StereoObservation> observations_by_frame;
};

struct CameraClone
{
  long frame_id = -1;
  Sophus::SO3d world_from_camera_rotation;
  Eigen::Vector3d position_in_world = Eigen::Vector3d::Zero();
  Sophus::SO3d null_rotation; // OC: 增广时刻的首次估计, EKF 更新不改动
  Eigen::Vector3d null_position = Eigen::Vector3d::Zero();
};

struct SlamFeature
{
  long feature_id                   = -1;
  Eigen::Vector3d position_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d null_position
      = Eigen::Vector3d::Zero(); // OC/FEJ: 初始化时刻的首次估计
};

class Msckf
{
public:
  static constexpr int kImuErrorDim
      = 19; // [姿态3 位置3 速度3 陀螺零偏3 加计零偏3 时间偏移1 重力3]
  static constexpr int kTimeOffsetIndex              = 15;
  static constexpr int kGravityIndex                 = 16;
  static constexpr int kCloneErrorDim                = 6; // [姿态3 位置3]
  static constexpr size_t kMaxCloneCount             = 11;
  static constexpr size_t kMinTrackLength            = 3;
  static constexpr int kMaxUpdateRows                = 600;
  static constexpr double kInitialTimeOffsetVariance = 2.5e-5; // (5 ms)^2
  static constexpr double kZuptVelocitySigma   = 0.02; // m/s, 零速伪量测噪声
  static constexpr size_t kMaxSlamFeatureCount = 20;
  static constexpr size_t kMinSlamTrackLength = 8; // 升级为 SLAM 特征的最短轨迹
  static constexpr double kRedundantRotationThreshold
      = 0.2618; // rad, 冗余克隆判定 (15°)
  static constexpr double kRedundantTranslationThreshold = 0.4; // m
  static constexpr double kMonocularRotationSigma
      = 0.005; // rad, 单目旋转量测噪声 (高精度)
  static constexpr double kMonocularDirectionSigma
      = 0.1; // rad, 平移方向切平面噪声 (低精度)
  static constexpr double kMinBaselineForDirection
      = 0.01; // m, 基线过短时方向量测退化

  Msckf(const Sophus::SE3d &body_from_camera, double baseline,
        double focal_length, const ImuNoiseParameters &imu_noise,
        double initial_time_offset) :
    body_from_camera_(body_from_camera), baseline_(baseline),
    pixel_noise_normalized_(1.5 / focal_length), imu_noise_(imu_noise),
    time_offset_(initial_time_offset)
  {
  }

  void InitializeFromStaticImu(std::span<const ImuSample> samples)
  {
    if (samples.empty())
    {
      throw std::runtime_error("初始化需要静止 IMU 样本");
    }
    Eigen::Vector3d gyro_mean  = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_mean = Eigen::Vector3d::Zero();
    for (const ImuSample &sample : samples)
    {
      gyro_mean += sample.angular_velocity;
      accel_mean += sample.linear_acceleration;
    }
    gyro_mean /= static_cast<double>(samples.size());
    accel_mean /= static_cast<double>(samples.size());

    world_from_imu_rotation_ = Sophus::SO3d(
        Eigen::Quaterniond::FromTwoVectors(accel_mean, Eigen::Vector3d::UnitZ())
    );
    gyro_bias_ = gyro_mean;
    gravity_in_world_
        = -(world_from_imu_rotation_ * accel_mean); // 静止时比力 = -g
    imu_time_     = samples.back().time;
    last_imu_     = samples.back();
    has_last_imu_ = true;

    covariance_ = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
    covariance_.diagonal() << 1e-4, 1e-4, 1e-3, 1e-8, 1e-8, 1e-8, 1e-2, 1e-2,
        1e-2, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3, kInitialTimeOffsetVariance,
        1e-2, 1e-2, 1e-2;
    SaveNullLinearizationPoint();
  }

  void InitializeFromGroundTruth(const GroundTruthState &state)
  {
    world_from_imu_rotation_
        = Sophus::SO3d(state.world_from_body_orientation.normalized());
    imu_position_     = state.position;
    imu_velocity_     = state.velocity;
    gyro_bias_        = state.gyro_bias;
    accel_bias_       = state.accel_bias;
    gravity_in_world_ = Eigen::Vector3d(0, 0, -kGravity); // 真值世界系重力沿 -z
    imu_time_         = state.time;
    has_last_imu_     = false; // 之后遇到的第一个 IMU 样本仅记录为积分起点

    covariance_ = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
    covariance_.diagonal() << 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4,
        1e-4, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, kInitialTimeOffsetVariance,
        1e-4, 1e-4, 1e-4;
    SaveNullLinearizationPoint();
  }

  void PropagateWithImu(const ImuSample &sample)
  {
    if (!has_last_imu_)
    {
      last_imu_     = sample;
      imu_time_     = sample.time;
      has_last_imu_ = true;
      return;
    }
    const double dt = sample.time - last_imu_.time;
    if (dt <= 0 || dt > 0.1)
    {
      last_imu_ = sample;
      imu_time_ = sample.time;
      return;
    }

    const Eigen::Vector3d gyro
        = 0.5 * (last_imu_.angular_velocity + sample.angular_velocity)
          - gyro_bias_;
    const Eigen::Vector3d accel
        = 0.5 * (last_imu_.linear_acceleration + sample.linear_acceleration)
          - accel_bias_;
    const Eigen::Matrix3d rotation_matrix = world_from_imu_rotation_.matrix();
    const Eigen::Vector3d accel_in_world
        = rotation_matrix * accel + gravity_in_world_;

    world_from_imu_rotation_
        = world_from_imu_rotation_ * Sophus::SO3d::exp(gyro * dt);
    imu_position_ += imu_velocity_ * dt + 0.5 * accel_in_world * dt * dt;
    imu_velocity_ += accel_in_world * dt;
    imu_time_ = sample.time;
    last_imu_ = sample;

    Eigen::Matrix<double, kImuErrorDim, kImuErrorDim> transition
        = Eigen::Matrix<double, kImuErrorDim, kImuErrorDim>::Identity();
    transition.block<3, 3>(0, 0) = Sophus::SO3d::exp(-gyro * dt).matrix();
    transition.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt;
    transition.block<3, 3>(3, 0)
        = -0.5 * rotation_matrix * Sophus::SO3d::hat(accel) * dt * dt;
    transition.block<3, 3>(3, 6)  = Eigen::Matrix3d::Identity() * dt;
    transition.block<3, 3>(3, 12) = -0.5 * rotation_matrix * dt * dt;
    transition.block<3, 3>(3, kGravityIndex)
        = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
    transition.block<3, 3>(6, 0)
        = -rotation_matrix * Sophus::SO3d::hat(accel) * dt;
    transition.block<3, 3>(6, 12)            = -rotation_matrix * dt;
    transition.block<3, 3>(6, kGravityIndex) = Eigen::Matrix3d::Identity() * dt;

    // OC 一致性修正 (Hesch et al.): 强制 Φ·N_k = N_{k+1}, N 为不可观子空间
    // 全局平移列 Φ 天然满足; 绕重力偏航列需修正姿态/速度/位置对 δθ 的三个块
    transition.block<3, 3>(0, 0)
        = world_from_imu_rotation_.inverse().matrix() * null_rotation_.matrix();

    const Eigen::Vector3d yaw_direction_in_body
        = null_rotation_.inverse().matrix() * null_gravity_;
    const Eigen::RowVector3d yaw_projector
        = yaw_direction_in_body.transpose()
          / yaw_direction_in_body.squaredNorm();

    const Eigen::Matrix3d velocity_block = transition.block<3, 3>(6, 0);
    const Eigen::Vector3d velocity_constraint
        = Sophus::SO3d::hat(null_velocity_ - imu_velocity_) * null_gravity_;
    transition.block<3, 3>(6, 0)
        = velocity_block
          - (velocity_block * yaw_direction_in_body - velocity_constraint)
                * yaw_projector;

    const Eigen::Matrix3d position_block = transition.block<3, 3>(3, 0);
    const Eigen::Vector3d position_constraint
        = Sophus::SO3d::hat(dt * null_velocity_ + null_position_
                            - imu_position_)
          * null_gravity_;
    transition.block<3, 3>(3, 0)
        = position_block
          - (position_block * yaw_direction_in_body - position_constraint)
                * yaw_projector;

    Eigen::Matrix<double, kImuErrorDim, 12> noise_jacobian
        = Eigen::Matrix<double, kImuErrorDim, 12>::Zero();
    noise_jacobian.block<3, 3>(0, 0)  = -Eigen::Matrix3d::Identity();
    noise_jacobian.block<3, 3>(6, 3)  = rotation_matrix;
    noise_jacobian.block<3, 3>(9, 6)  = Eigen::Matrix3d::Identity();
    noise_jacobian.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 12, 12> continuous_noise
        = Eigen::Matrix<double, 12, 12>::Zero();
    continuous_noise.diagonal().segment<3>(0).setConstant(
        imu_noise_.gyro_noise_density * imu_noise_.gyro_noise_density
    );
    continuous_noise.diagonal().segment<3>(3).setConstant(
        imu_noise_.accel_noise_density * imu_noise_.accel_noise_density
    );
    continuous_noise.diagonal().segment<3>(6).setConstant(
        imu_noise_.gyro_random_walk * imu_noise_.gyro_random_walk
    );
    continuous_noise.diagonal().segment<3>(9).setConstant(
        imu_noise_.accel_random_walk * imu_noise_.accel_random_walk
    );

    const Eigen::Matrix<double, kImuErrorDim, kImuErrorDim> discrete_noise
        = noise_jacobian * continuous_noise * noise_jacobian.transpose() * dt;

    covariance_.topLeftCorner<kImuErrorDim, kImuErrorDim>()
        = transition * covariance_.topLeftCorner<kImuErrorDim, kImuErrorDim>()
              * transition.transpose()
          + discrete_noise;
    const int clone_dim = static_cast<int>(covariance_.rows()) - kImuErrorDim;
    if (clone_dim > 0)
    {
      covariance_.topRightCorner(kImuErrorDim, clone_dim)
          = transition * covariance_.topRightCorner(kImuErrorDim, clone_dim);
      covariance_.bottomLeftCorner(clone_dim, kImuErrorDim)
          = covariance_.topRightCorner(kImuErrorDim, clone_dim).transpose();
    }
    Symmetrize();
    SaveNullLinearizationPoint();
  }

  bool ApplyZeroVelocityUpdate()
  {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, StateDim());
    jacobian.block<3, 3>(0, 6)
        = Eigen::Matrix3d::Identity();            // 量测模型 h(x) = 速度
    Eigen::VectorXd residual    = -imu_velocity_; // 零速: z = 0
    const double noise_variance = kZuptVelocitySigma * kZuptVelocitySigma;
    if (!PassesChiSquareGate(jacobian, residual, noise_variance))
    {
      return false;
    }
    ApplyEkfUpdate(std::move(jacobian), std::move(residual), noise_variance);
    return true;
  }

  // 融合外部单目算法输出的帧间相对运动 (在 ProcessFrame 之后调用)。约定 (矫正后左相机系):
  //   rotation_vector:       上一帧到当前帧的相对旋转轴角, R_上帧_from_当帧 = Exp(rVec)
  //   translation_direction: 当前光心相对上一帧光心的平移方向, 在上一帧相机系中表达 (无尺度)
  // 旋转做 3 维流形残差 (高权重), 平移只约束量测方向切平面上的 2 维分量 (低权重, 尺度自然
  // 不受约束); 两者按顺序 EKF 更新。相对位姿量测对全局平移/绕重力偏航天然满足 H·N = 0,
  // 无需 OC 投影。上一帧克隆若已被冗余边缘化 (悬停时可能) 则跳过, 返回 false。
  bool MonocularUpdate(const Eigen::Vector3d &rotation_vector,
                       const Eigen::Vector3d &translation_direction)
  {
    if (clones_.size() < 2)
    {
      return false;
    }
    const int current_index           = static_cast<int>(clones_.size()) - 1;
    const int previous_index          = current_index - 1;
    const CameraClone &current_clone  = clones_[current_index];
    const CameraClone &previous_clone = clones_[previous_index];
    if (current_clone.frame_id != previous_clone.frame_id + 1)
    {
      return false;
    }

    // --- 旋转: r = Log(R̂_相对⁻¹·Exp(rVec)), H_θ上 = -R̂_相对⁻¹, H_θ当 = I ---
    const Sophus::SO3d predicted_relative_rotation
        = previous_clone.world_from_camera_rotation.inverse()
          * current_clone.world_from_camera_rotation;
    Eigen::MatrixXd rotation_jacobian = Eigen::MatrixXd::Zero(3, StateDim());
    rotation_jacobian.block<3, 3>(0, CloneStateIndex(previous_index))
        = -predicted_relative_rotation.inverse().matrix();
    rotation_jacobian.block<3, 3>(0, CloneStateIndex(current_index))
        = Eigen::Matrix3d::Identity();
    Eigen::VectorXd rotation_residual = (predicted_relative_rotation.inverse()
                                         * Sophus::SO3d::exp(rotation_vector))
                                            .log();

    const double rotation_noise_variance
        = kMonocularRotationSigma * kMonocularRotationSigma;
    bool rotation_applied = false;
    if (PassesChiSquareGate(rotation_jacobian, rotation_residual,
                            rotation_noise_variance))
    {
      ApplyEkfUpdate(std::move(rotation_jacobian), std::move(rotation_residual),
                     rotation_noise_variance);
      rotation_applied = true;
    }

    // --- 平移方向: 旋转更新后的最新状态上重新线性化 (克隆引用随更新自动生效) ---
    const double direction_norm = translation_direction.norm();
    if (direction_norm < 1e-6)
    {
      return rotation_applied;
    }
    const Eigen::Vector3d measured_direction
        = translation_direction / direction_norm;

    const Eigen::Matrix3d previous_camera_from_world
        = previous_clone.world_from_camera_rotation.inverse().matrix();
    const Eigen::Vector3d baseline_in_previous_camera
        = previous_camera_from_world
          * (current_clone.position_in_world
             - previous_clone.position_in_world);
    const double baseline_norm = baseline_in_previous_camera.norm();
    if (baseline_norm < kMinBaselineForDirection)
    {
      return rotation_applied; // 悬停: 方向退化
    }
    const Eigen::Vector3d predicted_direction
        = baseline_in_previous_camera / baseline_norm;

    const Eigen::Vector3d helper = std::abs(measured_direction.z()) < 0.9
                                       ? Eigen::Vector3d::UnitZ()
                                       : Eigen::Vector3d::UnitX();
    const Eigen::Vector3d tangent_basis_1
        = measured_direction.cross(helper).normalized();
    const Eigen::Vector3d tangent_basis_2
        = measured_direction.cross(tangent_basis_1);
    Eigen::Matrix<double, 2, 3> tangent_projector;
    tangent_projector.row(0) = tangent_basis_1.transpose();
    tangent_projector.row(1) = tangent_basis_2.transpose();

    const Eigen::Matrix3d normalization_jacobian
        = (Eigen::Matrix3d::Identity()
           - predicted_direction * predicted_direction.transpose())
          / baseline_norm;
    const Eigen::Matrix<double, 2, 3> direction_jacobian
        = tangent_projector * normalization_jacobian;

    Eigen::MatrixXd translation_jacobian = Eigen::MatrixXd::Zero(2, StateDim());
    translation_jacobian.block<2, 3>(0, CloneStateIndex(previous_index))
        = direction_jacobian * Sophus::SO3d::hat(baseline_in_previous_camera);
    translation_jacobian.block<2, 3>(0, CloneStateIndex(previous_index) + 3)
        = -direction_jacobian * previous_camera_from_world;
    translation_jacobian.block<2, 3>(0, CloneStateIndex(current_index) + 3)
        = direction_jacobian * previous_camera_from_world;
    Eigen::VectorXd translation_residual
        = -(tangent_projector * predicted_direction);

    const double direction_noise_variance
        = kMonocularDirectionSigma * kMonocularDirectionSigma;
    if (!PassesChiSquareGate(translation_jacobian, translation_residual,
                             direction_noise_variance))
    {
      return rotation_applied;
    }
    ApplyEkfUpdate(std::move(translation_jacobian),
                   std::move(translation_residual), direction_noise_variance);
    return true;
  }

  // 返回需要前端删除的特征 id (量测门限判为外点的 SLAM 特征)
  std::vector<long>
  ProcessFrame(long frame_id,
               const std::vector<FeatureTracker::TrackedFeature> &tracked)
  {
    AugmentCameraClone(frame_id);
    std::vector<long> dropped_feature_ids;

    std::set<long> visible_ids;
    std::vector<std::pair<int, StereoObservation>> slam_observations;
    for (const auto &feature : tracked)
    {
      visible_ids.insert(feature.feature_id);
      const int slam_index = FindSlamFeatureIndex(feature.feature_id);
      if (slam_index >= 0)
      {
        slam_observations.emplace_back(slam_index, feature.observation);
        continue;
      }
      FeatureTrack &track = active_tracks_[feature.feature_id];
      track.feature_id    = feature.feature_id;
      track.observations_by_frame[frame_id] = feature.observation;
    }

    UpdateWithSlamObservations(frame_id, slam_observations,
                               dropped_feature_ids);
    RemoveLostSlamFeatures(visible_ids);

    std::vector<FeatureTrack> finished_tracks;
    for (auto it = active_tracks_.begin(); it != active_tracks_.end();)
    {
      if (!visible_ids.contains(it->first))
      {
        finished_tracks.push_back(std::move(it->second));
        it = active_tracks_.erase(it);
      }
      else
      {
        ++it;
      }
    }
    UpdateWithTracks(finished_tracks);

    if (clones_.size() > kMaxCloneCount)
    {
      PruneClonesAndAbsorbObservations();
    }
    return dropped_feature_ids;
  }

  const Sophus::SO3d &world_from_imu_rotation() const
  {
    return world_from_imu_rotation_;
  }
  const Eigen::Vector3d &imu_position() const
  {
    return imu_position_;
  }
  const Eigen::Vector3d &imu_velocity() const
  {
    return imu_velocity_;
  }
  double time_offset_camera_to_imu() const
  {
    return time_offset_;
  }
  const Eigen::Vector3d &gravity_in_world() const
  {
    return gravity_in_world_;
  }
  size_t clone_count() const
  {
    return clones_.size();
  }
  size_t slam_feature_count() const
  {
    return slam_features_.size();
  }

private:
  int StateDim() const
  {
    return kImuErrorDim + 3 * static_cast<int>(slam_features_.size())
           + kCloneErrorDim * static_cast<int>(clones_.size());
  }
  int SlamFeatureStateIndex(int feature_index) const
  {
    return kImuErrorDim + 3 * feature_index;
  }
  int CloneStateIndex(int clone_index) const
  {
    return kImuErrorDim + 3 * static_cast<int>(slam_features_.size())
           + kCloneErrorDim * clone_index;
  }

  void Symmetrize()
  {
    covariance_ = ((covariance_ + covariance_.transpose()) * 0.5).eval();
  }

  int FindCloneIndex(long frame_id) const
  {
    for (size_t i = 0; i < clones_.size(); ++i)
    {
      if (clones_[i].frame_id == frame_id)
      {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  int FindSlamFeatureIndex(long feature_id) const
  {
    for (size_t i = 0; i < slam_features_.size(); ++i)
    {
      if (slam_features_[i].feature_id == feature_id)
      {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  void AugmentCameraClone(long frame_id)
  {
    const Sophus::SO3d world_from_camera_rotation
        = world_from_imu_rotation_ * body_from_camera_.so3();
    const Eigen::Vector3d camera_position
        = imu_position_
          + world_from_imu_rotation_ * body_from_camera_.translation();

    const int old_dim = StateDim();
    Eigen::MatrixXd clone_jacobian
        = Eigen::MatrixXd::Zero(kCloneErrorDim, old_dim);
    clone_jacobian.block<3, 3>(0, 0)
        = body_from_camera_.so3().inverse().matrix();
    clone_jacobian.block<3, 3>(3, 0)
        = -world_from_imu_rotation_.matrix()
          * Sophus::SO3d::hat(body_from_camera_.translation());
    clone_jacobian.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    // t_d 偏差使克隆位姿沿当前运动方向平移: dθ_c/dt_d = R_ci·ω, dp_c/dt_d = 相机质心的世界系速度
    const Eigen::Vector3d angular_velocity
        = last_imu_.angular_velocity - gyro_bias_;
    clone_jacobian.block<3, 1>(0, kTimeOffsetIndex)
        = body_from_camera_.so3().inverse().matrix() * angular_velocity;
    clone_jacobian.block<3, 1>(3, kTimeOffsetIndex)
        = imu_velocity_
          + world_from_imu_rotation_.matrix()
                * angular_velocity.cross(body_from_camera_.translation());

    Eigen::MatrixXd augmented(old_dim + kCloneErrorDim,
                              old_dim + kCloneErrorDim);
    augmented.topLeftCorner(old_dim, old_dim) = covariance_;
    augmented.bottomLeftCorner(kCloneErrorDim, old_dim)
        = clone_jacobian * covariance_;
    augmented.topRightCorner(old_dim, kCloneErrorDim)
        = augmented.bottomLeftCorner(kCloneErrorDim, old_dim).transpose();
    augmented.bottomRightCorner(kCloneErrorDim, kCloneErrorDim)
        = clone_jacobian * covariance_ * clone_jacobian.transpose();
    covariance_ = std::move(augmented);
    Symmetrize();

    clones_.push_back({frame_id, world_from_camera_rotation, camera_position,
                       world_from_camera_rotation, camera_position});
  }

  std::optional<Eigen::Vector3d>
  TriangulateFeature(const FeatureTrack &track) const
  {
    std::vector<std::pair<Sophus::SE3d, StereoObservation>> observations;
    for (const auto &[frame_id, observation] : track.observations_by_frame)
    {
      const int clone_index = FindCloneIndex(frame_id);
      if (clone_index < 0)
      {
        continue;
      }
      const CameraClone &clone = clones_[clone_index];
      const Sophus::SE3d world_from_camera(clone.world_from_camera_rotation,
                                           clone.position_in_world);
      observations.emplace_back(world_from_camera.inverse(), observation);
    }
    if (observations.size() < kMinTrackLength)
    {
      return std::nullopt;
    }

    const auto &[first_camera_from_world, first_observation]
        = observations.front();
    const double disparity = first_observation.left_normalized.x()
                             - first_observation.right_normalized.x();
    if (disparity < 1e-4)
    {
      return std::nullopt;
    }
    const double depth = std::clamp(baseline_ / disparity, 0.2, 50.0);
    Eigen::Vector3d world_point
        = first_camera_from_world.inverse()
          * Eigen::Vector3d(first_observation.left_normalized.x() * depth,
                            first_observation.left_normalized.y() * depth,
                            depth);

    ceres::Problem problem;
    for (const auto &[camera_from_world, observation] : observations)
    {
      auto *cost
          = new ceres::AutoDiffCostFunction<StereoReprojectionCost, 4, 3>(
              new StereoReprojectionCost(camera_from_world, observation,
                                         baseline_)
          );
      problem.AddResidualBlock(cost, new ceres::HuberLoss(0.01),
                               world_point.data());
    }
    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.max_num_iterations           = 20;
    options.logging_type                 = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
      return std::nullopt;
    }

    const double mean_squared_error
        = 2.0 * summary.final_cost
          / static_cast<double>(4 * observations.size());
    if (std::sqrt(mean_squared_error) > 10.0 * pixel_noise_normalized_)
    {
      return std::nullopt;
    }

    for (const auto &[camera_from_world, observation] : observations)
    {
      const Eigen::Vector3d point_in_camera = camera_from_world * world_point;
      if (point_in_camera.z() < 0.1 || point_in_camera.z() > 60.0)
      {
        return std::nullopt;
      }
    }
    return world_point;
  }

  // 构建轨迹的原始量测雅可比 (含 OC 投影); frame_filter 非空时仅使用其中的帧 (边缘化吸收用)
  bool BuildTrackJacobians(const FeatureTrack &track,
                           const Eigen::Vector3d &world_point,
                           const std::set<long> *frame_filter,
                           Eigen::MatrixXd &state_jacobian,
                           Eigen::MatrixXd &point_jacobian,
                           Eigen::VectorXd &residual) const
  {
    const int max_rows
        = 4 * static_cast<int>(track.observations_by_frame.size());
    state_jacobian = Eigen::MatrixXd::Zero(max_rows, StateDim());
    point_jacobian = Eigen::MatrixXd::Zero(max_rows, 3);
    residual       = Eigen::VectorXd::Zero(max_rows);

    int row = 0;
    for (const auto &[frame_id, observation] : track.observations_by_frame)
    {
      if (frame_filter != nullptr && !frame_filter->contains(frame_id))
      {
        continue;
      }
      const int clone_index = FindCloneIndex(frame_id);
      if (clone_index < 0)
      {
        continue;
      }
      const CameraClone &clone = clones_[clone_index];
      const Eigen::Matrix3d camera_from_world
          = clone.world_from_camera_rotation.inverse().matrix();
      const Eigen::Vector3d point_in_camera
          = camera_from_world * (world_point - clone.position_in_world);
      const double x = point_in_camera.x();
      const double y = point_in_camera.y();
      const double z = point_in_camera.z();
      if (z < 0.05)
      {
        return false;
      }

      Eigen::Matrix<double, 4, 3> projection_jacobian;
      projection_jacobian << 1 / z, 0, -x / (z * z), 0, 1 / z, -y / (z * z),
          1 / z, 0, -(x - baseline_) / (z * z), 0, 1 / z, -y / (z * z);

      const int column = CloneStateIndex(clone_index);
      Eigen::Matrix<double, 4, 6> clone_jacobian_block;
      clone_jacobian_block.leftCols<3>()
          = projection_jacobian * Sophus::SO3d::hat(point_in_camera);
      clone_jacobian_block.rightCols<3>()
          = -projection_jacobian * camera_from_world;

      // OC 一致性修正: 把绕重力偏航方向从量测雅可比中投影掉 (H·N = 0),
      // 再由 H_点 = -H_位置 重建特征雅可比, 同时保证全局平移方向的约束
      Eigen::Matrix<double, 6, 1> yaw_direction;
      yaw_direction.head<3>()
          = clone.null_rotation.inverse().matrix() * gravity_in_world_;
      yaw_direction.tail<3>()
          = Sophus::SO3d::hat(world_point - clone.null_position)
            * gravity_in_world_;
      clone_jacobian_block -= clone_jacobian_block * yaw_direction
                              * yaw_direction.transpose()
                              / yaw_direction.squaredNorm();

      state_jacobian.block<4, 6>(row, column) = clone_jacobian_block;
      point_jacobian.block<4, 3>(row, 0) = -clone_jacobian_block.rightCols<3>();
      residual.segment<4>(row) << observation.left_normalized.x() - x / z,
          observation.left_normalized.y() - y / z,
          observation.right_normalized.x() - (x - baseline_) / z,
          observation.right_normalized.y() - y / z;
      row += 4;
    }
    if (row == 0)
    {
      return false;
    }
    state_jacobian.conservativeResize(row, Eigen::NoChange);
    point_jacobian.conservativeResize(row, Eigen::NoChange);
    residual.conservativeResize(row);
    return true;
  }

  bool ComputeProjectedFeatureJacobian(const FeatureTrack &track,
                                       const Eigen::Vector3d &world_point,
                                       Eigen::MatrixXd &projected_jacobian,
                                       Eigen::VectorXd &projected_residual,
                                       const std::set<long> *frame_filter
                                       = nullptr) const
  {
    Eigen::MatrixXd state_jacobian, point_jacobian;
    Eigen::VectorXd residual;
    if (!BuildTrackJacobians(track, world_point, frame_filter, state_jacobian,
                             point_jacobian, residual))
    {
      return false;
    }
    const int rows = static_cast<int>(residual.size());
    if (rows <= 3)
    {
      return false;
    }

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(point_jacobian);
    const Eigen::MatrixXd q_full         = qr.householderQ();
    const Eigen::MatrixXd left_nullspace = q_full.rightCols(rows - 3);
    projected_jacobian = left_nullspace.transpose() * state_jacobian;
    projected_residual = left_nullspace.transpose() * residual;
    return true;
  }

  static double ChiSquare95(int degrees_of_freedom)
  {
    const double k = static_cast<double>(degrees_of_freedom);
    const double w
        = 1.0 - 2.0 / (9.0 * k) + 1.6449 * std::sqrt(2.0 / (9.0 * k));
    return k * w * w * w;
  }

  bool PassesChiSquareGate(const Eigen::MatrixXd &jacobian,
                           const Eigen::VectorXd &residual,
                           double noise_variance) const
  {
    Eigen::MatrixXd innovation_covariance
        = jacobian * covariance_ * jacobian.transpose();
    innovation_covariance.diagonal().array() += noise_variance;
    const double mahalanobis_distance
        = residual.dot(innovation_covariance.ldlt().solve(residual));
    return mahalanobis_distance
           < ChiSquare95(static_cast<int>(residual.size()));
  }

  void UpdateWithTracks(const std::vector<FeatureTrack> &finished_tracks)
  {
    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    std::vector<Eigen::MatrixXd> jacobian_blocks;
    std::vector<Eigen::VectorXd> residual_blocks;
    int total_rows = 0;
    for (const FeatureTrack &track : finished_tracks)
    {
      if (track.observations_by_frame.size() < kMinTrackLength)
      {
        continue;
      }
      const auto world_point = TriangulateFeature(track);
      if (!world_point)
      {
        continue;
      }
      Eigen::MatrixXd projected_jacobian;
      Eigen::VectorXd projected_residual;
      if (!ComputeProjectedFeatureJacobian(track, *world_point,
                                           projected_jacobian,
                                           projected_residual))
      {
        continue;
      }
      if (!PassesChiSquareGate(projected_jacobian, projected_residual,
                               vision_noise_variance))
      {
        continue;
      }
      total_rows += static_cast<int>(projected_jacobian.rows());
      jacobian_blocks.push_back(std::move(projected_jacobian));
      residual_blocks.push_back(std::move(projected_residual));
      if (total_rows >= kMaxUpdateRows)
      {
        break;
      }
    }
    if (jacobian_blocks.empty())
    {
      return;
    }

    Eigen::MatrixXd stacked_jacobian(total_rows, StateDim());
    Eigen::VectorXd stacked_residual(total_rows);
    int row = 0;
    for (size_t i = 0; i < jacobian_blocks.size(); ++i)
    {
      stacked_jacobian.middleRows(row, jacobian_blocks[i].rows())
          = jacobian_blocks[i];
      stacked_residual.segment(row, residual_blocks[i].size())
          = residual_blocks[i];
      row += static_cast<int>(jacobian_blocks[i].rows());
    }
    ApplyEkfUpdate(std::move(stacked_jacobian), std::move(stacked_residual),
                   vision_noise_variance);
  }

  bool BuildSlamObservationJacobian(int feature_index, int clone_index,
                                    const StereoObservation &observation,
                                    Eigen::MatrixXd &jacobian,
                                    Eigen::VectorXd &residual) const
  {
    const SlamFeature &feature = slam_features_[feature_index];
    const CameraClone &clone   = clones_[clone_index];
    const Eigen::Matrix3d camera_from_world
        = clone.world_from_camera_rotation.inverse().matrix();
    const Eigen::Vector3d point_in_camera
        = camera_from_world
          * (feature.position_in_world - clone.position_in_world);
    const double x = point_in_camera.x();
    const double y = point_in_camera.y();
    const double z = point_in_camera.z();
    if (z < 0.05)
    {
      return false;
    }

    Eigen::Matrix<double, 4, 3> projection_jacobian;
    projection_jacobian << 1 / z, 0, -x / (z * z), 0, 1 / z, -y / (z * z),
        1 / z, 0, -(x - baseline_) / (z * z), 0, 1 / z, -y / (z * z);

    Eigen::Matrix<double, 4, 6> clone_jacobian_block;
    clone_jacobian_block.leftCols<3>()
        = projection_jacobian * Sophus::SO3d::hat(point_in_camera);
    clone_jacobian_block.rightCols<3>()
        = -projection_jacobian * camera_from_world;

    Eigen::Matrix<double, 6, 1> yaw_direction;
    yaw_direction.head<3>()
        = clone.null_rotation.inverse().matrix() * gravity_in_world_;
    yaw_direction.tail<3>()
        = Sophus::SO3d::hat(feature.null_position - clone.null_position)
          * gravity_in_world_;
    clone_jacobian_block -= clone_jacobian_block * yaw_direction
                            * yaw_direction.transpose()
                            / yaw_direction.squaredNorm();

    jacobian = Eigen::MatrixXd::Zero(4, StateDim());
    jacobian.block<4, 6>(0, CloneStateIndex(clone_index))
        = clone_jacobian_block;
    jacobian.block<4, 3>(0, SlamFeatureStateIndex(feature_index))
        = -clone_jacobian_block.rightCols<3>();
    residual.resize(4);
    residual << observation.left_normalized.x() - x / z,
        observation.left_normalized.y() - y / z,
        observation.right_normalized.x() - (x - baseline_) / z,
        observation.right_normalized.y() - y / z;
    return true;
  }

  void UpdateWithSlamObservations(
      long frame_id,
      const std::vector<std::pair<int, StereoObservation>> &observations,
      std::vector<long> &dropped_feature_ids
  )
  {
    if (observations.empty())
    {
      return;
    }
    const int clone_index = FindCloneIndex(frame_id);
    if (clone_index < 0)
    {
      return;
    }

    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    std::vector<Eigen::MatrixXd> jacobian_blocks;
    std::vector<Eigen::VectorXd> residual_blocks;
    std::vector<int> failed_feature_indices;
    for (const auto &[feature_index, observation] : observations)
    {
      Eigen::MatrixXd jacobian;
      Eigen::VectorXd residual;
      if (!BuildSlamObservationJacobian(feature_index, clone_index, observation,
                                        jacobian, residual)
          || !PassesChiSquareGate(jacobian, residual, vision_noise_variance))
      {
        failed_feature_indices.push_back(feature_index);
        continue;
      }
      jacobian_blocks.push_back(std::move(jacobian));
      residual_blocks.push_back(std::move(residual));
    }

    if (!jacobian_blocks.empty())
    {
      const int total_rows = 4 * static_cast<int>(jacobian_blocks.size());
      Eigen::MatrixXd stacked_jacobian(total_rows, StateDim());
      Eigen::VectorXd stacked_residual(total_rows);
      for (size_t i = 0; i < jacobian_blocks.size(); ++i)
      {
        stacked_jacobian.middleRows(4 * static_cast<int>(i), 4)
            = jacobian_blocks[i];
        stacked_residual.segment(4 * static_cast<int>(i), 4)
            = residual_blocks[i];
      }
      ApplyEkfUpdate(std::move(stacked_jacobian), std::move(stacked_residual),
                     vision_noise_variance);
    }

    // 门限失败视为外点: 移出状态并反馈前端删除 (索引降序保证删除安全)
    std::ranges::sort(failed_feature_indices);
    for (const int feature_index : std::views::reverse(failed_feature_indices))
    {
      dropped_feature_ids.push_back(slam_features_[feature_index].feature_id);
      RemoveSlamFeatureAt(feature_index);
    }
  }

  void RemoveLostSlamFeatures(const std::set<long> &visible_ids)
  {
    for (int i = static_cast<int>(slam_features_.size()) - 1; i >= 0; --i)
    {
      if (!visible_ids.contains(slam_features_[i].feature_id))
      {
        RemoveSlamFeatureAt(i);
      }
    }
  }

  // 延迟初始化 (OpenVINS 风格): 对 H_点 做 QR, 前 3 行初始化特征均值/协方差/交叉协方差,
  // 其余行做标准 MSCKF 更新; 成功后特征进入状态向量, 后续逐帧走 SLAM 更新
  bool PromoteTrackToSlamFeature(const FeatureTrack &track)
  {
    if (slam_features_.size() >= kMaxSlamFeatureCount)
    {
      return false;
    }
    const auto world_point = TriangulateFeature(track);
    if (!world_point)
    {
      return false;
    }
    Eigen::MatrixXd state_jacobian, point_jacobian;
    Eigen::VectorXd residual;
    if (!BuildTrackJacobians(track, *world_point, nullptr, state_jacobian,
                             point_jacobian, residual))
    {
      return false;
    }
    const int rows = static_cast<int>(residual.size());
    if (rows <= 3)
    {
      return false;
    }

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(point_jacobian);
    const Eigen::MatrixXd rotated_state_jacobian
        = qr.householderQ().transpose() * state_jacobian;
    const Eigen::VectorXd rotated_residual
        = qr.householderQ().transpose() * residual;
    Eigen::Matrix3d point_triangle;
    point_triangle = qr.matrixQR().topRows(3).triangularView<Eigen::Upper>();
    if (std::abs(point_triangle(0, 0)) < 1e-6
        || std::abs(point_triangle(1, 1)) < 1e-6
        || std::abs(point_triangle(2, 2)) < 1e-6)
    {
      return false;
    }

    const Eigen::MatrixXd initializer_jacobian
        = rotated_state_jacobian.topRows(3);
    const Eigen::Vector3d initializer_residual = rotated_residual.head(3);
    Eigen::MatrixXd msckf_jacobian
        = rotated_state_jacobian.bottomRows(rows - 3);
    Eigen::VectorXd msckf_residual = rotated_residual.tail(rows - 3);
    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    if (!PassesChiSquareGate(msckf_jacobian, msckf_residual,
                             vision_noise_variance))
    {
      return false;
    }

    const Eigen::Matrix3d triangle_inverse = point_triangle.inverse();
    const Eigen::Matrix3d feature_covariance
        = triangle_inverse
          * (initializer_jacobian * covariance_
                 * initializer_jacobian.transpose()
             + vision_noise_variance * Eigen::Matrix3d::Identity())
          * triangle_inverse.transpose();
    const Eigen::MatrixXd cross_covariance = -covariance_
                                             * initializer_jacobian.transpose()
                                             * triangle_inverse.transpose();

    const Eigen::Vector3d corrected_position
        = *world_point + triangle_inverse * initializer_residual;
    const int insert_at
        = kImuErrorDim + 3 * static_cast<int>(slam_features_.size());
    InsertCovarianceBlock(insert_at, cross_covariance, feature_covariance);
    slam_features_.push_back({track.feature_id, corrected_position,
                              corrected_position});
    Symmetrize();

    // MSCKF 部分在扩维后的状态上执行, 新特征对应列为零 (QR 已保证正交)
    Eigen::MatrixXd grown_jacobian
        = Eigen::MatrixXd::Zero(msckf_jacobian.rows(), StateDim());
    grown_jacobian.leftCols(insert_at) = msckf_jacobian.leftCols(insert_at);
    grown_jacobian.rightCols(msckf_jacobian.cols() - insert_at)
        = msckf_jacobian.rightCols(msckf_jacobian.cols() - insert_at);
    ApplyEkfUpdate(std::move(grown_jacobian), std::move(msckf_residual),
                   vision_noise_variance);
    return true;
  }

  void ApplyEkfUpdate(Eigen::MatrixXd jacobian, Eigen::VectorXd residual,
                      double noise_variance)
  {
    if (jacobian.rows() > jacobian.cols())
    {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(jacobian);
      const int cols = static_cast<int>(jacobian.cols());
      const Eigen::VectorXd rotated_residual
          = qr.householderQ().transpose() * residual;
      Eigen::MatrixXd compressed(cols, cols);
      compressed = qr.matrixQR().topRows(cols).triangularView<Eigen::Upper>();
      jacobian   = std::move(compressed);
      residual   = rotated_residual.head(cols);
    }

    Eigen::MatrixXd innovation_covariance
        = jacobian * covariance_ * jacobian.transpose();
    innovation_covariance.diagonal().array() += noise_variance;
    const Eigen::MatrixXd kalman_gain = innovation_covariance.ldlt()
                                            .solve(jacobian * covariance_)
                                            .transpose();
    const Eigen::VectorXd correction  = kalman_gain * residual;

    world_from_imu_rotation_ = world_from_imu_rotation_
                               * Sophus::SO3d::exp(correction.segment<3>(0));
    imu_position_ += correction.segment<3>(3);
    imu_velocity_ += correction.segment<3>(6);
    gyro_bias_ += correction.segment<3>(9);
    accel_bias_ += correction.segment<3>(12);
    time_offset_ += correction(kTimeOffsetIndex);
    gravity_in_world_ += correction.segment<3>(kGravityIndex);
    for (size_t i = 0; i < slam_features_.size(); ++i)
    {
      slam_features_[i].position_in_world
          += correction.segment<3>(SlamFeatureStateIndex(static_cast<int>(i)));
    }
    for (size_t i = 0; i < clones_.size(); ++i)
    {
      const int base = CloneStateIndex(static_cast<int>(i));
      clones_[i].world_from_camera_rotation
          = clones_[i].world_from_camera_rotation
            * Sophus::SO3d::exp(correction.segment<3>(base));
      clones_[i].position_in_world += correction.segment<3>(base + 3);
    }

    const Eigen::MatrixXd identity_minus_kh
        = Eigen::MatrixXd::Identity(StateDim(), StateDim())
          - kalman_gain * jacobian;
    covariance_
        = identity_minus_kh * covariance_ * identity_minus_kh.transpose()
          + kalman_gain * noise_variance * kalman_gain.transpose();
    Symmetrize();
  }

  void RemoveCovarianceBlock(int start, int size)
  {
    const int dim  = static_cast<int>(covariance_.rows());
    const int tail = dim - start - size;
    Eigen::MatrixXd reduced(dim - size, dim - size);
    reduced.topLeftCorner(start, start)
        = covariance_.topLeftCorner(start, start);
    reduced.topRightCorner(start, tail)
        = covariance_.topRightCorner(start, tail);
    reduced.bottomLeftCorner(tail, start)
        = covariance_.bottomLeftCorner(tail, start);
    reduced.bottomRightCorner(tail, tail)
        = covariance_.bottomRightCorner(tail, tail);
    covariance_ = std::move(reduced);
  }

  void InsertCovarianceBlock(int insert_at,
                             const Eigen::MatrixXd &cross_covariance,
                             const Eigen::Matrix3d &block_covariance)
  {
    const int old_dim     = static_cast<int>(covariance_.rows());
    const int tail        = old_dim - insert_at;
    Eigen::MatrixXd grown = Eigen::MatrixXd::Zero(old_dim + 3, old_dim + 3);
    grown.topLeftCorner(insert_at, insert_at)
        = covariance_.topLeftCorner(insert_at, insert_at);
    grown.block(0, insert_at + 3, insert_at, tail)
        = covariance_.block(0, insert_at, insert_at, tail);
    grown.block(insert_at + 3, 0, tail, insert_at)
        = covariance_.block(insert_at, 0, tail, insert_at);
    grown.block(insert_at + 3, insert_at + 3, tail, tail)
        = covariance_.block(insert_at, insert_at, tail, tail);
    grown.block(0, insert_at, insert_at, 3)
        = cross_covariance.topRows(insert_at);
    grown.block(insert_at + 3, insert_at, tail, 3)
        = cross_covariance.bottomRows(tail);
    grown.block(insert_at, 0, 3, insert_at)
        = cross_covariance.topRows(insert_at).transpose();
    grown.block(insert_at, insert_at + 3, 3, tail)
        = cross_covariance.bottomRows(tail).transpose();
    grown.block<3, 3>(insert_at, insert_at) = block_covariance;
    covariance_                             = std::move(grown);
  }

  void RemoveCloneAt(int clone_index)
  {
    RemoveCovarianceBlock(CloneStateIndex(clone_index), kCloneErrorDim);
    clones_.erase(clones_.begin() + clone_index);
  }

  void RemoveSlamFeatureAt(int feature_index)
  {
    RemoveCovarianceBlock(SlamFeatureStateIndex(feature_index), 3);
    slam_features_.erase(slam_features_.begin() + feature_index);
  }

  // 冗余克隆选择 (MSCKF-VIO): 相对参考帧运动小的近期克隆优先, 否则删最老的; 每次选 2 个
  std::vector<int> SelectCloneIndicesToRemove() const
  {
    const int count              = static_cast<int>(clones_.size());
    const CameraClone &key_clone = clones_[count - 4];
    int moving_candidate         = count - 3;
    int oldest_candidate         = 0;
    std::vector<int> remove_indices;
    for (int i = 0; i < 2; ++i)
    {
      const CameraClone &candidate = clones_[moving_candidate];
      const double rotation_change
          = (key_clone.world_from_camera_rotation.inverse()
             * candidate.world_from_camera_rotation)
                .log()
                .norm();
      const double translation_change
          = (candidate.position_in_world - key_clone.position_in_world).norm();
      if (rotation_change < kRedundantRotationThreshold
          && translation_change < kRedundantTranslationThreshold)
      {
        remove_indices.push_back(moving_candidate++);
      }
      else
      {
        remove_indices.push_back(oldest_candidate++);
      }
    }
    std::ranges::sort(remove_indices);
    return remove_indices;
  }

  void PruneClonesAndAbsorbObservations()
  {
    const std::vector<int> remove_indices = SelectCloneIndicesToRemove();
    std::set<long> removed_frame_ids;
    for (const int index : remove_indices)
    {
      removed_frame_ids.insert(clones_[index].frame_id);
    }

    // 第一遍: 观测到被删克隆、且足够长的活跃轨迹升级为 SLAM 特征 (吸收其全部历史观测)
    for (auto it = active_tracks_.begin(); it != active_tracks_.end();)
    {
      FeatureTrack &track            = it->second;
      const bool observed_in_removed = std::ranges::any_of(
          removed_frame_ids, [&](long removed_frame_id)
          { return track.observations_by_frame.contains(removed_frame_id); }
      );
      if (observed_in_removed
          && track.observations_by_frame.size() >= kMinSlamTrackLength
          && PromoteTrackToSlamFeature(track))
      {
        it = active_tracks_.erase(it); // 前端继续跟踪, 后续观测走 SLAM 更新
      }
      else
      {
        ++it;
      }
    }

    // 第二遍: 其余轨迹只吸收被删克隆上的观测 (>=2 帧才有零空间余量), 轨迹保活
    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    std::vector<Eigen::MatrixXd> jacobian_blocks;
    std::vector<Eigen::VectorXd> residual_blocks;
    int total_rows = 0;
    for (auto it = active_tracks_.begin(); it != active_tracks_.end();)
    {
      FeatureTrack &track = it->second;
      std::vector<long> involved_frames;
      for (const long removed_frame_id : removed_frame_ids)
      {
        if (track.observations_by_frame.contains(removed_frame_id))
        {
          involved_frames.push_back(removed_frame_id);
        }
      }
      if (involved_frames.empty())
      {
        ++it;
        continue;
      }

      if (involved_frames.size() >= 2 && total_rows < kMaxUpdateRows)
      {
        const auto world_point = TriangulateFeature(track);
        if (world_point)
        {
          Eigen::MatrixXd projected_jacobian;
          Eigen::VectorXd projected_residual;
          if (ComputeProjectedFeatureJacobian(track, *world_point,
                                              projected_jacobian,
                                              projected_residual,
                                              &removed_frame_ids)
              && PassesChiSquareGate(projected_jacobian, projected_residual,
                                     vision_noise_variance))
          {
            total_rows += static_cast<int>(projected_jacobian.rows());
            jacobian_blocks.push_back(std::move(projected_jacobian));
            residual_blocks.push_back(std::move(projected_residual));
          }
        }
      }
      for (const long involved_frame_id : involved_frames)
      {
        track.observations_by_frame.erase(involved_frame_id);
      }
      if (track.observations_by_frame.empty())
      {
        it = active_tracks_.erase(it);
      }
      else
      {
        ++it;
      }
    }

    if (!jacobian_blocks.empty())
    {
      Eigen::MatrixXd stacked_jacobian(total_rows, StateDim());
      Eigen::VectorXd stacked_residual(total_rows);
      int row = 0;
      for (size_t i = 0; i < jacobian_blocks.size(); ++i)
      {
        stacked_jacobian.middleRows(row, jacobian_blocks[i].rows())
            = jacobian_blocks[i];
        stacked_residual.segment(row, residual_blocks[i].size())
            = residual_blocks[i];
        row += static_cast<int>(jacobian_blocks[i].rows());
      }
      ApplyEkfUpdate(std::move(stacked_jacobian), std::move(stacked_residual),
                     vision_noise_variance);
    }

    for (const int index : std::views::reverse(remove_indices))
    {
      RemoveCloneAt(index);
    }
  }

  // OC 线性化点仅在初始化与传播时保存, EKF 更新不改动, 保证不可观子空间跨时刻一致
  void SaveNullLinearizationPoint()
  {
    null_rotation_ = world_from_imu_rotation_;
    null_position_ = imu_position_;
    null_velocity_ = imu_velocity_;
    null_gravity_  = gravity_in_world_;
  }

  Sophus::SE3d body_from_camera_;
  double baseline_               = 0;
  double pixel_noise_normalized_ = 0;
  ImuNoiseParameters imu_noise_;

  Sophus::SO3d world_from_imu_rotation_;
  Eigen::Vector3d imu_position_     = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_velocity_     = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_        = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias_       = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_in_world_ = Eigen::Vector3d(0, 0, -kGravity);
  double time_offset_ = 0; // 图像时刻 + time_offset_ = 对应的 IMU 时刻
  double imu_time_    = 0;
  ImuSample last_imu_;
  bool has_last_imu_ = false;

  Sophus::SO3d null_rotation_;
  Eigen::Vector3d null_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d null_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d null_gravity_  = Eigen::Vector3d(0, 0, -kGravity);

  std::deque<CameraClone> clones_;
  std::vector<SlamFeature> slam_features_;
  Eigen::MatrixXd covariance_
      = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
  std::unordered_map<long, FeatureTrack> active_tracks_;
};

// ============================ 主流程 ============================
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  try
  {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    const fs::path &dataset_root     = options.dataset_root;
    std::println("初始化方式: {}",
                 options.initialization_mode == InitializationMode::kGroundTruth
                     ? "groundtruth 姿态"
                     : "静止 IMU");
    const std::vector<ImuSample> imu_samples = LoadImuSamples(dataset_root);
    std::vector<StereoFrame> stereo_frames   = LoadStereoFrames(dataset_root);
    std::println("IMU 样本数 {}, 双目帧数 {}", imu_samples.size(),
                 stereo_frames.size());
    if (imu_samples.empty() || stereo_frames.empty())
    {
      return 1;
    }

    const CameraCalibration left_calibration
        = LoadCameraCalibration(dataset_root / "cam0" / "sensor.yaml");
    const CameraCalibration right_calibration
        = LoadCameraCalibration(dataset_root / "cam1" / "sensor.yaml");
    const ImuNoiseParameters imu_noise
        = LoadImuNoiseParameters(dataset_root / "imu0" / "sensor.yaml");
    std::println("标定读取完成: cam0 fu={:.3f}, cam1 fu={:.3f}, 分辨率 {}x{}, "
                 "陀螺白噪声 {:.4e}",
                 left_calibration.camera_matrix.at<double>(0, 0),
                 right_calibration.camera_matrix.at<double>(0, 0),
                 left_calibration.image_size.width,
                 left_calibration.image_size.height,
                 imu_noise.gyro_noise_density);

    StereoRectifier rectifier(left_calibration, right_calibration);
    std::println("矫正后焦距 {:.2f} px, 基线 {:.4f} m",
                 rectifier.focal_length(), rectifier.baseline());

    ClaheEnhancer enhancer;
    StationaryDetector stationary_detector;
    FeatureTracker tracker(rectifier);
    Msckf filter(rectifier.body_from_rectified_left(), rectifier.baseline(),
                 rectifier.focal_length(), imu_noise,
                 options.initial_time_offset);
    std::println("相机-IMU 时间偏移初值 {:+.2f} ms (滤波器在线估计)",
                 options.initial_time_offset * 1e3);

    size_t imu_index = 0;
    if (options.initialization_mode == InitializationMode::kGroundTruth)
    {
      // groundtruth 可能晚于传感器数据开始: 抛弃 groundtruth 首条记录之前的
      // 双目帧与 IMU 样本, 从有真值的时刻开始
      const double groundtruth_start_time
          = LoadGroundTruthStartTime(dataset_root);
      const auto first_covered_frame = std::ranges::find_if(
          stereo_frames, [groundtruth_start_time](const StereoFrame &frame)
          { return frame.time >= groundtruth_start_time; }
      );
      if (first_covered_frame == stereo_frames.end())
      {
        throw std::runtime_error("所有双目帧都早于 groundtruth 起始时刻");
      }
      if (first_covered_frame != stereo_frames.begin())
      {
        std::println("groundtruth 起始于 t={:.3f}s, 抛弃之前的 {} 帧双目图像",
                     groundtruth_start_time,
                     std::distance(stereo_frames.begin(), first_covered_frame));
        stereo_frames.erase(stereo_frames.begin(), first_covered_frame);
      }
      const double first_frame_time = stereo_frames.front().time;
      const GroundTruthState initial_state
          = LoadClosestGroundTruthState(dataset_root, first_frame_time);
      filter.InitializeFromGroundTruth(initial_state);
      while (imu_index < imu_samples.size()
             && imu_samples[imu_index].time < first_frame_time)
      {
        ++imu_index; // 首帧之前的 IMU 不参与积分
      }
      std::println(
          "groundtruth 初始状态: t={:.3f}s 位置 [{:.3f} {:.3f} {:.3f}]",
          initial_state.time, initial_state.position.x(),
          initial_state.position.y(), initial_state.position.z()
      );
    }
    else
    {
      const double first_frame_time = stereo_frames.front().time;
      std::vector<ImuSample> initial_samples;
      while (imu_index < imu_samples.size()
             && imu_samples[imu_index].time < first_frame_time)
      {
        initial_samples.push_back(imu_samples[imu_index++]);
      }
      while (initial_samples.size() < 200 && imu_index < imu_samples.size())
      {
        initial_samples.push_back(imu_samples[imu_index++]);
      }
      filter.InitializeFromStaticImu(initial_samples);
    }

    const fs::path &trajectory_path = options.output_trajectory_path;
    if (trajectory_path.has_parent_path())
    {
      fs::create_directories(trajectory_path.parent_path());
    }
    std::ofstream trajectory_file(trajectory_path);
    if (!trajectory_file)
    {
      throw std::runtime_error("无法创建轨迹输出文件: "
                               + trajectory_path.string());
    }
    long frame_id                   = 0;
    long zero_velocity_update_count = 0;
    const Sophus::SO3d camera_rotation_in_body
        = rectifier.body_from_rectified_left().so3();
    Sophus::SO3d previous_world_from_camera;
    bool has_previous_camera_rotation = false;
    for (const StereoFrame &frame : stereo_frames)
    {
      const double clone_time = frame.time + filter.time_offset_camera_to_imu();
      while (imu_index < imu_samples.size()
             && imu_samples[imu_index].time <= clone_time)
      {
        stationary_detector.AddImuSample(imu_samples[imu_index]);
        filter.PropagateWithImu(imu_samples[imu_index++]);
      }
      if (imu_index > 0 && imu_index < imu_samples.size())
      {
        filter.PropagateWithImu(InterpolateImuSample(imu_samples[imu_index - 1],
                                                     imu_samples[imu_index],
                                                     clone_time));
      }

      if (stationary_detector.IsStationary()
          && filter.ApplyZeroVelocityUpdate())
      {
        ++zero_velocity_update_count;
      }

      const cv::Mat raw_left
          = cv::imread(frame.left_image_path.string(), cv::IMREAD_GRAYSCALE);
      const cv::Mat raw_right
          = cv::imread(frame.right_image_path.string(), cv::IMREAD_GRAYSCALE);
      if (raw_left.empty() || raw_right.empty())
      {
        continue;
      }

      cv::Mat rectified_left, rectified_right;
      rectifier.Rectify(raw_left, raw_right, rectified_left, rectified_right);
      const cv::Mat enhanced_left  = enhancer.Enhance(rectified_left);
      const cv::Mat enhanced_right = enhancer.Enhance(rectified_right);

      // 陀螺辅助: 滤波器传播的姿态差即两帧间(去零偏)陀螺积分, 作为光流的旋转先验
      const Sophus::SO3d world_from_camera_now
          = filter.world_from_imu_rotation() * camera_rotation_in_body;
      const Sophus::SO3d current_camera_from_previous_camera
          = has_previous_camera_rotation
                ? world_from_camera_now.inverse() * previous_world_from_camera
                : Sophus::SO3d();
      const auto tracked = tracker.Track(enhanced_left, enhanced_right,
                                         current_camera_from_previous_camera);
      const std::vector<long> dropped_feature_ids
          = filter.ProcessFrame(frame_id, tracked);
      tracker.DropFeatures(dropped_feature_ids);
      // 接入单目相对运动算法时在此融合 (rVec/tVec 约定见 Msckf::MonocularUpdate 注释):
      //   filter.MonocularUpdate(estimated_rotation_vector, estimated_translation_direction);
      previous_world_from_camera
          = filter.world_from_imu_rotation() * camera_rotation_in_body;
      has_previous_camera_rotation = true;

      const Eigen::Quaterniond orientation
          = filter.world_from_imu_rotation().unit_quaternion();
      const Eigen::Vector3d &position = filter.imu_position();
      std::println(trajectory_file,
                   "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                   clone_time, position.x(), position.y(), position.z(),
                   orientation.x(), orientation.y(), orientation.z(),
                   orientation.w());

      if (frame_id % 50 == 0)
      {
        std::println(
            "帧 {:4}  位置 [{:7.3f} {:7.3f} {:7.3f}]  速度 {:5.2f} m/s  t_d "
            "{:+.2f} ms  |g| {:.3f}  特征 {:3}  SLAM {:2}  克隆 {}",
            frame_id, position.x(), position.y(), position.z(),
            filter.imu_velocity().norm(),
            filter.time_offset_camera_to_imu() * 1e3,
            filter.gravity_in_world().norm(), tracked.size(),
            filter.slam_feature_count(), filter.clone_count()
        );
      }
      ++frame_id;
    }
    const Eigen::Vector3d &gravity = filter.gravity_in_world();
    std::println("完成, 轨迹已写入 {} (TUM 格式), ZUPT 触发 {} 次, 最终 t_d "
                 "{:+.3f} ms, 重力 [{:.4f} {:.4f} {:.4f}] m/s^2",
                 fs::absolute(trajectory_path).string(),
                 zero_velocity_update_count,
                 filter.time_offset_camera_to_imu() * 1e3, gravity.x(),
                 gravity.y(), gravity.z());
  }
  catch (const std::exception &error)
  {
    std::println(stderr, "错误: {}", error.what());
    std::println(
        stderr,
        "用法: {} <数据集mav0路径> [--init imu|groundtruth] [--time-offset 秒] "
        "[--output /path/to/filename.tum]",
        argv[0]
    );
    return 1;
  }
  return 0;
}
