// msckf.cpp —— 基于 MSCKF 的双目视觉 + IMU 紧耦合里程计 (EuRoC MAV V2_01_easy)
// 依赖: OpenCV 4 / Eigen 3.4 / Sophus / Ceres Solver / yaml-cpp, 标准: C++26
// 标定与噪声参数在运行时用 yaml-cpp 从数据集各 sensor.yaml 读取, 不做硬编码
//
// 编译:
//   g++ -std=c++2c -O3 -march=native msckf.cpp -o msckf \
//       $(pkg-config --cflags --libs opencv4 eigen3 yaml-cpp) -lceres -lglog -pthread
// 运行:
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0                      # 静止 IMU 初始化 (默认)
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --init groundtruth   # 真值姿态初始化
// 输出:
//   trajectory_tum.txt (TUM 格式: time px py pz qx qy qz qw)

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
  fs::path dataset_root                  = "EuRoC_MAV_Datasets/V2_01_easy/mav0";
  InitializationMode initialization_mode = InitializationMode::kStaticImu;
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
    else
    {
      options.dataset_root = fs::path(argument);
    }
  }
  return options;
}

// ================ 标定参数读取 (yaml-cpp 解析 EuRoC sensor.yaml) ================
constexpr double kGravity = 9.81; // sensor.yaml 中无重力项, 保留为常量

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
      = (cv::Mat_<double>(3, 3) << intrinsics[0].as<double>(), 0,
         intrinsics[2].as<double>(), 0, intrinsics[1].as<double>(),
         intrinsics[3].as<double>(), 0, 0, 1);
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

  std::vector<TrackedFeature> Track(const cv::Mat &rectified_left,
                                    const cv::Mat &rectified_right)
  {
    std::vector<cv::Point2f> left_points;
    std::vector<long> feature_ids;
    TrackFromPreviousFrame(rectified_left, left_points, feature_ids);
    DetectNewFastCorners(rectified_left, left_points, feature_ids);

    std::vector<TrackedFeature> result;
    std::vector<cv::Point2f> kept_points;
    std::vector<long> kept_ids;
    if (!left_points.empty())
    {
      std::vector<cv::Point2f> right_points = left_points;
      std::vector<uchar> status;
      std::vector<float> error;
      cv::calcOpticalFlowPyrLK(rectified_left, rectified_right, left_points,
                               right_points, status, error, cv::Size(21, 21), 3,
                               {cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                30, 0.01},
                               cv::OPTFLOW_USE_INITIAL_FLOW);
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
    previous_left_image_  = rectified_left.clone();
    return result;
  }

private:
  static constexpr size_t kMaxFeatureCount    = 200;
  static constexpr int kFastThreshold         = 20;
  static constexpr double kMinFeatureDistance = 25.0;
  static constexpr double kMaxEpipolarError   = 3.0;
  static constexpr double kMinDisparity       = 0.5;
  static constexpr int kImageBorder           = 5;

  bool InsideImage(const cv::Point2f &point) const
  {
    const cv::Size &image_size = rectifier_.image_size();
    return point.x >= kImageBorder && point.y >= kImageBorder
           && point.x < image_size.width - kImageBorder
           && point.y < image_size.height - kImageBorder;
  }

  void TrackFromPreviousFrame(const cv::Mat &current_left,
                              std::vector<cv::Point2f> &left_points,
                              std::vector<long> &feature_ids)
  {
    if (previous_left_image_.empty() || previous_left_points_.empty())
    {
      return;
    }
    std::vector<cv::Point2f> current_points;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(previous_left_image_, current_left,
                             previous_left_points_, current_points, status,
                             error);

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
  cv::Mat previous_left_image_;
  std::vector<cv::Point2f> previous_left_points_;
  std::vector<long> previous_feature_ids_;
  long next_feature_id_ = 0;
};

// ==================== 特征三角化 (Ceres 双目重投影) ====================
struct StereoReprojectionCost
{
  StereoReprojectionCost(const Sophus::SE3d &camera_from_world,
                         const StereoObservation &observation,
                         double baseline) :
    rotation(camera_from_world.rotationMatrix()),
    translation(camera_from_world.translation()), measurement(observation),
    baseline(baseline)
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
};

class Msckf
{
public:
  static constexpr int kImuErrorDim
      = 15; // [姿态3 位置3 速度3 陀螺零偏3 加计零偏3]
  static constexpr int kCloneErrorDim     = 6; // [姿态3 位置3]
  static constexpr size_t kMaxCloneCount  = 11;
  static constexpr size_t kMinTrackLength = 3;
  static constexpr int kMaxUpdateRows     = 600;

  Msckf(const Sophus::SE3d &body_from_camera, double baseline,
        double focal_length, const ImuNoiseParameters &imu_noise) :
    body_from_camera_(body_from_camera), baseline_(baseline),
    pixel_noise_normalized_(1.5 / focal_length), imu_noise_(imu_noise)
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
    gyro_bias_    = gyro_mean;
    imu_time_     = samples.back().time;
    last_imu_     = samples.back();
    has_last_imu_ = true;

    covariance_ = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
    covariance_.diagonal() << 1e-4, 1e-4, 1e-3, 1e-8, 1e-8, 1e-8, 1e-2, 1e-2,
        1e-2, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3;
  }

  void InitializeFromGroundTruth(const GroundTruthState &state)
  {
    world_from_imu_rotation_
        = Sophus::SO3d(state.world_from_body_orientation.normalized());
    imu_position_ = state.position;
    imu_velocity_ = state.velocity;
    gyro_bias_    = state.gyro_bias;
    accel_bias_   = state.accel_bias;
    imu_time_     = state.time;
    has_last_imu_ = false; // 之后遇到的第一个 IMU 样本仅记录为积分起点

    covariance_ = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
    covariance_.diagonal() << 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4,
        1e-4, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5;
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
    const Eigen::Vector3d gravity(0, 0, -kGravity);
    const Eigen::Matrix3d rotation_matrix = world_from_imu_rotation_.matrix();
    const Eigen::Vector3d accel_in_world  = rotation_matrix * accel + gravity;

    world_from_imu_rotation_
        = world_from_imu_rotation_ * Sophus::SO3d::exp(gyro * dt);
    imu_position_ += imu_velocity_ * dt + 0.5 * accel_in_world * dt * dt;
    imu_velocity_ += accel_in_world * dt;
    imu_time_ = sample.time;
    last_imu_ = sample;

    Eigen::Matrix<double, 15, 15> transition
        = Eigen::Matrix<double, 15, 15>::Identity();
    transition.block<3, 3>(0, 0) = Sophus::SO3d::exp(-gyro * dt).matrix();
    transition.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt;
    transition.block<3, 3>(3, 0)
        = -0.5 * rotation_matrix * Sophus::SO3d::hat(accel) * dt * dt;
    transition.block<3, 3>(3, 6)  = Eigen::Matrix3d::Identity() * dt;
    transition.block<3, 3>(3, 12) = -0.5 * rotation_matrix * dt * dt;
    transition.block<3, 3>(6, 0)
        = -rotation_matrix * Sophus::SO3d::hat(accel) * dt;
    transition.block<3, 3>(6, 12) = -rotation_matrix * dt;

    Eigen::Matrix<double, 15, 12> noise_jacobian
        = Eigen::Matrix<double, 15, 12>::Zero();
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

    const Eigen::Matrix<double, 15, 15> discrete_noise
        = noise_jacobian * continuous_noise * noise_jacobian.transpose() * dt;

    covariance_.topLeftCorner<15, 15>()
        = transition * covariance_.topLeftCorner<15, 15>()
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
  }

  void ProcessFrame(long frame_id,
                    const std::vector<FeatureTracker::TrackedFeature> &tracked)
  {
    AugmentCameraClone(frame_id);

    std::set<long> visible_ids;
    for (const auto &feature : tracked)
    {
      FeatureTrack &track = active_tracks_[feature.feature_id];
      track.feature_id    = feature.feature_id;
      track.observations_by_frame[frame_id] = feature.observation;
      visible_ids.insert(feature.feature_id);
    }

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

    if (clones_.size() > kMaxCloneCount)
    {
      const long oldest_frame_id = clones_.front().frame_id;
      for (auto it = active_tracks_.begin(); it != active_tracks_.end();)
      {
        if (it->second.observations_by_frame.contains(oldest_frame_id))
        {
          finished_tracks.push_back(std::move(it->second));
          it = active_tracks_.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }

    UpdateWithTracks(finished_tracks);

    if (clones_.size() > kMaxCloneCount)
    {
      RemoveOldestClone();
    }
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
  size_t clone_count() const
  {
    return clones_.size();
  }

private:
  int StateDim() const
  {
    return kImuErrorDim + kCloneErrorDim * static_cast<int>(clones_.size());
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

    clones_.push_back({frame_id, world_from_camera_rotation, camera_position});
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

  bool
  ComputeProjectedFeatureJacobian(const FeatureTrack &track,
                                  const Eigen::Vector3d &world_point,
                                  Eigen::MatrixXd &projected_jacobian,
                                  Eigen::VectorXd &projected_residual) const
  {
    const int max_rows
        = 4 * static_cast<int>(track.observations_by_frame.size());
    Eigen::MatrixXd state_jacobian
        = Eigen::MatrixXd::Zero(max_rows, StateDim());
    Eigen::MatrixXd point_jacobian = Eigen::MatrixXd::Zero(max_rows, 3);
    Eigen::VectorXd residual       = Eigen::VectorXd::Zero(max_rows);

    int row = 0;
    for (const auto &[frame_id, observation] : track.observations_by_frame)
    {
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

      const int column = kImuErrorDim + kCloneErrorDim * clone_index;
      state_jacobian.block<4, 3>(row, column)
          = projection_jacobian * Sophus::SO3d::hat(point_in_camera);
      state_jacobian.block<4, 3>(row, column + 3)
          = -projection_jacobian * camera_from_world;
      point_jacobian.block<4, 3>(row, 0)
          = projection_jacobian * camera_from_world;
      residual.segment<4>(row) << observation.left_normalized.x() - x / z,
          observation.left_normalized.y() - y / z,
          observation.right_normalized.x() - (x - baseline_) / z,
          observation.right_normalized.y() - y / z;
      row += 4;
    }
    if (row <= 3)
    {
      return false;
    }
    state_jacobian.conservativeResize(row, Eigen::NoChange);
    point_jacobian.conservativeResize(row, Eigen::NoChange);
    residual.conservativeResize(row);

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(point_jacobian);
    const Eigen::MatrixXd q_full         = qr.householderQ();
    const Eigen::MatrixXd left_nullspace = q_full.rightCols(row - 3);
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
                           const Eigen::VectorXd &residual) const
  {
    Eigen::MatrixXd innovation_covariance
        = jacobian * covariance_ * jacobian.transpose();
    innovation_covariance.diagonal().array()
        += pixel_noise_normalized_ * pixel_noise_normalized_;
    const double mahalanobis_distance
        = residual.dot(innovation_covariance.ldlt().solve(residual));
    return mahalanobis_distance
           < ChiSquare95(static_cast<int>(residual.size()));
  }

  void UpdateWithTracks(const std::vector<FeatureTrack> &finished_tracks)
  {
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
      if (!PassesChiSquareGate(projected_jacobian, projected_residual))
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
    ApplyEkfUpdate(std::move(stacked_jacobian), std::move(stacked_residual));
  }

  void ApplyEkfUpdate(Eigen::MatrixXd jacobian, Eigen::VectorXd residual)
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

    const double noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
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
    for (size_t i = 0; i < clones_.size(); ++i)
    {
      const int base = kImuErrorDim + kCloneErrorDim * static_cast<int>(i);
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

  void RemoveOldestClone()
  {
    const int dim  = StateDim();
    const int head = kImuErrorDim;
    const int tail = dim - head - kCloneErrorDim;
    Eigen::MatrixXd reduced(dim - kCloneErrorDim, dim - kCloneErrorDim);
    reduced.topLeftCorner(head, head)  = covariance_.topLeftCorner(head, head);
    reduced.topRightCorner(head, tail) = covariance_.topRightCorner(head, tail);
    reduced.bottomLeftCorner(tail, head)
        = covariance_.bottomLeftCorner(tail, head);
    reduced.bottomRightCorner(tail, tail)
        = covariance_.bottomRightCorner(tail, tail);
    covariance_ = std::move(reduced);
    clones_.pop_front();
  }

  Sophus::SE3d body_from_camera_;
  double baseline_               = 0;
  double pixel_noise_normalized_ = 0;
  ImuNoiseParameters imu_noise_;

  Sophus::SO3d world_from_imu_rotation_;
  Eigen::Vector3d imu_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_    = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias_   = Eigen::Vector3d::Zero();
  double imu_time_              = 0;
  ImuSample last_imu_;
  bool has_last_imu_ = false;

  std::deque<CameraClone> clones_;
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
    const std::vector<StereoFrame> stereo_frames
        = LoadStereoFrames(dataset_root);
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
    FeatureTracker tracker(rectifier);
    Msckf filter(rectifier.body_from_rectified_left(), rectifier.baseline(),
                 rectifier.focal_length(), imu_noise);

    const double first_frame_time = stereo_frames.front().time;
    size_t imu_index              = 0;
    if (options.initialization_mode == InitializationMode::kGroundTruth)
    {
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

    std::ofstream trajectory_file("trajectory_tum.txt");
    long frame_id = 0;
    for (const StereoFrame &frame : stereo_frames)
    {
      while (imu_index < imu_samples.size()
             && imu_samples[imu_index].time <= frame.time)
      {
        filter.PropagateWithImu(imu_samples[imu_index++]);
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
      const auto tracked = tracker.Track(enhanced_left, enhanced_right);
      filter.ProcessFrame(frame_id, tracked);

      const Eigen::Quaterniond orientation
          = filter.world_from_imu_rotation().unit_quaternion();
      const Eigen::Vector3d &position = filter.imu_position();
      std::println(trajectory_file,
                   "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                   frame.time, position.x(), position.y(), position.z(),
                   orientation.x(), orientation.y(), orientation.z(),
                   orientation.w());

      if (frame_id % 50 == 0)
      {
        std::println("帧 {:4}  位置 [{:7.3f} {:7.3f} {:7.3f}]  速度 {:5.2f} "
                     "m/s  特征 {:3}  克隆 {}",
                     frame_id, position.x(), position.y(), position.z(),
                     filter.imu_velocity().norm(), tracked.size(),
                     filter.clone_count());
      }
      ++frame_id;
    }
    std::println("完成, 轨迹已写入 trajectory_tum.txt (TUM 格式)");
  }
  catch (const std::exception &error)
  {
    std::println(stderr, "错误: {}", error.what());
    std::println(stderr, "用法: {} <数据集mav0路径> [--init imu|groundtruth]",
                 argv[0]);
    return 1;
  }
  return 0;
}
