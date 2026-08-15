// msckf.cpp —— 基于 MSCKF 的双目/单目视觉 + IMU 紧耦合里程计 (EuRoC MAV V2_01_easy)
// 运行时检测数据集相机数目自动切换: cam0+cam1 存在 → 双目; 仅单相机 → 单目
// 依赖: OpenCV 4 / Eigen 3.4 / Sophus / Ceres Solver / yaml-cpp, 标准: C++26
// 标定与噪声参数在运行时用 yaml-cpp 从数据集各 sensor.yaml 读取, 不做硬编码
// 延迟补偿: IMU 插值精确传播到成像时刻, 相机-IMU 时间偏移 t_d 作为状态在线估计
// 重力向量 (gx, gy, gz) 加入误差状态在线估计, 初始化时由静止比力或先验给出
// ZUPT: IMU 方差静止检测触发零速伪量测; 滑窗消费的特征 id 回馈前端删除, 避免重复关联
// OC-EKF 一致性修正: 对 Φ 与 H 施加不可观子空间约束 (全局平移 + 绕重力偏航), 防伪可观
// 前端: 滤波姿态(陀螺积分)辅助光流初值预测 + buildOpticalFlowPyramid 金字塔复用
// 单目三角化: 首末帧视差角门限 + 两视图线性初值 + Ceres 重投影优化; 尺度由加速度计-重力可观
// 边缘化: 低运动冗余克隆优先剔除, 仅吸收被删克隆上的观测、轨迹保活
// MonocularUpdate: 融合外部单目算法输出的帧间相对旋转(高精度)与平移方向(无尺度、低精度),
//                  数据来自 ~/vio_ws/estimated_motion_cam0.csv, 按图像时间戳查找;
//                  流程: IMU 预测 → 克隆增广 → 单目先验融合 → 陀螺辅助光流 → 单目量测更新
// 状态向量: [IMU(19) | 相机克隆(6×N)], N ≤ kMaxCloneCount; 路标点不进入状态向量
// 建图: PointCloudMapper 收集每帧 MSCKF 量测中成功三角化的路标点, 运行结束后写入 PLY 文件
//
// 编译 (或直接使用配套 CMakeLists.txt):
//   g++ -std=c++2c -O3 -march=native msckf.cpp -o msckf $(pkg-config --cflags --libs opencv4 eigen3 yaml-cpp) -lceres -lglog -pthread
// 运行:
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0                      # 静止 IMU 初始化 (默认)
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --init groundtruth   # 真值姿态初始化
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --time-offset 0.005  # t_d 初值(秒), 在线精化
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --output /tmp/v2_01.tum  # 指定轨迹输出路径
//   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0 --pointcloud /tmp/map.ply # 指定点云输出路径
// 输出:
//   TUM 格式轨迹 (time px py pz qx qy qz qw), 默认写入当前目录 trajectory_tum.txt
//   ASCII PLY 点云文件, 默认写入当前目录 pointcloud.ply

#include <ceres/ceres.h>

#include <glog/logging.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
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

#include "euroc_vio/DatumFast.hpp"
#include "euroc_vio/VisionMode.hpp"

using FastVIO::DetectVisionMode;
using FastVIO::SelectMonoCameraDirectory;
using FastVIO::VisionMode;

namespace fs = std::filesystem;

// 帧编号
using FrameId = long;
// 特征编号
using FeatureId = long;

// ==================== 计时工具 (评估 MonocularUpdate 对运行效率的影响) ====================
// 用法: 分别在启用 / 不启用 MonocularUpdate (如 --mono-csv 指向不存在的文件) 的
// 两种配置下运行, 对比 "MonocularUpdate 耗时 + 视觉估计相关函数总耗时" 的报表,
// 验证单目先验能否通过改善光流初值与线性化点降低视觉部分的耗时。
// 宏 ENABLE_TIMER 默认开启; 编译时以 -DENABLE_TIMER=0 关闭,
// 关闭后所有计时代码退化为空操作, 完全回到无计时的业务逻辑。
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1
#endif

// 宏 ENABLE_TIME_LOGGER 控制逐帧视觉任务合计耗时的 CSV 输出
// (~/vio_ws/CornerTrackingStats.csv, 同名文件已存在时自动追加数字后缀),
// 仅在 ENABLE_TIMER == 1 时生效; 编译时以 -DENABLE_TIME_LOGGER=0 关闭。
// 输出数据可用 scripts/plot_corner_tracking_stats.py 绘制多次实验的对比曲线
#ifndef ENABLE_TIME_LOGGER
#define ENABLE_TIME_LOGGER 1
#endif

#if ENABLE_TIMER
// 按名字累计函数调用次数与总耗时, 程序结束时打印统计报表
class FunctionTimer
{
public:
  using Clock = std::chrono::steady_clock;

  // RAII 计时: 构造时记录起点, 析构时把区间耗时累加到同名条目
  class Scope
  {
  public:
    Scope(FunctionTimer &timer, std::string_view name) :
      timer_(timer), name_(name), start_(Clock::now())
    {
    }
    ~Scope()
    {
      timer_.Accumulate(name_, Clock::now() - start_);
    }
    Scope(const Scope &)            = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    FunctionTimer &timer_;
    std::string_view name_;
    Clock::time_point start_;
  };

  void Accumulate(std::string_view name, Clock::duration elapsed)
  {
    Entry &entry = entries_[std::string{name}];
    entry.total_elapsed += elapsed;
    ++entry.call_count;
  }

  double TotalMilliseconds(std::string_view name) const
  {
    const auto found = entries_.find(name);
    return found == entries_.end()
               ? 0.0
               : ToMilliseconds(found->second.total_elapsed);
  }

  void PrintReport() const
  {
    std::println("==== 耗时统计 (ENABLE_TIMER=1) ====");
    for (const auto &[name, entry] : entries_)
    {
      const double total_ms = ToMilliseconds(entry.total_elapsed);
      std::println("  {:<32} 调用 {:6} 次  总计 {:10.2f} ms  平均 {:8.4f} ms",
                   name, entry.call_count, total_ms,
                   entry.call_count > 0
                       ? total_ms / static_cast<double>(entry.call_count)
                       : 0.0);
    }
  }

private:
  struct Entry
  {
    Clock::duration total_elapsed{};
    long call_count = 0;
  };

  static double ToMilliseconds(Clock::duration elapsed)
  {
    return std::chrono::duration<double, std::milli>(elapsed).count();
  }

  // std::less<> 透明比较器: find 可直接用 string_view 查找, 免临时 string
  std::map<std::string, Entry, std::less<>> entries_;
};

#define TIMER_CONCAT_IMPL(a, b) a##b
#define TIMER_CONCAT(a, b) TIMER_CONCAT_IMPL(a, b)
// 对当前作用域计时, 计入 timer 中名为 name 的条目
#define TIME_SCOPE(timer, name)                                                \
  const FunctionTimer::Scope TIMER_CONCAT(timed_scope_, __LINE__)              \
  {                                                                            \
    (timer), (name)                                                            \
  }
#else
#define TIME_SCOPE(timer, name) static_cast<void>(0)
#endif

#if ENABLE_TIMER && ENABLE_TIME_LOGGER
// 逐帧记录视觉任务合计耗时并写入 CSV, 供 scripts/plot_corner_tracking_stats.py
// 对比多次实验的耗时曲线; 构造时确定输出路径, 不覆盖历史实验数据
class FrameTimeLogger
{
public:
  FrameTimeLogger()
  {
    const fs::path output_path
        = ResolveNonClobberPath(fs::path{std::getenv("HOME")} / kOutputDirectory
                                / kOutputFileName);
    if (output_path.has_parent_path())
    {
      fs::create_directories(output_path.parent_path());
    }
    stream_.open(output_path);
    if (!stream_)
    {
      throw std::runtime_error(
          std::format("无法创建逐帧视觉耗时输出文件: '{}'.",
                      fs::absolute(output_path).string())
      );
    }
    std::println("逐帧视觉任务耗时将写入 {} (ENABLE_TIME_LOGGER=1)",
                 fs::absolute(output_path).string());
    std::println(stream_, "frame_id,visual_total_ms");
  }

  void LogFrame(FrameId frame_id, double visual_total_ms)
  {
    std::println(stream_, "{},{:.4f}", frame_id, visual_total_ms);
  }

private:
  static constexpr std::string_view kOutputDirectory = "vio_ws";
  static constexpr std::string_view kOutputFileName = "CornerTrackingStats.csv";

  // 已有同名文件时不覆盖, 在文件名 stem 与后缀之间插入递增数字: 1, 2, 3, ...
  static fs::path ResolveNonClobberPath(const fs::path &desired_path)
  {
    if (!fs::exists(desired_path))
    {
      return desired_path;
    }
    const std::string stem      = desired_path.stem().string();
    const std::string extension = desired_path.extension().string();
    for (long sequence_number = 1;; ++sequence_number)
    {
      const fs::path candidate
          = desired_path.parent_path()
            / std::format("{}{}{}", stem, sequence_number, extension);
      if (!fs::exists(candidate))
      {
        return candidate;
      }
    }
  }

  std::ofstream stream_;
};
#endif

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
  throw std::runtime_error(
      std::format("未知的初始化方式 (可选: imu | groundtruth): {}.", value)
  );
}

struct CommandLineOptions
{
  fs::path dataset_root = fs::path{std::getenv("HOME")} / "EuRoC_MAV_Datasets"
                          / "V2_01_easy" / "mav0";
  InitializationMode initialization_mode = InitializationMode::kStaticImu;
  double initial_time_offset      = 0; // 图像时刻 + t_d = 对应的 IMU 时刻 (秒)
  fs::path output_trajectory_path = "trajectory_tum.txt";
  fs::path output_pointcloud_path = "pointcloud.ply";
  // 外部单目算法输出: 时间戳 + 左目系相对旋转轴角 + 左目系平移方向
  fs::path monocular_estimation_path
      = fs::path{std::getenv("HOME")} / "vio_ws" / "estimated_motion_cam0.csv";
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
    else if (argument.starts_with("--mono-csv="))
    {
      options.monocular_estimation_path = fs::path(argument.substr(11));
    }
    else if (argument == "--mono-csv")
    {
      if (++i >= argc)
      {
        throw std::runtime_error("--mono-csv 缺少取值 (外部单目估计 CSV 路径)");
      }
      options.monocular_estimation_path = fs::path(argv[i]);
    }
    else if (argument.starts_with("--pointcloud="))
    {
      options.output_pointcloud_path = fs::path(argument.substr(13));
    }
    else if (argument == "--pointcloud")
    {
      if (++i >= argc)
      {
        throw std::runtime_error("--pointcloud 缺少取值 (点云文件路径)");
      }
      options.output_pointcloud_path = fs::path(argv[i]);
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

// 从文件 sensor.yaml 中读取：从传感器坐标系到载具坐标系的变换
Sophus::SE3d ReadBodyFromSensorPose(const YAML::Node &sensor_node)
{
  const YAML::Node T_BS = sensor_node["T_BS"];
  if (!T_BS)
  {
    // 部分裁剪过的数据集缺 T_BS 字段 (如本测试数据集, VisualSim 仿真相机
    // 外参实际为单位变换), 回退单位外参并提示, 避免运行中断
    std::println(stderr, "警告: sensor.yaml 缺少 T_BS, 回退单位外参");
    return Sophus::SE3d();
  }
  const YAML::Node pose_data = T_BS["data"];
  if (!pose_data || pose_data.size() != 16)
  {
    throw std::runtime_error("sensor.yaml 缺少配置 T_BS.data");
  }
  Eigen::Matrix4d matrix;
  for (int i = 0; i < 16; ++i)
  {
    matrix(i / 4, i % 4) = pose_data[i].as<double>();
  }
  // 进行 SVD 分解，核心目的是将可能因数值误差而不满足正交性的矩阵，
  // “强行”投影到最近的合法旋转群 SO(3) 上，
  // 确保程序能正确构造 Sophus::SE3d 且避免后续数值发散
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      matrix.topLeftCorner<3, 3>(), Eigen::ComputeFullU | Eigen::ComputeFullV
  );
  // 忽略奇异值，是投影的必然要求
  const Eigen::Matrix3d orthonormal_rotation
      = svd.matrixU() * svd.matrixV().transpose();
  return {Sophus::SO3d(orthonormal_rotation), matrix.topRightCorner<3, 1>()};
}

// 从文件 sensor.yaml 中读取：相机标定参数
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
  // 兼容两种畸变系数形式: EuRoC 标准 sequence [k1,k2,p1,p2] 与
  // 以 k1/k2/p1/p2 命名键的 map (部分自定义数据集采用, 如本测试数据集)
  static constexpr const char *kRadtanKeys[] = {"k1", "k2", "p1", "p2"};
  const int coefficient_count
      = std::min(calibration.distortion.cols,
                 static_cast<int>(std::size(kRadtanKeys)));
  if (distortion_coefficients.IsMap())
  {
    for (int i = 0; i < coefficient_count; ++i)
    {
      calibration.distortion.at<double>(0, i)
          = distortion_coefficients[kRadtanKeys[i]].as<double>();
    }
  }
  else
  {
    for (int i = 0; i < calibration.distortion.cols; ++i)
    {
      calibration.distortion.at<double>(0, i)
          = distortion_coefficients[i].as<double>();
    }
  }
  calibration.image_size = {resolution[0].as<int>(), resolution[1].as<int>()};
  calibration.body_from_camera = ReadBodyFromSensorPose(node);
  return calibration;
}

// 从文件 sensor.yaml 中读取：陀螺仪和加速度计的白噪声、随机游走
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

// IMU 测量值
struct ImuSample
{
  double time                         = 0;
  Eigen::Vector3d angular_velocity    = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
};

// 在两个相邻 IMU 测量值之间进行线性插值
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

// 图像帧 (双目模式左右路径均有效; 单目模式 right_image_path 为空)
struct StereoFrame
{
  double time               = 0; // 秒
  std::int64_t timestamp_ns = 0; // 原始纳秒时间戳 (用于外部单目估计查找)
  fs::path left_image_path;
  fs::path right_image_path;
};

// 加载所有 IMU 测量值
std::vector<ImuSample> LoadImuSamples(const fs::path &dataset_root)
{
  std::ifstream file(dataset_root / "imu0" / "data.csv");
  if (!file)
  {
    throw std::runtime_error(
        std::format("无法打开 '{}'.",
                    fs::absolute(dataset_root / "imu0" / "data.csv").string())
    );
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
    std::int64_t timestamp_ns = 0;
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

// 加载所有图像帧; 双目模式要求左右图同时存在, 单目模式右路径留空
std::vector<StereoFrame> LoadStereoFrames(const fs::path &dataset_root,
                                          VisionMode vision_mode)
{
  const fs::path left_camera_directory
      = vision_mode == VisionMode::kStereo
            ? dataset_root / "cam0"
            : SelectMonoCameraDirectory(dataset_root);
  const fs::path right_camera_directory = dataset_root / "cam1";
  std::ifstream file(left_camera_directory / "data.csv");
  if (!file)
  {
    throw std::runtime_error(
        std::format("无法打开 '{}'.",
                    fs::absolute(left_camera_directory / "data.csv").string())
    );
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
    std::int64_t timestamp_ns = 0;
    std::string file_name;
    if (!(stream >> timestamp_ns >> file_name))
    {
      continue;
    }
    const fs::path left_image_path = left_camera_directory / "data" / file_name;
    const fs::path right_image_path
        = vision_mode == VisionMode::kStereo
              ? right_camera_directory / "data" / file_name
              : fs::path{};
    if (fs::exists(left_image_path)
        && (vision_mode == VisionMode::kMono || fs::exists(right_image_path)))
    {
      frames.push_back(StereoFrame{.time
                                   = static_cast<double>(timestamp_ns) * 1e-9,
                                   .timestamp_ns     = timestamp_ns,
                                   .left_image_path  = left_image_path,
                                   .right_image_path = right_image_path});
    }
  }
  return frames;
}

// ================ 外部单目算法输出的帧间相对运动 (DatumFast) ================

// 时间戳匹配容差: 图像与外部估计来自同一 cam0 时间轴, 理论上应精确相等,
// 留 0.5 ms 容差以吸收浮点/导出误差
constexpr std::int64_t kMonocularTimestampToleranceNs = 500'000;

// 在按时间戳升序排列的外部单目估计中二分查找与 timestamp_ns 最接近的记录;
// 偏差超过容差时视为该帧无外部估计, 返回空
std::optional<FastVIO::DatumFast>
FindDatumFastByTimestamp(const std::vector<FastVIO::DatumFast> &data,
                         std::int64_t timestamp_ns)
{
  if (data.empty())
  {
    return std::nullopt;
  }
  const auto next = std::ranges::lower_bound(data, timestamp_ns, {},
                                             &FastVIO::DatumFast::timestamp_);
  const FastVIO::DatumFast *closest = nullptr;
  if (next != data.end())
  {
    closest = &*next;
  }
  if (next != data.begin())
  {
    const auto previous = std::prev(next);
    if (closest == nullptr
        || timestamp_ns - previous->timestamp_
               < closest->timestamp_ - timestamp_ns)
    {
      closest = &*previous;
    }
  }
  if (std::abs(closest->timestamp_ - timestamp_ns)
      > kMonocularTimestampToleranceNs)
  {
    return std::nullopt;
  }
  return *closest;
}

// 加载外部单目估计; 文件缺失时返回空表 (MonocularUpdate 整体退化为不启用)。
// rectified_from_raw_left: 原始左目系到矫正后左目系的旋转, 把 CSV 中在原始
// 左目系表达的轴角/平移方向变换到滤波器克隆所在的矫正后左目系
std::vector<FastVIO::DatumFast>
LoadMonocularEstimations(const fs::path &estimation_csv_path,
                         const Sophus::SO3d &rectified_from_raw_left)
{
  if (!fs::exists(estimation_csv_path))
  {
    std::println("外部单目估计文件不存在, 跳过 MonocularUpdate: '{}'",
                 fs::absolute(estimation_csv_path).string());
    return {};
  }
  if (std::ifstream probe{estimation_csv_path}; !probe)
  {
    throw std::runtime_error(
        std::format("外部单目估计文件无法读取 (请检查权限): '{}'.",
                    fs::absolute(estimation_csv_path).string())
    );
  }
  std::vector<FastVIO::DatumFast> data
      = FastVIO::DatumFast::Load(estimation_csv_path.string(),
                                 rectified_from_raw_left);
  std::ranges::sort(data, {}, &FastVIO::DatumFast::timestamp_);
  std::println("外部单目估计: {} 条记录, 来自 '{}'", data.size(),
               estimation_csv_path.string());
  return data;
}

// 真实轨迹数据
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

// 获取真实轨迹的第一帧数据的时间戳
double LoadGroundTruthStartTime(const fs::path &dataset_root)
{
  const fs::path csv_path
      = dataset_root / "state_groundtruth_estimate0" / "data.csv";
  std::ifstream file(csv_path);
  if (!file)
  {
    throw std::runtime_error(std::format("无法打开 '{}'.",
                                         fs::absolute(csv_path).string()));
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
    std::int64_t timestamp_ns = 0;
    if (stream >> timestamp_ns)
    {
      return static_cast<double>(timestamp_ns) * 1e-9;
    }
  }

  throw std::runtime_error("groundtruth 文件中没有有效记录: "
                           + csv_path.string());
}

// 查找最接近 IMU 数据的时间戳的真实轨迹数据
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
    std::int64_t timestamp_ns = 0;
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
  static constexpr double kWindowDuration      = 0.3; // 秒
  static constexpr std::size_t kMinSampleCount = 20;
  static constexpr double kGyroStdThreshold  = 0.015; // rad/s, 静止段约为其 1/5
  static constexpr double kAccelStdThreshold = 0.1;   // m/s^2, 飞行振动远超此值

  std::deque<ImuSample> window_;
};

// ==================== 相机几何预处理 (双目矫正 / 单目去畸变) ====================
// 抽象接口统一提供图像预处理、归一化坐标转换与标定参数,
// 使前端跟踪与滤波器量测模型对单目/双目透明
class CameraGeometry
{
public:
  virtual ~CameraGeometry() = default;

  // 预处理一帧图像对; 单目模式 raw_right 为空, out_right 不做处理
  virtual void Preprocess(const cv::Mat &raw_left, const cv::Mat &raw_right,
                          cv::Mat &out_left, cv::Mat &out_right) const = 0;
  virtual Eigen::Vector2d LeftPixelToNormalized(const cv::Point2f &pixel) const
      = 0;
  virtual Eigen::Vector2d RightPixelToNormalized(const cv::Point2f &pixel) const
      = 0;
  virtual cv::Point2f
  LeftNormalizedToPixel(const Eigen::Vector2d &normalized) const = 0;
  virtual double focal_length() const                            = 0;
  virtual double baseline() const            = 0; // 单目返回 0
  virtual const cv::Size &image_size() const = 0;
  // 双目: 矫正后左目系; 单目: 原始相机系
  virtual const Sophus::SE3d &body_from_camera() const = 0;
};

// ============================ 立体矫正 (双目) ============================
class StereoRectifier : public CameraGeometry
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

  void Preprocess(const cv::Mat &raw_left, const cv::Mat &raw_right,
                  cv::Mat &out_left, cv::Mat &out_right) const override
  {
    cv::remap(raw_left, out_left, left_map_x_, left_map_y_, cv::INTER_LINEAR);
    cv::remap(raw_right, out_right, right_map_x_, right_map_y_,
              cv::INTER_LINEAR);
  }

  Eigen::Vector2d LeftPixelToNormalized(const cv::Point2f &pixel) const override
  {
    return {(pixel.x - left_principal_point_.x()) / focal_length_,
            (pixel.y - left_principal_point_.y()) / focal_length_};
  }

  Eigen::Vector2d
  RightPixelToNormalized(const cv::Point2f &pixel) const override
  {
    return {(pixel.x - right_principal_point_.x()) / focal_length_,
            (pixel.y - right_principal_point_.y()) / focal_length_};
  }

  cv::Point2f
  LeftNormalizedToPixel(const Eigen::Vector2d &normalized) const override
  {
    return {static_cast<float>(normalized.x() * focal_length_
                               + left_principal_point_.x()),
            static_cast<float>(normalized.y() * focal_length_
                               + left_principal_point_.y())};
  }

  double focal_length() const override
  {
    return focal_length_;
  }
  double baseline() const override
  {
    return baseline_;
  }
  const cv::Size &image_size() const override
  {
    return image_size_;
  }
  const Sophus::SE3d &body_from_camera() const override
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

// ============================ 去畸变 (单目) ============================
class MonoUndistorter : public CameraGeometry
{
public:
  explicit MonoUndistorter(const CameraCalibration &calibration) :
    image_size_(calibration.image_size),
    body_from_camera_(calibration.body_from_camera)
  {
    // 恒等矫正旋转 + 原相机矩阵: 只去畸变不改变相机系, 量测仍在原始相机系表达
    cv::initUndistortRectifyMap(calibration.camera_matrix,
                                calibration.distortion,
                                cv::Mat::eye(3, 3, CV_64F),
                                calibration.camera_matrix, image_size_,
                                CV_32FC1, map_x_, map_y_);
    focal_length_    = calibration.camera_matrix.at<double>(0, 0);
    principal_point_ = {calibration.camera_matrix.at<double>(0, 2),
                        calibration.camera_matrix.at<double>(1, 2)};
  }

  void Preprocess(const cv::Mat &raw_left, const cv::Mat &, cv::Mat &out_left,
                  cv::Mat &) const override
  {
    cv::remap(raw_left, out_left, map_x_, map_y_, cv::INTER_LINEAR);
    // 单目无右图, 仅左目去畸变
  }

  Eigen::Vector2d LeftPixelToNormalized(const cv::Point2f &pixel) const override
  {
    return {(pixel.x - principal_point_.x()) / focal_length_,
            (pixel.y - principal_point_.y()) / focal_length_};
  }

  Eigen::Vector2d
  RightPixelToNormalized(const cv::Point2f &pixel) const override
  {
    return LeftPixelToNormalized(pixel); // 单目模式不会被调用
  }

  cv::Point2f
  LeftNormalizedToPixel(const Eigen::Vector2d &normalized) const override
  {
    return {static_cast<float>(normalized.x() * focal_length_
                               + principal_point_.x()),
            static_cast<float>(normalized.y() * focal_length_
                               + principal_point_.y())};
  }

  double focal_length() const override
  {
    return focal_length_;
  }
  double baseline() const override
  {
    return 0;
  }
  const cv::Size &image_size() const override
  {
    return image_size_;
  }
  const Sophus::SE3d &body_from_camera() const override
  {
    return body_from_camera_;
  }

private:
  cv::Size image_size_;
  cv::Mat map_x_, map_y_;
  double focal_length_             = 0;
  Eigen::Vector2d principal_point_ = Eigen::Vector2d::Zero();
  Sophus::SE3d body_from_camera_;
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
// 统一观测: 单目模式仅填充 left_normalized, 双目模式填充左右两目
struct Observation
{
  Eigen::Vector2d left_normalized  = Eigen::Vector2d::Zero();
  Eigen::Vector2d right_normalized = Eigen::Vector2d::Zero();
};

class FeatureTracker
{
public:
  struct TrackedFeature
  {
    FeatureId feature_id = -1;
    Observation observation;
  };

  FeatureTracker(const CameraGeometry &geometry, VisionMode vision_mode) :
    geometry_(geometry), vision_mode_(vision_mode)
  {
  }

  // current_camera_from_previous_camera: 由外部姿态估计(陀螺积分)给出的两帧间相机旋转
  // 单目模式 rectified_right 为空 Mat
  std::vector<TrackedFeature>
  Track(const cv::Mat &rectified_left, const cv::Mat &rectified_right,
        const Sophus::SO3d &current_camera_from_previous_camera)
  {
    // 构造金字塔
    std::vector<cv::Mat> left_pyramid;
    cv::buildOpticalFlowPyramid(rectified_left, left_pyramid, kLkWindowSize,
                                kPyramidLevels);

    // 角点检测 + 时序数据关联
    std::vector<cv::Point2f> left_points;
    std::vector<FeatureId> feature_ids;
    TrackFromPreviousFrame(left_pyramid, current_camera_from_previous_camera,
                           left_points, feature_ids);
    DetectNewFastCorners(rectified_left, left_points, feature_ids);

    std::vector<TrackedFeature> result;
    std::vector<cv::Point2f> kept_points;
    std::vector<FeatureId> kept_ids;
    if (vision_mode_ == VisionMode::kStereo)
    {
      // 双目数据关联: 左→右 LK 匹配 + 极线/视差检查
      std::vector<cv::Mat> right_pyramid;
      cv::buildOpticalFlowPyramid(rectified_right, right_pyramid, kLkWindowSize,
                                  kPyramidLevels);
      std::vector<cv::Point2f> right_points = left_points;
      std::vector<uchar> status;
      std::vector<float> error;
      cv::calcOpticalFlowPyrLK(
          left_pyramid, right_pyramid, left_points, right_points, status, error,
          kLkWindowSize, kPyramidLevels,
          {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01},
          cv::OPTFLOW_USE_INITIAL_FLOW
      );
      for (std::size_t i = 0; i < left_points.size(); ++i)
      {
        const double vertical_error
            = std::abs(left_points[i].y - right_points[i].y);
        const double disparity = left_points[i].x - right_points[i].x;
        // 检查是否成功匹配、视差是否足够大、极线是否水平
        if (!status[i] || !InsideImage(right_points[i])
            || vertical_error > kMaxEpipolarError || disparity < kMinDisparity)
        {
          continue;
        }
        result.push_back(TrackedFeature{
            feature_ids[i],
            Observation{geometry_.LeftPixelToNormalized(left_points[i]),
                        geometry_.RightPixelToNormalized(right_points[i])},
        });
        kept_points.push_back(left_points[i]);
        kept_ids.push_back(feature_ids[i]);
      }
    }
    else
    {
      // 单目: 仅左目归一化平面观测
      result.reserve(left_points.size());
      for (std::size_t i = 0; i < left_points.size(); ++i)
      {
        result.push_back(TrackedFeature{
            feature_ids[i],
            Observation{geometry_.LeftPixelToNormalized(left_points[i]),
                        Eigen::Vector2d::Zero()},
        });
        kept_points.push_back(left_points[i]);
        kept_ids.push_back(feature_ids[i]);
      }
    }

    // 更新上一帧的左目像素点、特征编号、金字塔
    previous_left_points_ = std::move(kept_points);
    previous_feature_ids_ = std::move(kept_ids);
    previous_left_pyramid_
        = std::move(left_pyramid); // 复用: 下一帧的"上一帧金字塔"

    return result;
  }

  // 删除滤波器已消费的特征: 停止跟踪旧 id, 其占据的网格下一帧可重新提取为新特征
  void DropFeatures(std::span<const FeatureId> feature_ids)
  {
    if (feature_ids.empty())
    {
      // 没有需要删除的路标点，直接返回
      return;
    }
    // 去重
    const std::set<FeatureId> ids_to_drop(feature_ids.begin(),
                                          feature_ids.end());
    std::vector<cv::Point2f> kept_points;
    std::vector<FeatureId> kept_ids;
    for (std::size_t i = 0; i < previous_feature_ids_.size(); ++i)
    {
      if (ids_to_drop.contains(previous_feature_ids_[i]))
      {
        continue;
      }
      kept_points.push_back(previous_left_points_[i]);
      kept_ids.push_back(previous_feature_ids_[i]);
    }
    // 更新路标点和特征编号
    previous_left_points_ = std::move(kept_points);
    previous_feature_ids_ = std::move(kept_ids);
  }

private:
  static constexpr std::size_t kMaxFeatureCount = 200;
  static constexpr int kFastThreshold           = 20;
  static constexpr double kMinFeatureDistance   = 25.0;
  static constexpr double kMaxEpipolarError     = 3.0;
  static constexpr double kMinDisparity         = 0.5;
  static constexpr int kImageBorder             = 5;
  static constexpr int kPyramidLevels           = 3;
  inline static const cv::Size kLkWindowSize{21, 21};

  bool InsideImage(const cv::Point2f &point) const
  {
    const cv::Size &image_size = geometry_.image_size();
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
    const Eigen::Vector2d normalized = geometry_.LeftPixelToNormalized(pixel);
    const Eigen::Vector3d rotated_direction
        = current_camera_from_previous_camera
          * Eigen::Vector3d(normalized.x(), normalized.y(), 1.0);
    if (rotated_direction.z() < 0.1)
    {
      return pixel;
    }
    return geometry_.LeftNormalizedToPixel(
        {rotated_direction.x() / rotated_direction.z(),
         rotated_direction.y() / rotated_direction.z()}
    );
  }

  // 利用 LK 光流法，在两个视图之间进行数据关联
  void TrackFromPreviousFrame(
      const std::vector<cv::Mat> &current_left_pyramid,
      const Sophus::SO3d &current_camera_from_previous_camera,
      std::vector<cv::Point2f> &left_points, std::vector<FeatureId> &feature_ids
  )
  {
    if (previous_left_pyramid_.empty() || previous_left_points_.empty())
    {
      return;
    }

    // 准备初步猜测
    std::vector<cv::Point2f> current_points(previous_left_points_.size());
    for (std::size_t i = 0; i < previous_left_points_.size(); ++i)
    {
      current_points[i]
          = PredictPixelWithRotation(previous_left_points_[i],
                                     current_camera_from_previous_camera);
    }

    // 执行 LK 光流
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(
        previous_left_pyramid_, current_left_pyramid, previous_left_points_,
        current_points, status, error, kLkWindowSize, kPyramidLevels,
        {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01},
        cv::OPTFLOW_USE_INITIAL_FLOW
    );

    // 筛选匹配点
    std::vector<cv::Point2f> matched_previous, matched_current;
    std::vector<FeatureId> matched_ids;
    for (std::size_t i = 0; i < previous_left_points_.size(); ++i)
    {
      if (!status[i] || !InsideImage(current_points[i]))
      {
        continue;
      }
      matched_previous.push_back(previous_left_points_[i]);
      matched_current.push_back(current_points[i]);
      matched_ids.push_back(previous_feature_ids_[i]);
    }

    // 筛选内点
    std::vector<uchar> inlier_mask(matched_current.size(), 1);
    if (matched_current.size() >= 8)
    {
      cv::findFundamentalMat(matched_previous, matched_current, cv::FM_RANSAC,
                             1.0, 0.99, inlier_mask);
    }
    for (std::size_t i = 0; i < matched_current.size(); ++i)
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
                            std::vector<FeatureId> &feature_ids)
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
    // 按照响应值（response）降序排列，保留最显著最靠前若干个角点，提取最优角点
    std::ranges::sort(keypoints, std::greater<>{}, &cv::KeyPoint::response);

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

  const CameraGeometry &geometry_;
  VisionMode vision_mode_ = VisionMode::kMono;
  std::vector<cv::Mat> previous_left_pyramid_;
  std::vector<cv::Point2f> previous_left_points_;
  std::vector<FeatureId> previous_feature_ids_;
  FeatureId next_feature_id_ = 0;
};

// ==================== 特征三角化 (Ceres 双目/单目重投影) ====================
struct StereoReprojectionCost
{
  StereoReprojectionCost(const Sophus::SE3d &camera_from_world,
                         const Observation &observation,
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
  Observation measurement;
  double baseline;
};

struct MonoReprojectionCost
{
  MonoReprojectionCost(const Sophus::SE3d &camera_from_world,
                       const Observation &observation) :
    rotation(camera_from_world.rotationMatrix()),
    translation(camera_from_world.translation()), measurement(observation)
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
    return true;
  }

  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  Observation measurement;
};

// ============================ MSCKF 滤波器 ============================

// 路标点对应的不同帧中的像素点
struct FeatureTrack
{
  FrameId feature_id = -1;
  std::map<FrameId, Observation> observations_by_frame;
};

struct CameraClone
{
  FrameId frame_id = -1;
  Sophus::SO3d world_from_camera_rotation;
  Eigen::Vector3d position_in_world = Eigen::Vector3d::Zero();
  Sophus::SO3d null_rotation; // OC: 增广时刻的首次估计, EKF 更新不改动
  Eigen::Vector3d null_position = Eigen::Vector3d::Zero();
};

class Msckf
{
public:
  static constexpr int kImuErrorDim
      = 19; // [姿态3 位置3 速度3 陀螺零偏3 加计零偏3 时间偏移1 重力3]
  static constexpr int kTimeOffsetIndex              = 15;
  static constexpr int kGravityIndex                 = 16;
  static constexpr int kCloneErrorDim                = 6; // [姿态3 位置3]
  static constexpr std::size_t kMaxCloneCount        = 11;
  static constexpr std::size_t kMinTrackLength       = 3;
  static constexpr int kMaxUpdateRows                = 600;
  static constexpr double kInitialTimeOffsetVariance = 2.5e-5; // (5 ms)^2
  static constexpr double kZuptVelocitySigma = 0.02; // m/s, 零速伪量测噪声
  static constexpr double kRedundantRotationThreshold
      = 0.2618; // rad, 冗余克隆判定 (15°)
  static constexpr double kRedundantTranslationThreshold = 0.4; // m
  static constexpr double kMonocularRotationSigma
      = 0.005; // rad, 单目旋转量测噪声 (高精度)
  static constexpr double kMonocularDirectionSigma
      = 0.1; // rad, 平移方向切平面噪声 (低精度)
  static constexpr double kMinBaselineForDirection
      = 0.01; // m, 基线过短时方向量测退化
  static constexpr double kHuberLossThreshold
      = 0.01; // 三角化重投影残差鲁棒核阈值 (归一化平面)
  static constexpr double kMinParallaxRadians
      = 0.01; // 单目三角化视差角门限: 首末帧光线夹角过小时深度不可观
  static constexpr int kStereoResidualDim
      = 4;                                   // 双目每观测残差维数 (左 2 + 右 2)
  static constexpr int kMonoResidualDim = 2; // 单目每观测残差维数 (左 2)

  // 单目模式 baseline 传 0, 三角化/量测雅可比按 vision_mode 走两套模型
  Msckf(const Sophus::SE3d &body_from_camera, double baseline,
        double focal_length, const ImuNoiseParameters &imu_noise,
        double initial_time_offset, VisionMode vision_mode) :
    body_from_camera_(body_from_camera), baseline_(baseline),
    pixel_noise_normalized_(1.5 / focal_length), imu_noise_(imu_noise),
    time_offset_(initial_time_offset), vision_mode_(vision_mode)
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

    PropagateImuCovariance(dt, gyro, accel, rotation_matrix);
    SaveNullLinearizationPoint();
  }

  // ZUPT 也视作测量函数，交由滤波器处理
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

  // 更新协方差矩阵，增广相机克隆。
  // 每帧图像到来时先调用本函数, 再依次执行 MonocularUpdate / ProcessFrame。
  void AugmentCameraClone(FrameId frame_id)
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

  // 融合外部单目算法输出的帧间相对运动。
  // 调用时机: 当前帧克隆增广 (AugmentCameraClone) 之后、前端跟踪与 ProcessFrame
  // 之前, 修正后的姿态可直接改善陀螺辅助光流初值与单目量测线性化点。
  // 约定 (矫正后左相机系):
  //   rotation_vector:       上一帧到当前帧的相对旋转轴角, R_上帧_from_当帧 = Exp(rVec)
  //   translation_direction: 当前光心相对上一帧光心的平移方向, 在上一帧相机系中表达 (无尺度)
  // 旋转做 3 维流形残差 (高权重), 平移只约束量测方向切平面上的 2 维分量 (低权重, 尺度自然
  // 不受约束); 两者按顺序 EKF 更新。相对位姿量测对全局平移/绕重力偏航天然满足 H·N = 0,
  // 无需 OC 投影。上一帧克隆若缺失 (如上一帧图像读取失败) 则跳过, 返回 false。
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
    const bool rotation_applied
        = ApplyMonocularRotationUpdate(previous_index, current_index,
                                       rotation_vector);
    // 平移方向在旋转更新后的最新状态上重新线性化
    const bool direction_applied
        = ApplyMonocularDirectionUpdate(previous_index, current_index,
                                        translation_direction);
    return rotation_applied || direction_applied;
  }

  // 每帧 ProcessFrame 的返回值
  struct FilterResult
  {
    std::vector<FeatureId> dropped_feature_ids;
    std::vector<Eigen::Vector3d> new_map_points; // world 系已三角化点
  };

  // 前置条件: 本帧克隆已通过 AugmentCameraClone(frame_id) 增广。
  // 返回 FilterResult: dropped_feature_ids 为空 (MSCKF 无 SLAM 外点概念),
  // new_map_points 为本帧成功三角化并通过量测门限的路标点世界坐标。
  FilterResult
  ProcessFrame(FrameId frame_id,
               const std::vector<FeatureTracker::TrackedFeature> &tracked)
  {
    FilterResult result;

    std::set<FeatureId> visible_ids;
    for (const auto &feature : tracked)
    {
      visible_ids.insert(feature.feature_id);
      FeatureTrack &track = active_tracks_[feature.feature_id];
      track.feature_id    = feature.feature_id;
      track.observations_by_frame[frame_id] = feature.observation;
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
    result.new_map_points = UpdateWithTracks(finished_tracks);

    if (clones_.size() > kMaxCloneCount)
    {
      auto prune_points = PruneClonesAndAbsorbObservations();
      result.new_map_points.insert(
          result.new_map_points.end(),
          std::make_move_iterator(prune_points.begin()),
          std::make_move_iterator(prune_points.end())
      );
    }
    return result;
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
  std::size_t clone_count() const
  {
    return clones_.size();
  }

private:
  // 误差状态向量的维数: IMU块 + 克隆块，路标点不进入状态
  int StateDim() const
  {
    return kImuErrorDim + kCloneErrorDim * static_cast<int>(clones_.size());
  }

  // 克隆相机的索引
  int CloneStateIndex(int clone_index) const
  {
    return kImuErrorDim + kCloneErrorDim * clone_index;
  }

  // 协方差对称化
  void Symmetrize()
  {
    covariance_ = ((covariance_ + covariance_.transpose()) * 0.5).eval();
  }

  // IMU 传播的误差状态转移矩阵 (含 OC 一致性修正)
  Eigen::Matrix<double, kImuErrorDim, kImuErrorDim>
  BuildImuTransitionMatrix(double dt, const Eigen::Vector3d &gyro,
                           const Eigen::Vector3d &accel,
                           const Eigen::Matrix3d &rotation_matrix) const
  {
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
    return transition;
  }

  // IMU 传播的离散过程噪声
  Eigen::Matrix<double, kImuErrorDim, kImuErrorDim>
  BuildImuDiscreteNoise(double dt, const Eigen::Matrix3d &rotation_matrix) const
  {
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
    return noise_jacobian * continuous_noise * noise_jacobian.transpose() * dt;
  }

  // 用状态转移矩阵与过程噪声传播协方差 (IMU 块及其与克隆的互协方差)
  void PropagateImuCovariance(double dt, const Eigen::Vector3d &gyro,
                              const Eigen::Vector3d &accel,
                              const Eigen::Matrix3d &rotation_matrix)
  {
    const Eigen::Matrix<double, kImuErrorDim, kImuErrorDim> transition
        = BuildImuTransitionMatrix(dt, gyro, accel, rotation_matrix);
    const Eigen::Matrix<double, kImuErrorDim, kImuErrorDim> discrete_noise
        = BuildImuDiscreteNoise(dt, rotation_matrix);

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
  }

  // 根据帧编号，检索克隆在队列中的序号
  int FindCloneIndex(FrameId frame_id) const
  {
    for (std::size_t i = 0; i < clones_.size(); ++i)
    {
      if (clones_[i].frame_id == frame_id)
      {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  // 三角化，计算路标点三维坐标 (双目: 视差初值 + 4 残差; 单目: 视差角门限 + 线性初值 + 2 残差)
  std::optional<Eigen::Vector3d>
  TriangulateFeature(const FeatureTrack &track) const
  {
    std::vector<std::pair<Sophus::SE3d, Observation>> observations;
    for (const auto &[frame_id, observation] : track.observations_by_frame)
    {
      // 查找克隆在队列中的序号
      const int clone_index = FindCloneIndex(frame_id);
      if (clone_index < 0)
      {
        // 没有找到克隆
        continue;
      }
      const CameraClone &clone = clones_[clone_index];
      // 克隆对应的相机位姿变换
      const Sophus::SE3d world_from_camera(clone.world_from_camera_rotation,
                                           clone.position_in_world);
      observations.emplace_back(world_from_camera.inverse(), observation);
    }

    if (observations.size() < kMinTrackLength)
    {
      // 成功跟踪次数过少
      return std::nullopt;
    }

    Eigen::Vector3d world_point;
    int residual_dimension = kMonoResidualDim;
    if (vision_mode_ == VisionMode::kStereo)
    {
      // 双目: 首帧视差给出深度初值
      const auto &[first_camera_from_world, first_observation]
          = observations.front();
      const double disparity = first_observation.left_normalized.x()
                               - first_observation.right_normalized.x();
      if (disparity < 1e-4)
      {
        // 视差过小
        return std::nullopt;
      }
      const double depth = std::clamp(baseline_ / disparity, 0.2, 50.0);
      world_point
          = first_camera_from_world.inverse()
            * Eigen::Vector3d(first_observation.left_normalized.x() * depth,
                              first_observation.left_normalized.y() * depth,
                              depth);
      residual_dimension = kStereoResidualDim;
    }
    else
    {
      // 单目: 首末两帧光线夹角过小时深度不可观, 拒绝三角化
      const auto &[first_camera_from_world, first_observation]
          = observations.front();
      const auto &[last_camera_from_world, last_observation]
          = observations.back();
      const Eigen::Vector3d first_ray_direction
          = first_camera_from_world.rotationMatrix().transpose()
            * Eigen::Vector3d(first_observation.left_normalized.x(),
                              first_observation.left_normalized.y(), 1.0);
      const Eigen::Vector3d last_ray_direction
          = last_camera_from_world.rotationMatrix().transpose()
            * Eigen::Vector3d(last_observation.left_normalized.x(),
                              last_observation.left_normalized.y(), 1.0);
      const double parallax_cosine
          = first_ray_direction.dot(last_ray_direction)
            / (first_ray_direction.norm() * last_ray_direction.norm());
      if (std::acos(std::clamp(parallax_cosine, -1.0, 1.0))
          < kMinParallaxRadians)
      {
        return std::nullopt;
      }

      // 两视图线性最小二乘: d1·R1ᵀu1 - d2·R2ᵀu2 = R1ᵀt1 - R2ᵀt2, 解出首末帧深度
      Eigen::Matrix<double, 3, 2> depth_system;
      depth_system.col(0) = first_ray_direction;
      depth_system.col(1) = -last_ray_direction;
      const Eigen::Vector3d depth_rhs
          = first_camera_from_world.rotationMatrix().transpose()
                * first_camera_from_world.translation()
            - last_camera_from_world.rotationMatrix().transpose()
                  * last_camera_from_world.translation();
      const Eigen::Vector2d depths
          = depth_system.colPivHouseholderQr().solve(depth_rhs);
      if (depths.x() < 0.05 || depths.y() < 0.05)
      {
        // 深度为负或过近
        return std::nullopt;
      }
      const Eigen::Vector3d world_from_first_view
          = depths.x() * first_ray_direction
            - first_camera_from_world.rotationMatrix().transpose()
                  * first_camera_from_world.translation();
      const Eigen::Vector3d world_from_last_view
          = depths.y() * last_ray_direction
            - last_camera_from_world.rotationMatrix().transpose()
                  * last_camera_from_world.translation();
      world_point = 0.5 * (world_from_first_view + world_from_last_view);
    }

    return RefineAndValidateWorldPoint(std::move(world_point), observations,
                                       residual_dimension);
  }

  // 对路标点做 Ceres 重投影优化并执行质量/深度门限检查 (双目 4 残差 / 单目 2 残差)
  std::optional<Eigen::Vector3d> RefineAndValidateWorldPoint(
      Eigen::Vector3d world_point,
      const std::vector<std::pair<Sophus::SE3d, Observation>> &observations,
      int residual_dimension
  ) const
  {
    ceres::Problem::Options problem_options;
    // Ceres (含 2.2) 的 AddResidualBlock 只接受裸指针且默认接管所有权;
    // 这里显式设置 DO_NOT_TAKE_OWNERSHIP, 由 unique_ptr/栈对象管理生命周期,
    // 避免"裸 new 无对应 delete"的隐患
    problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // 对路标点进行非线性优化
    ceres::Problem problem{problem_options};

    ceres::HuberLoss huber_loss{kHuberLossThreshold};
    // 两个容器声明在分支外: Ceres 以 DO_NOT_TAKE_OWNERSHIP 持有裸指针,
    // 必须在 Solve 期间保持存活 (分支内声明会提前析构造成悬垂)
    using StereoCostFunction
        = ceres::AutoDiffCostFunction<StereoReprojectionCost, 4, 3>;
    using MonoCostFunction
        = ceres::AutoDiffCostFunction<MonoReprojectionCost, 2, 3>;
    std::vector<std::unique_ptr<StereoCostFunction>> stereo_cost_functions;
    std::vector<std::unique_ptr<MonoCostFunction>> mono_cost_functions;
    if (residual_dimension == kStereoResidualDim)
    {
      stereo_cost_functions.reserve(observations.size());
      for (const auto &[camera_from_world, observation] : observations)
      {
        auto functor
            = std::make_unique<StereoReprojectionCost>(camera_from_world,
                                                       observation, baseline_);
        // AutoDiffCostFunction 内部以 unique_ptr 持有 functor 并接管所有权
        stereo_cost_functions.push_back(
            std::make_unique<StereoCostFunction>(functor.release())
        );
        problem.AddResidualBlock(stereo_cost_functions.back().get(),
                                 &huber_loss, world_point.data());
      }
    }
    else
    {
      mono_cost_functions.reserve(observations.size());
      for (const auto &[camera_from_world, observation] : observations)
      {
        auto functor = std::make_unique<MonoReprojectionCost>(camera_from_world,
                                                              observation);
        // AutoDiffCostFunction 内部以 unique_ptr 持有 functor 并接管所有权
        mono_cost_functions.push_back(
            std::make_unique<MonoCostFunction>(functor.release())
        );
        problem.AddResidualBlock(mono_cost_functions.back().get(), &huber_loss,
                                 world_point.data());
      }
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
      // 非线性优化失败
      return std::nullopt;
    }

    const double mean_squared_error
        = 2.0 * summary.final_cost
          / static_cast<double>(residual_dimension * observations.size());
    if (std::sqrt(mean_squared_error) > 10.0 * pixel_noise_normalized_)
    {
      // 误差过大
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
  // 双目模式每观测 4 行 (左 2 + 右 2), 单目模式每观测 2 行
  bool BuildTrackJacobians(const FeatureTrack &track,
                           const Eigen::Vector3d &world_point,
                           const std::set<FrameId> *frame_filter,
                           Eigen::MatrixXd &state_jacobian,
                           Eigen::MatrixXd &point_jacobian,
                           Eigen::VectorXd &residual) const
  {
    const int residuals_per_observation = vision_mode_ == VisionMode::kStereo
                                              ? kStereoResidualDim
                                              : kMonoResidualDim;
    const int max_rows = residuals_per_observation
                         * static_cast<int>(track.observations_by_frame.size());
    state_jacobian     = Eigen::MatrixXd::Zero(max_rows, StateDim());
    point_jacobian     = Eigen::MatrixXd::Zero(max_rows, 3);
    residual           = Eigen::VectorXd::Zero(max_rows);

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

      const int column = CloneStateIndex(clone_index);
      if (vision_mode_ == VisionMode::kStereo)
      {
        // 双目: 左目 + 右目 (沿基线平移 -baseline) 的四维投影雅可比
        Eigen::Matrix<double, 4, 3> projection_jacobian;
        projection_jacobian << 1 / z, 0, -x / (z * z), 0, 1 / z, -y / (z * z),
            1 / z, 0, -(x - baseline_) / (z * z), 0, 1 / z, -y / (z * z);

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
        point_jacobian.block<4, 3>(row, 0)
            = -clone_jacobian_block.rightCols<3>();
        residual.segment<4>(row) << observation.left_normalized.x() - x / z,
            observation.left_normalized.y() - y / z,
            observation.right_normalized.x() - (x - baseline_) / z,
            observation.right_normalized.y() - y / z;
        row += kStereoResidualDim;
      }
      else
      {
        // 单目: 仅左目的二维投影雅可比
        Eigen::Matrix<double, 2, 3> projection_jacobian;
        projection_jacobian << 1 / z, 0, -x / (z * z), 0, 1 / z, -y / (z * z);

        Eigen::Matrix<double, 2, 6> clone_jacobian_block;
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

        state_jacobian.block<2, 6>(row, column) = clone_jacobian_block;
        point_jacobian.block<2, 3>(row, 0)
            = -clone_jacobian_block.rightCols<3>();
        residual.segment<2>(row) << observation.left_normalized.x() - x / z,
            observation.left_normalized.y() - y / z;
        row += kMonoResidualDim;
      }
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
                                       const std::set<FrameId> *frame_filter
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

  // 单目相对旋转量测: r = Log(R̂_相对⁻¹·Exp(rVec)), H_θ上 = -R̂_相对⁻¹, H_θ当 = I
  bool ApplyMonocularRotationUpdate(int previous_index, int current_index,
                                    const Eigen::Vector3d &rotation_vector)
  {
    const CameraClone &current_clone  = clones_[current_index];
    const CameraClone &previous_clone = clones_[previous_index];

    // --- 旋转: r = Log(R̂_相对⁻¹·Exp(rVec)), H_θ上 = -R̂_相对⁻¹, H_θ当 = I ---
    const Sophus::SO3d predicted_relative_rotation
        = previous_clone.world_from_camera_rotation.inverse()
          * current_clone.world_from_camera_rotation;
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, StateDim());
    jacobian.block<3, 3>(0, CloneStateIndex(previous_index))
        = -predicted_relative_rotation.inverse().matrix();
    jacobian.block<3, 3>(0, CloneStateIndex(current_index))
        = Eigen::Matrix3d::Identity();
    Eigen::VectorXd residual = (predicted_relative_rotation.inverse()
                                * Sophus::SO3d::exp(rotation_vector))
                                   .log();

    const double noise_variance
        = kMonocularRotationSigma * kMonocularRotationSigma;
    if (!PassesChiSquareGate(jacobian, residual, noise_variance))
    {
      return false;
    }
    ApplyEkfUpdate(std::move(jacobian), std::move(residual), noise_variance);
    return true;
  }

  // 单目平移方向量测: 只约束量测方向切平面上的 2 维分量 (尺度自然不受约束)
  bool
  ApplyMonocularDirectionUpdate(int previous_index, int current_index,
                                const Eigen::Vector3d &translation_direction)
  {
    // --- 平移方向: 旋转更新后的最新状态上重新线性化 (克隆引用随更新自动生效) ---
    const double direction_norm = translation_direction.norm();
    if (direction_norm < 1e-6)
    {
      return false;
    }
    const Eigen::Vector3d measured_direction
        = translation_direction / direction_norm;

    const CameraClone &previous_clone = clones_[previous_index];
    const CameraClone &current_clone  = clones_[current_index];
    const Eigen::Matrix3d previous_camera_from_world
        = previous_clone.world_from_camera_rotation.inverse().matrix();
    const Eigen::Vector3d baseline_in_previous_camera
        = previous_camera_from_world
          * (current_clone.position_in_world
             - previous_clone.position_in_world);
    const double baseline_norm = baseline_in_previous_camera.norm();
    if (baseline_norm < kMinBaselineForDirection)
    {
      return false; // 悬停: 方向退化
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

    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(2, StateDim());
    jacobian.block<2, 3>(0, CloneStateIndex(previous_index))
        = direction_jacobian * Sophus::SO3d::hat(baseline_in_previous_camera);
    jacobian.block<2, 3>(0, CloneStateIndex(previous_index) + 3)
        = -direction_jacobian * previous_camera_from_world;
    jacobian.block<2, 3>(0, CloneStateIndex(current_index) + 3)
        = direction_jacobian * previous_camera_from_world;
    Eigen::VectorXd residual = -(tangent_projector * predicted_direction);

    const double noise_variance
        = kMonocularDirectionSigma * kMonocularDirectionSigma;
    if (!PassesChiSquareGate(jacobian, residual, noise_variance))
    {
      return false;
    }
    ApplyEkfUpdate(std::move(jacobian), std::move(residual), noise_variance);
    return true;
  }

  // 把若干量测块按行堆叠成单个大雅可比后执行一次 EKF 更新
  void ApplyStackedEkfUpdate(std::vector<Eigen::MatrixXd> jacobian_blocks,
                             std::vector<Eigen::VectorXd> residual_blocks,
                             int total_rows, double noise_variance)
  {
    if (jacobian_blocks.empty())
    {
      return;
    }
    Eigen::MatrixXd stacked_jacobian(total_rows, StateDim());
    Eigen::VectorXd stacked_residual(total_rows);
    int row = 0;
    for (std::size_t i = 0; i < jacobian_blocks.size(); ++i)
    {
      stacked_jacobian.middleRows(row, jacobian_blocks[i].rows())
          = jacobian_blocks[i];
      stacked_residual.segment(row, residual_blocks[i].size())
          = residual_blocks[i];
      row += static_cast<int>(jacobian_blocks[i].rows());
    }
    ApplyEkfUpdate(std::move(stacked_jacobian), std::move(stacked_residual),
                   noise_variance);
  }

  std::vector<Eigen::Vector3d>
  UpdateWithTracks(const std::vector<FeatureTrack> &finished_tracks)
  {
    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    std::vector<Eigen::MatrixXd> jacobian_blocks;
    std::vector<Eigen::VectorXd> residual_blocks;
    std::vector<Eigen::Vector3d> map_points;
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
      map_points.push_back(*world_point);
      if (total_rows >= kMaxUpdateRows)
      {
        break;
      }
    }
    ApplyStackedEkfUpdate(std::move(jacobian_blocks),
                          std::move(residual_blocks), total_rows,
                          vision_noise_variance);
    return map_points;
  }

  // 更新名义状态向量、误差状态向量及其协方差矩阵
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

    // 新息协方差
    Eigen::MatrixXd innovation_covariance
        = jacobian * covariance_ * jacobian.transpose();
    innovation_covariance.diagonal().array() += noise_variance;
    // 卡尔曼增益
    const Eigen::MatrixXd kalman_gain = innovation_covariance.ldlt()
                                            .solve(jacobian * covariance_)
                                            .transpose();
    // 误差状态的修正项
    const Eigen::VectorXd correction = kalman_gain * residual;

    // 将误差状态注入名义状态
    world_from_imu_rotation_ = world_from_imu_rotation_
                               * Sophus::SO3d::exp(correction.segment<3>(0));
    imu_position_ += correction.segment<3>(3);
    imu_velocity_ += correction.segment<3>(6);
    gyro_bias_ += correction.segment<3>(9);
    accel_bias_ += correction.segment<3>(12);
    time_offset_ += correction(kTimeOffsetIndex);
    gravity_in_world_ += correction.segment<3>(kGravityIndex);
    for (std::size_t i = 0; i < clones_.size(); ++i)
    {
      const int base = CloneStateIndex(static_cast<int>(i));
      clones_[i].world_from_camera_rotation
          = clones_[i].world_from_camera_rotation
            * Sophus::SO3d::exp(correction.segment<3>(base));
      clones_[i].position_in_world += correction.segment<3>(base + 3);
    }

    // 使用 Joseph 稳定形式，更新误差状态协方差矩阵
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

  void RemoveCloneAt(int clone_index)
  {
    RemoveCovarianceBlock(CloneStateIndex(clone_index), kCloneErrorDim);
    clones_.erase(clones_.begin() + clone_index);
  }

  // 冗余克隆选择 (MSCKF-VIO): 相对参考帧运动小的近期克隆优先, 否则删最老的; 每次选 2 个
  std::vector<int> SelectCloneIndicesToRemove() const
  {
    static constexpr int count_clones_to_remove = 2;

    // 克隆队列长度
    const int count              = static_cast<int>(clones_.size());
    const CameraClone &key_clone = clones_[count - 4];
    int moving_candidate         = count - 3;
    int oldest_candidate         = 0;
    std::vector<int> remove_indices;
    for (int i = 0; i < count_clones_to_remove; ++i)
    {
      // 计算候选克隆的相对旋转、位移
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
        // 候选克隆的相对运动未达阈值，标记为待删除
        remove_indices.push_back(moving_candidate++);
      }
      else
      {
        // 保留候选克隆，将最老克隆标记为待删除
        remove_indices.push_back(oldest_candidate++);
      }
    }
    std::ranges::sort(remove_indices);
    return remove_indices;
  }

  std::vector<Eigen::Vector3d> PruneClonesAndAbsorbObservations()
  {
    const std::vector<int> remove_indices = SelectCloneIndicesToRemove();
    std::set<FrameId> removed_frame_ids;
    for (const int index : remove_indices)
    {
      removed_frame_ids.insert(clones_[index].frame_id);
    }

    auto map_points = AbsorbObservationsOnRemovedClones(removed_frame_ids);

    for (const int index : std::views::reverse(remove_indices))
    {
      RemoveCloneAt(index);
    }
    return map_points;
  }

  // 其余轨迹只吸收被删克隆上的观测 (>=2 帧才有零空间余量), 轨迹保活
  // 返回通过量测门限的已三角化路标点世界坐标
  std::vector<Eigen::Vector3d>
  AbsorbObservationsOnRemovedClones(const std::set<FrameId> &removed_frame_ids)
  {
    const double vision_noise_variance
        = pixel_noise_normalized_ * pixel_noise_normalized_;
    std::vector<Eigen::MatrixXd> jacobian_blocks;
    std::vector<Eigen::VectorXd> residual_blocks;
    std::vector<Eigen::Vector3d> map_points;
    int total_rows = 0;
    for (auto it = active_tracks_.begin(); it != active_tracks_.end();)
    {
      FeatureTrack &track = it->second;
      std::vector<FrameId> involved_frames;
      for (const FrameId removed_frame_id : removed_frame_ids)
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
            map_points.push_back(*world_point);
          }
        }
      }
      for (const FrameId involved_frame_id : involved_frames)
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

    ApplyStackedEkfUpdate(std::move(jacobian_blocks),
                          std::move(residual_blocks), total_rows,
                          vision_noise_variance);
    return map_points;
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
  double baseline_               = 0; // 双目基线, 单目模式为 0
  double pixel_noise_normalized_ = 0;
  ImuNoiseParameters imu_noise_;

  Sophus::SO3d world_from_imu_rotation_;
  Eigen::Vector3d imu_position_     = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_velocity_     = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_        = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias_       = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_in_world_ = Eigen::Vector3d(0, 0, -kGravity);
  double time_offset_     = 0; // 图像时刻 + time_offset_ = 对应的 IMU 时刻
  VisionMode vision_mode_ = VisionMode::kMono;
  double imu_time_        = 0;
  ImuSample last_imu_;
  bool has_last_imu_ = false;

  Sophus::SO3d null_rotation_;
  Eigen::Vector3d null_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d null_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d null_gravity_  = Eigen::Vector3d(0, 0, -kGravity);

  std::deque<CameraClone> clones_;
  Eigen::MatrixXd covariance_
      = Eigen::MatrixXd::Zero(kImuErrorDim, kImuErrorDim);
  std::unordered_map<FeatureId, FeatureTrack> active_tracks_;
};

// ============================ 主流程 ============================

void PrintCalibrationSummary(const CameraCalibration &calibration,
                             const CameraCalibration *right_calibration,
                             const ImuNoiseParameters &imu_noise)
{
  if (right_calibration != nullptr)
  {
    std::println("标定读取完成:\n"
                 "\tcam0\n"
                 "\t\tfu={:.3f},\n"
                 "\t\tfv={:.3f},\n"
                 "\tcam1\n"
                 "\t\tfu={:.3f},\n"
                 "\t\tfv={:.3f},\n"
                 "\t分辨率 {}x{},\n"
                 "\t陀螺仪白噪声 {:.4e}\n"
                 "\t陀螺仪随机游走 {:.4e}\n"
                 "\t加速度计白噪声 {:.4e}\n"
                 "\t加速度计随机游走 {:.4e}\n",
                 calibration.camera_matrix.at<double>(0, 0),
                 calibration.camera_matrix.at<double>(1, 1),
                 right_calibration->camera_matrix.at<double>(0, 0),
                 right_calibration->camera_matrix.at<double>(1, 1),
                 calibration.image_size.width, calibration.image_size.height,
                 imu_noise.gyro_noise_density, imu_noise.gyro_random_walk,
                 imu_noise.accel_noise_density, imu_noise.accel_random_walk);
  }
  else
  {
    std::println("标定读取完成:\n"
                 "\tcam0\n"
                 "\t\tfu={:.3f},\n"
                 "\t\tfv={:.3f},\n"
                 "\t分辨率 {}x{},\n"
                 "\t陀螺仪白噪声 {:.4e}\n"
                 "\t陀螺仪随机游走 {:.4e}\n"
                 "\t加速度计白噪声 {:.4e}\n"
                 "\t加速度计随机游走 {:.4e}\n",
                 calibration.camera_matrix.at<double>(0, 0),
                 calibration.camera_matrix.at<double>(1, 1),
                 calibration.image_size.width, calibration.image_size.height,
                 imu_noise.gyro_noise_density, imu_noise.gyro_random_walk,
                 imu_noise.accel_noise_density, imu_noise.accel_random_walk);
  }
}

// groundtruth 姿态初始化: 抛弃 groundtruth 首条记录之前的图像帧与 IMU 样本,
// 返回滤波起始的 IMU 样本下标
std::size_t
InitializeFromGroundTruthPose(const fs::path &dataset_root,
                              std::vector<StereoFrame> &stereo_frames,
                              std::span<const ImuSample> imu_samples,
                              Msckf &filter)
{
  const double groundtruth_start_time = LoadGroundTruthStartTime(dataset_root);
  const auto first_covered_frame
      = std::ranges::find_if(stereo_frames,
                             [groundtruth_start_time](const StereoFrame &frame)
                             { return frame.time >= groundtruth_start_time; });
  if (first_covered_frame == stereo_frames.end())
  {
    throw std::runtime_error("所有图像帧都早于 groundtruth 起始时刻");
  }
  if (first_covered_frame != stereo_frames.begin())
  {
    std::println("groundtruth 起始于 t={:.3f}s, 抛弃之前的 {} 帧图像",
                 groundtruth_start_time,
                 std::distance(stereo_frames.begin(), first_covered_frame));
    stereo_frames.erase(stereo_frames.begin(), first_covered_frame);
  }
  const double first_frame_time = stereo_frames.front().time;
  const GroundTruthState initial_state
      = LoadClosestGroundTruthState(dataset_root, first_frame_time);
  filter.InitializeFromGroundTruth(initial_state);

  std::size_t imu_index = 0;
  while (imu_index < imu_samples.size()
         && imu_samples[imu_index].time < first_frame_time)
  {
    ++imu_index; // 首帧之前的 IMU 不参与积分
  }
  std::println("groundtruth 初始状态: t={:.3f}s 位置 [{:.3f} {:.3f} {:.3f}]",
               initial_state.time, initial_state.position.x(),
               initial_state.position.y(), initial_state.position.z());
  return imu_index;
}

// 静止 IMU 姿态初始化, 返回滤波起始的 IMU 样本下标
std::size_t InitializeFromStaticImuPose(std::span<const ImuSample> imu_samples,
                                        double first_frame_time, Msckf &filter)
{
  static constexpr std::size_t kMinInitialSampleCount = 200;
  std::size_t imu_index                               = 0;
  std::vector<ImuSample> initial_samples;
  while (imu_index < imu_samples.size()
         && imu_samples[imu_index].time < first_frame_time)
  {
    initial_samples.push_back(imu_samples[imu_index++]);
  }
  while (initial_samples.size() < kMinInitialSampleCount
         && imu_index < imu_samples.size())
  {
    initial_samples.push_back(imu_samples[imu_index++]);
  }
  filter.InitializeFromStaticImu(initial_samples);
  return imu_index;
}

std::ofstream OpenTrajectoryFile(const fs::path &trajectory_path)
{
  if (trajectory_path.has_parent_path())
  {
    fs::create_directories(trajectory_path.parent_path());
  }
  std::ofstream trajectory_file(trajectory_path);
  if (!trajectory_file)
  {
    throw std::runtime_error(std::format(
        "无法创建轨迹输出文件: '{}'.", fs::absolute(trajectory_path).string()
    ));
  }
  return trajectory_file;
}

// 收集 MSCKF 三角化的路标点，运行结束后写出 ASCII PLY 点云文件
class PointCloudMapper
{
public:
  void AddPoints(const std::vector<Eigen::Vector3d> &points)
  {
    for (const Eigen::Vector3d &p : points)
    {
      points_.push_back(p);
    }
  }

  // 按 EuRoC pointcloud0/data.ply 格式写出 ASCII PLY
  void WritePly(const fs::path &output_path) const
  {
    if (output_path.has_parent_path())
    {
      fs::create_directories(output_path.parent_path());
    }
    std::ofstream file(output_path);
    if (!file)
    {
      throw std::runtime_error(std::format("无法创建点云文件: '{}'.",
                                           fs::absolute(output_path).string()));
    }
    file << "ply\n"
         << "format ascii 1.0\n"
         << std::format("element vertex {}\n", points_.size())
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "end_header\n";
    for (const Eigen::Vector3d &p : points_)
    {
      std::println(file, "{} {} {}", static_cast<float>(p.x()),
                   static_cast<float>(p.y()), static_cast<float>(p.z()));
    }
    std::println("点云已写入 {} ({} 个点)", fs::absolute(output_path).string(),
                 points_.size());
  }

  std::size_t point_count() const
  {
    return points_.size();
  }

private:
  std::vector<Eigen::Vector3d> points_;
};

// 逐帧滤波主循环: IMU 预测 → 克隆增广 → 单目先验融合 → 陀螺辅助光流 → 双目/单目量测
class VioPipeline
{
public:
  VioPipeline(const CameraGeometry &geometry, const ClaheEnhancer &enhancer,
              StationaryDetector &stationary_detector, FeatureTracker &tracker,
              Msckf &filter,
              const std::vector<FastVIO::DatumFast> &monocular_estimations,
              std::span<const ImuSample> imu_samples, std::size_t imu_index,
              std::ostream &trajectory_stream,
              fs::path pointcloud_output_path) :
    geometry_(geometry), enhancer_(enhancer),
    stationary_detector_(stationary_detector), tracker_(tracker),
    filter_(filter), monocular_estimations_(monocular_estimations),
    imu_samples_(imu_samples), imu_index_(imu_index),
    trajectory_stream_(trajectory_stream),
    camera_rotation_in_body_(geometry.body_from_camera().so3()),
    pointcloud_output_path_(std::move(pointcloud_output_path))
  {
  }

  void Run(std::span<const StereoFrame> stereo_frames)
  {
    for (const StereoFrame &frame : stereo_frames)
    {
      ProcessStereoFrame(frame);
    }
    mapper_.WritePly(pointcloud_output_path_);
  }

  long zero_velocity_update_count() const
  {
    return zero_velocity_update_count_;
  }
  long monocular_update_count() const
  {
    return monocular_update_count_;
  }

#if ENABLE_TIMER
  // 打印计时报表: MonocularUpdate 耗时、视觉估计相关函数总耗时及其占比对照
  void PrintTimerReport() const
  {
    timer_.PrintReport();
    const double monocular_ms = timer_.TotalMilliseconds(kTimerMonocularUpdate);
    const double vision_ms    = timer_.TotalMilliseconds(kTimerVisionTotal);
    std::println("MonocularUpdate 总耗时 {:.2f} ms, 视觉估计总耗时 {:.2f} ms, "
                 "视觉部分合计 {:.2f} ms",
                 monocular_ms, vision_ms, monocular_ms + vision_ms);
    std::println("(对比方法: 用 --mono-csv 指向不存在的文件再跑一次, "
                 "对比两次的视觉估计总耗时与视觉部分合计)");
  }
#endif

private:
  static constexpr FrameId kProgressLogInterval = 50;
#if ENABLE_TIMER
  static constexpr std::string_view kTimerMonocularUpdate
      = "MonocularUpdate(含查找)";
  static constexpr std::string_view kTimerVisionTotal = "视觉估计(总)";
  static constexpr std::string_view kTimerVisionTrack = "视觉估计/角点+光流";
  static constexpr std::string_view kTimerVisionFilterUpdate
      = "视觉估计/三角化+EKF更新";
#endif

  void ProcessStereoFrame(const StereoFrame &frame)
  {
#if ENABLE_TIMER && ENABLE_TIME_LOGGER
    const double visual_ms_before_frame = VisualTotalMilliseconds();
#endif
    const double clone_time = frame.time + filter_.time_offset_camera_to_imu();
    PropagateImuUntil(clone_time);

    // 实施 ZUPT
    if (stationary_detector_.IsStationary()
        && filter_.ApplyZeroVelocityUpdate())
    {
      ++zero_velocity_update_count_;
    }

    // 读取图像 (单目模式右图路径为空, 跳过)
    const cv::Mat raw_left
        = cv::imread(frame.left_image_path.string(), cv::IMREAD_GRAYSCALE);
    const cv::Mat raw_right = frame.right_image_path.empty()
                                  ? cv::Mat{}
                                  : cv::imread(frame.right_image_path.string(),
                                               cv::IMREAD_GRAYSCALE);
    if (raw_left.empty()
        || (!frame.right_image_path.empty() && raw_right.empty()))
    {
      return;
    }

    // 立体矫正/去畸变和图像增强 (单目模式右图为空, 跳过增强)
    cv::Mat rectified_left, rectified_right;
    geometry_.Preprocess(raw_left, raw_right, rectified_left, rectified_right);
    const cv::Mat enhanced_left  = enhancer_.Enhance(rectified_left);
    const cv::Mat enhanced_right = rectified_right.empty()
                                       ? cv::Mat{}
                                       : enhancer_.Enhance(rectified_right);

    // 先增广本帧克隆, 再融合外部单目算法的相对运动初始猜测
    // (rVec/tVec 约定见 Msckf::MonocularUpdate 注释); 修正后的姿态
    // 随即用于陀螺辅助光流与双目/单目量测更新
    filter_.AugmentCameraClone(frame_id_);
    {
      TIME_SCOPE(timer_, kTimerMonocularUpdate);
      ApplyMonocularEstimation(frame.timestamp_ns);
    }

    // 陀螺辅助: 滤波器传播的姿态差即两帧间(去零偏)陀螺积分, 作为光流的旋转先验
    const Sophus::SO3d world_from_camera_now
        = filter_.world_from_imu_rotation() * camera_rotation_in_body_;
    const Sophus::SO3d current_camera_from_previous_camera
        = has_previous_camera_rotation_
              ? world_from_camera_now.inverse() * previous_world_from_camera_
              : Sophus::SO3d();
    {
      // 视觉估计: 角点提取/光流跟踪 + 三角化/非线性优化/EKF 量测更新
      TIME_SCOPE(timer_, kTimerVisionTotal);
      std::vector<FeatureTracker::TrackedFeature> tracked;
      {
        TIME_SCOPE(timer_, kTimerVisionTrack);
        tracked = tracker_.Track(enhanced_left, enhanced_right,
                                 current_camera_from_previous_camera);
      }
      Msckf::FilterResult filter_result;
      {
        TIME_SCOPE(timer_, kTimerVisionFilterUpdate);
        filter_result = filter_.ProcessFrame(frame_id_, tracked);
      }
      tracker_.DropFeatures(filter_result.dropped_feature_ids);
      mapper_.AddPoints(filter_result.new_map_points);
      tracked_count_last_frame_ = tracked.size();
    }
    previous_world_from_camera_
        = filter_.world_from_imu_rotation() * camera_rotation_in_body_;
    has_previous_camera_rotation_ = true;

    WriteTrajectoryRecord(clone_time);
#if ENABLE_TIMER && ENABLE_TIME_LOGGER
    frame_time_logger_.LogFrame(frame_id_, VisualTotalMilliseconds()
                                               - visual_ms_before_frame);
#endif
    if (frame_id_ % kProgressLogInterval == 0)
    {
      PrintProgress(tracked_count_last_frame_);
    }
    ++frame_id_;
  }

  // IMU 插值精确传播到成像时刻
  void PropagateImuUntil(double clone_time)
  {
    while (imu_index_ < imu_samples_.size()
           && imu_samples_[imu_index_].time <= clone_time)
    {
      stationary_detector_.AddImuSample(imu_samples_[imu_index_]);
      filter_.PropagateWithImu(imu_samples_[imu_index_++]);
    }
    if (imu_index_ > 0 && imu_index_ < imu_samples_.size())
    {
      filter_.PropagateWithImu(InterpolateImuSample(
          imu_samples_[imu_index_ - 1], imu_samples_[imu_index_], clone_time
      ));
    }
  }

  // 按时间戳查找外部单目估计并做 EKF 更新
  void ApplyMonocularEstimation(std::int64_t timestamp_ns)
  {
    const std::optional<FastVIO::DatumFast> estimation
        = FindDatumFastByTimestamp(monocular_estimations_, timestamp_ns);
    if (estimation.has_value()
        && filter_.MonocularUpdate(estimation->angular_displacement_,
                                   estimation->normalized_translation_))
    {
      ++monocular_update_count_;
    }
  }

  void WriteTrajectoryRecord(double clone_time)
  {
    const Eigen::Quaterniond orientation
        = filter_.world_from_imu_rotation().unit_quaternion();
    const Eigen::Vector3d &position = filter_.imu_position();
    std::println(trajectory_stream_,
                 "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                 clone_time, position.x(), position.y(), position.z(),
                 orientation.x(), orientation.y(), orientation.z(),
                 orientation.w());
  }

  void PrintProgress(std::size_t tracked_count) const
  {
    const Eigen::Vector3d &position = filter_.imu_position();
    std::println(
        "帧 {:4}  位置 [{:7.3f} {:7.3f} {:7.3f}]  速度 {:5.2f} m/s  t_d "
        "{:+.2f} ms  |g| {:.3f}  特征 {:3}  克隆 {}",
        frame_id_, position.x(), position.y(), position.z(),
        filter_.imu_velocity().norm(),
        filter_.time_offset_camera_to_imu() * 1e3,
        filter_.gravity_in_world().norm(), tracked_count, filter_.clone_count()
    );
  }

  const CameraGeometry &geometry_;
  const ClaheEnhancer &enhancer_;
  StationaryDetector &stationary_detector_;
  FeatureTracker &tracker_;
  Msckf &filter_;
  const std::vector<FastVIO::DatumFast> &monocular_estimations_;
  std::span<const ImuSample> imu_samples_;
  std::size_t imu_index_;
  std::ostream &trajectory_stream_;
  const Sophus::SO3d camera_rotation_in_body_;
  fs::path pointcloud_output_path_;
  PointCloudMapper mapper_;

  FrameId frame_id_                     = 0;
  long zero_velocity_update_count_      = 0;
  long monocular_update_count_          = 0;
  std::size_t tracked_count_last_frame_ = 0;
  Sophus::SO3d previous_world_from_camera_;
  bool has_previous_camera_rotation_ = false;
#if ENABLE_TIMER
  mutable FunctionTimer timer_;
#if ENABLE_TIME_LOGGER
  // 视觉任务合计 = MonocularUpdate(含查找) + 视觉估计(总), 与报表口径一致
  double VisualTotalMilliseconds() const
  {
    return timer_.TotalMilliseconds(kTimerMonocularUpdate)
           + timer_.TotalMilliseconds(kTimerVisionTotal);
  }

  FrameTimeLogger frame_time_logger_;
#endif
#endif
};

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
    // 运行时检测数据集相机数目, 决定视觉模式
    const VisionMode vision_mode = DetectVisionMode(dataset_root);
    std::println("视觉模式: {}", vision_mode == VisionMode::kStereo
                                     ? "双目 (cam0+cam1)"
                                     : "单目 (仅一个相机)");
    const std::vector<ImuSample> imu_samples = LoadImuSamples(dataset_root);
    std::vector<StereoFrame> stereo_frames
        = LoadStereoFrames(dataset_root, vision_mode);
    std::println("IMU 样本数 {}, 图像帧数 {}", imu_samples.size(),
                 stereo_frames.size());
    if (imu_samples.empty() || stereo_frames.empty())
    {
      return 1;
    }

    // 单目模式只加载图像源相机 (cam0 优先) 的标定, 双目模式加载 cam0+cam1
    const fs::path camera_calibration_directory
        = vision_mode == VisionMode::kStereo
              ? dataset_root / "cam0"
              : SelectMonoCameraDirectory(dataset_root);
    const CameraCalibration calibration
        = LoadCameraCalibration(camera_calibration_directory / "sensor.yaml");
    const ImuNoiseParameters imu_noise
        = LoadImuNoiseParameters(dataset_root / "imu0" / "sensor.yaml");
    const CameraCalibration *right_calibration = nullptr;
    std::unique_ptr<CameraGeometry> geometry;
    if (vision_mode == VisionMode::kStereo)
    {
      const CameraCalibration right
          = LoadCameraCalibration(dataset_root / "cam1" / "sensor.yaml");
      right_calibration = &right;
      PrintCalibrationSummary(calibration, right_calibration, imu_noise);
      geometry = std::make_unique<StereoRectifier>(calibration, right);
      std::println("矫正后焦距 {:.2f} px, 基线 {:.4f} m",
                   geometry->focal_length(), geometry->baseline());
    }
    else
    {
      PrintCalibrationSummary(calibration, right_calibration, imu_noise);
      geometry = std::make_unique<MonoUndistorter>(calibration);
      std::println("去畸变后焦距 {:.2f} px", geometry->focal_length());
    }

    ClaheEnhancer enhancer;
    StationaryDetector stationary_detector;
    FeatureTracker tracker(*geometry, vision_mode);
    Msckf filter(geometry->body_from_camera(), geometry->baseline(),
                 geometry->focal_length(), imu_noise,
                 options.initial_time_offset, vision_mode);
    std::println("相机-IMU 时间偏移初值 {:+.2f} ms (滤波器在线估计)",
                 options.initial_time_offset * 1e3);

    // 姿态初始化
    const std::size_t imu_index
        = options.initialization_mode == InitializationMode::kGroundTruth
              ? InitializeFromGroundTruthPose(dataset_root, stereo_frames,
                                              imu_samples, filter)
              : InitializeFromStaticImuPose(imu_samples,
                                            stereo_frames.front().time, filter);

    const fs::path &trajectory_path = options.output_trajectory_path;
    std::ofstream trajectory_file   = OpenTrajectoryFile(trajectory_path);

    // 外部单目估计所在系到量测相机系 (矫正后左目系) 的旋转; 单目去畸变时恒为恒等
    const Sophus::SO3d rectified_from_raw_left
        = geometry->body_from_camera().so3().inverse()
          * calibration.body_from_camera.so3();
    const std::vector<FastVIO::DatumFast> monocular_estimations
        = LoadMonocularEstimations(options.monocular_estimation_path,
                                   rectified_from_raw_left);

    VioPipeline pipeline(*geometry, enhancer, stationary_detector, tracker,
                         filter, monocular_estimations, imu_samples, imu_index,
                         trajectory_file, options.output_pointcloud_path);
    pipeline.Run(stereo_frames);
#if ENABLE_TIMER
    pipeline.PrintTimerReport();
#endif

    const Eigen::Vector3d &gravity = filter.gravity_in_world();
    std::println("完成, 轨迹已写入 {} (TUM 格式), ZUPT 触发 {} 次, 单目更新 {} "
                 "次, 最终 t_d {:+.3f} ms, 重力 [{:.4f} {:.4f} {:.4f}] m/s^2",
                 fs::absolute(trajectory_path).string(),
                 pipeline.zero_velocity_update_count(),
                 pipeline.monocular_update_count(),
                 filter.time_offset_camera_to_imu() * 1e3, gravity.x(),
                 gravity.y(), gravity.z());
  }
  catch (const std::exception &error)
  {
    std::println(stderr, "错误: {}", error.what());
    std::println(
        stderr,
        "用法: {} <数据集mav0路径> [--init imu|groundtruth] [--time-offset 秒] "
        "[--output /path/to/filename.tum] [--mono-csv /path/to/estimation.csv] "
        "[--pointcloud /path/to/map.ply]",
        argv[0]
    );
    return 1;
  }
  return 0;
}
