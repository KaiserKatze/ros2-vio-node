#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

namespace FastVIO
{

template <typename T>
concept PathLike = std::convertible_to<T, std::filesystem::path>;

struct SensorYaml
{
  // 姿态, 变换矩阵
  Eigen::Matrix4d transform_matrix_{Eigen::Matrix4d::Identity()};
  // 图像分辨率 [宽, 高] (相机节点专用, 默认 0x0)
  Eigen::Vector2i resolution_{Eigen::Vector2i::Zero()};
  // 针孔内参 [fu, fv, cu, cv] (相机节点专用)
  Eigen::Vector4d intrinsics_{Eigen::Vector4d::Zero()};
  // 畸变系数 [k1, k2, p1, p2] (相机节点专用)
  Eigen::Vector4d distortion_coefficients_{Eigen::Vector4d::Zero()};
  // 采样频率 (单位: Hz)
  double rate_hz_{1.0};
  // 陀螺仪白噪声密度 (单位: rad / s / sqrt(Hz))
  double gyroscope_noise_density_{0.0};
  // 陀螺仪零偏随机游走 (单位: rad / s^2 / sqrt(Hz))
  double gyroscope_random_walk_{0.0};
  // 加速度计白噪声密度 (单位: m / s^2 / sqrt(Hz))
  double accelerometer_noise_density_{0.0};
  // 加速度计零偏随机游走 (单位: m / s^3 / sqrt(Hz))
  double accelerometer_random_walk_{0.0};

  SensorYaml() {}

  SensorYaml(const Eigen::Matrix4d &transform_matrix) :
    transform_matrix_{transform_matrix}
  {
  }

  SensorYaml(Eigen::Matrix4d &&transform_matrix) :
    transform_matrix_{std::move(transform_matrix)}
  {
  }

  ~SensorYaml() = default;

  static std::optional<SensorYaml>
  ReadSensorYaml(const PathLike auto &path_sensor_yaml)
  {
    const std::filesystem::path yaml_path{path_sensor_yaml};
    try
    {
      return ParseSensorNode(YAML::LoadFile(yaml_path.string()), yaml_path);
    }
    catch (const YAML::Exception &ex)
    {
      throw std::runtime_error{std::format("解析传感器配置 '{}' 失败{}: {}.",
                                           yaml_path.string(),
                                           DescribeYamlMark(ex.mark), ex.msg)};
    }
  }

private:
  static constexpr std::string_view kImuSensorType{"imu"};
  static constexpr std::size_t kTransformMatrixCols{4};
  static constexpr std::size_t kTransformDataElementCount{16};
  static constexpr std::size_t kResolutionElementCount{2};
  static constexpr std::size_t kIntrinsicsElementCount{4};
  static constexpr std::size_t kRadtanCoefficientCount{4};

  // 按需解析: T_BS 对所有传感器必需; rate_hz 与噪声参数仅对 IMU 必需,
  // 分辨率/内参/畸变系数仅对相机必需, 其他传感器缺失这些字段时保留默认值
  static SensorYaml ParseSensorNode(const YAML::Node &node_sensor,
                                    const std::filesystem::path &yaml_path)
  {
    SensorYaml config;
    config.transform_matrix_ = ReadTransformMatrix(node_sensor, yaml_path);
    if (IsImuSensor(node_sensor))
    {
      config.rate_hz_ = ReadPositiveDouble(node_sensor, "rate_hz", yaml_path);
      config.gyroscope_noise_density_
          = ReadNonNegativeDouble(node_sensor, "gyroscope_noise_density",
                                  yaml_path);
      config.gyroscope_random_walk_
          = ReadNonNegativeDouble(node_sensor, "gyroscope_random_walk",
                                  yaml_path);
      config.accelerometer_noise_density_
          = ReadNonNegativeDouble(node_sensor, "accelerometer_noise_density",
                                  yaml_path);
      config.accelerometer_random_walk_
          = ReadNonNegativeDouble(node_sensor, "accelerometer_random_walk",
                                  yaml_path);
    }
    else if (HasCameraIntrinsics(node_sensor))
    {
      // 相机标定字段: 按字段存在性判定, 兼容缺少 sensor_type 声明的相机 yaml
      config.resolution_ = ReadImageResolution(node_sensor, yaml_path);
      config.intrinsics_ = ReadIntrinsics(node_sensor, yaml_path);
      config.distortion_coefficients_
          = ReadDistortionCoefficients(node_sensor, yaml_path);
      if (node_sensor["rate_hz"])
      {
        config.rate_hz_ = ReadPositiveDouble(node_sensor, "rate_hz", yaml_path);
      }
    }
    else if (node_sensor["rate_hz"])
    {
      config.rate_hz_ = ReadPositiveDouble(node_sensor, "rate_hz", yaml_path);
    }
    return config;
  }

  static bool IsImuSensor(const YAML::Node &node_sensor)
  {
    const YAML::Node node_type{node_sensor["sensor_type"]};
    return node_type && node_type.as<std::string>() == kImuSensorType;
  }

  static bool HasCameraIntrinsics(const YAML::Node &node_sensor)
  {
    return static_cast<bool>(node_sensor["intrinsics"])
           || static_cast<bool>(node_sensor["resolution"]);
  }

  static Eigen::Vector2i
  ReadImageResolution(const YAML::Node &node_sensor,
                      const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_resolution{node_sensor["resolution"]};
    if (!node_resolution || node_resolution.size() != kResolutionElementCount)
    {
      throw std::runtime_error{std::format(
          "'{}' 缺少配置 resolution (2 元素 [宽, 高]).", yaml_path.string()
      )};
    }
    const Eigen::Vector2i resolution{node_resolution[0].as<int>(),
                                     node_resolution[1].as<int>()};
    if (resolution.x() <= 0 || resolution.y() <= 0)
    {
      throw std::runtime_error{
          std::format("'{}' 的配置 resolution 必须为正整数, 实际为 {}x{}.",
                      yaml_path.string(), resolution.x(), resolution.y())
      };
    }
    return resolution;
  }

  static Eigen::Vector4d ReadIntrinsics(const YAML::Node &node_sensor,
                                        const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_intrinsics{node_sensor["intrinsics"]};
    if (!node_intrinsics || node_intrinsics.size() != kIntrinsicsElementCount)
    {
      throw std::runtime_error{
          std::format("'{}' 缺少配置 intrinsics (4 元素 [fu, fv, cu, cv]).",
                      yaml_path.string())
      };
    }
    Eigen::Vector4d intrinsics;
    for (std::size_t i = 0; i < kIntrinsicsElementCount; ++i)
    {
      intrinsics(static_cast<Eigen::Index>(i))
          = node_intrinsics[i].as<double>();
    }
    return intrinsics;
  }

  static Eigen::Vector4d
  ReadDistortionCoefficients(const YAML::Node &node_sensor,
                             const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_coefficients{node_sensor["distortion_coefficients"]};
    if (!node_coefficients)
    {
      throw std::runtime_error{std::format(
          "'{}' 缺少配置 distortion_coefficients.", yaml_path.string()
      )};
    }
    // 兼容两种形式: EuRoC 标准 sequence [k1,k2,p1,p2] 与
    // 以 k1/k2/p1/p2 命名键的 map (部分自定义数据集采用)
    static constexpr std::array<std::string_view, kRadtanCoefficientCount>
        kRadtanKeys{"k1", "k2", "p1", "p2"};
    const std::size_t coefficient_count
        = std::min(kRadtanKeys.size(), node_coefficients.size());
    Eigen::Vector4d coefficients{Eigen::Vector4d::Zero()};
    for (std::size_t i = 0; i < coefficient_count; ++i)
    {
      const double value
          = node_coefficients.IsMap()
                ? node_coefficients[std::string{kRadtanKeys[i]}].as<double>()
                : node_coefficients[i].as<double>();
      coefficients(static_cast<Eigen::Index>(i)) = value;
    }
    return coefficients;
  }

  static Eigen::Matrix4d
  ReadTransformMatrix(const YAML::Node &node_sensor,
                      const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_T_BS{node_sensor["T_BS"]};
    if (!node_T_BS)
    {
      // 部分裁剪过的数据集缺 T_BS 字段 (如本测试数据集, 仿真相机外参
      // 实际为单位变换), 回退单位外参并提示, 避免运行中断
      std::print(stderr, "警告: '{}' 缺少 T_BS, 回退单位外参\n",
                 yaml_path.string());
      return Eigen::Matrix4d::Identity();
    }
    const YAML::Node node_data{node_T_BS["data"]};
    if (!node_data || node_data.size() != kTransformDataElementCount)
    {
      throw std::runtime_error{
          std::format("'{}' 的配置 T_BS.data 缺失或元素个数不是 {}.",
                      yaml_path.string(), kTransformDataElementCount)
      };
    }
    // EuRoC 的 T_BS.data 为行主序展开
    Eigen::Matrix4d matrix;
    for (std::size_t i = 0; i < kTransformDataElementCount; ++i)
    {
      matrix(i / kTransformMatrixCols, i % kTransformMatrixCols)
          = node_data[i].as<double>();
    }
    return matrix;
  }

  static double ReadFiniteDouble(const YAML::Node &node_sensor,
                                 std::string_view key,
                                 const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_value{node_sensor[std::string{key}]};
    if (!node_value)
    {
      throw std::runtime_error{std::format("'{}' 缺少配置 {}.",
                                           yaml_path.string(), key)};
    }
    const double value{node_value.as<double>()};
    if (!std::isfinite(value))
    {
      throw std::runtime_error{std::format("'{}' 的配置 {} 不是有限数值.",
                                           yaml_path.string(), key)};
    }
    return value;
  }

  static double ReadPositiveDouble(const YAML::Node &node_sensor,
                                   std::string_view key,
                                   const std::filesystem::path &yaml_path)
  {
    const double value{ReadFiniteDouble(node_sensor, key, yaml_path)};
    if (value <= 0.0)
    {
      throw std::runtime_error{
          std::format("'{}' 的配置 {} 必须为正数, 实际为 {}.",
                      yaml_path.string(), key, value)
      };
    }
    return value;
  }

  static double ReadNonNegativeDouble(const YAML::Node &node_sensor,
                                      std::string_view key,
                                      const std::filesystem::path &yaml_path)
  {
    const double value{ReadFiniteDouble(node_sensor, key, yaml_path)};
    if (value < 0.0)
    {
      throw std::runtime_error{
          std::format("'{}' 的配置 {} 不允许为负数, 实际为 {}.",
                      yaml_path.string(), key, value)
      };
    }
    return value;
  }

  static std::string DescribeYamlMark(const YAML::Mark &mark)
  {
    return mark.pos >= 0
               ? std::format(" (行 {}, 列 {})", mark.line + 1, mark.column + 1)
               : std::string{};
  }
};

} // namespace FastVIO
