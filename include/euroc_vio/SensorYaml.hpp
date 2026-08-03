#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <optional>
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

  // 按需解析: T_BS 对所有传感器必需; rate_hz 与噪声参数仅对 IMU 必需,
  // 其他传感器 (如相机/真值) 缺失这些字段时保留默认值
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

  static Eigen::Matrix4d
  ReadTransformMatrix(const YAML::Node &node_sensor,
                      const std::filesystem::path &yaml_path)
  {
    const YAML::Node node_T_BS{node_sensor["T_BS"]};
    if (!node_T_BS)
    {
      throw std::runtime_error{std::format("'{}' 缺少配置 T_BS.",
                                           yaml_path.string())};
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
