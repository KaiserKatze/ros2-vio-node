#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
    SensorYaml result_sensor_config;
    YAML::Node node_sensor{YAML::LoadFile(path_sensor_yaml)};
    if (node_sensor["sensor_type"]
        && node_sensor["sensor_type"].as<std::string>() == "imu")
    {
      result_sensor_config.gyroscope_noise_density_
          = node_sensor["gyroscope_noise_density"].as<double>();
      result_sensor_config.gyroscope_random_walk_
          = node_sensor["gyroscope_random_walk"].as<double>();
      result_sensor_config.accelerometer_noise_density_
          = node_sensor["accelerometer_noise_density"].as<double>();
      result_sensor_config.accelerometer_random_walk_
          = node_sensor["accelerometer_random_walk"].as<double>();

      assert(result_sensor_config.gyroscope_noise_density_ >= 0.0
             && !std::isnan(result_sensor_config.gyroscope_noise_density_)
             && !std::isinf(result_sensor_config.gyroscope_noise_density_)
             && result_sensor_config.gyroscope_random_walk_ >= 0.0
             && !std::isnan(result_sensor_config.gyroscope_random_walk_)
             && !std::isinf(result_sensor_config.gyroscope_random_walk_)
             && result_sensor_config.accelerometer_noise_density_ >= 0.0
             && !std::isnan(result_sensor_config.accelerometer_noise_density_)
             && !std::isinf(result_sensor_config.accelerometer_noise_density_)
             && result_sensor_config.accelerometer_random_walk_ >= 0.0
             && !std::isnan(result_sensor_config.accelerometer_random_walk_)
             && !std::isinf(result_sensor_config.accelerometer_random_walk_));
    }
    if (!(node_sensor["T_BS"] && node_sensor["T_BS"]["data"]))
    {
      return std::nullopt;
    }
    std::vector<double> T_BS_data{
        node_sensor["T_BS"]["data"].as<std::vector<double>>()
    };
    Eigen::Map<Eigen::Matrix4d> T_BS_mat{T_BS_data.data()};
    result_sensor_config.transform_matrix_ = std::move(T_BS_mat);
    result_sensor_config.rate_hz_ = node_sensor["rate_hz"].as<double>();
    assert(result_sensor_config.rate_hz_ > 0.0);
    return result_sensor_config;
  }
};

} // namespace FastVIO
