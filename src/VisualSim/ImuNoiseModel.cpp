#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ImuNoiseModel.hpp"

namespace FastVIO::VisualSim
{

namespace
{
// 固定种子保证噪声注入结果可复现
constexpr std::uint32_t kDefaultSeed{42U};
// imu0/sensor.yaml 中需要同步更新的四个噪声密度字段
constexpr std::array<std::string_view, 4> kImuNoiseKeys{
    "gyroscope_noise_density",
    "gyroscope_random_walk",
    "accelerometer_noise_density",
    "accelerometer_random_walk",
};
} // namespace

// 一行 IMU 测量, 列序与 EuRoC imu0/data.csv 一致:
// timestamp [ns], w_RS_S_x/y/z [rad s^-1], a_RS_S_x/y/z [m s^-2]
struct ImuSample
{
  std::int64_t timestamp_ns{0};
  ImuNoiseModel<double>::Vector3 angular_velocity{};
  ImuNoiseModel<double>::Vector3 linear_acceleration{};
};

std::vector<ImuSample> ReadImuData(const std::filesystem::path &path_imu_data,
                                   std::vector<std::string> &header_lines)
{
  std::ifstream fin{path_imu_data};
  if (!fin)
  {
    throw std::runtime_error{
        std::format("无法打开 '{}'.", path_imu_data.string()),
    };
  }

  std::vector<ImuSample> samples;
  std::string line;
  std::size_t line_number{0};
  while (std::getline(fin, line))
  {
    ++line_number;
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    if (line.empty())
    {
      continue;
    }
    if (line.front() == '#')
    {
      header_lines.push_back(line);
      continue;
    }

    std::ranges::replace(line, ',', ' ');
    std::istringstream stream{line};
    ImuSample sample;
    if (!(stream >> sample.timestamp_ns >> sample.angular_velocity.x()
          >> sample.angular_velocity.y() >> sample.angular_velocity.z()
          >> sample.linear_acceleration.x() >> sample.linear_acceleration.y()
          >> sample.linear_acceleration.z()))
    {
      throw std::runtime_error{std::format(
          "数据行格式错误 (第 {} 行): '{}'.", line_number, line
      )};
    }
    samples.push_back(sample);
  }

  if (samples.empty())
  {
    throw std::runtime_error{
        std::format("未读取到任何 IMU 样本: '{}'.", path_imu_data.string()),
    };
  }
  return samples;
}

double EstimateSampleRateHz(const std::vector<ImuSample> &samples)
{
  std::vector<double> intervals;
  for (std::size_t i{1}; i < samples.size(); ++i)
  {
    const double dt{
        (samples[i].timestamp_ns - samples[i - 1].timestamp_ns) * 1e-9,
    };
    if (dt > 0.0)
    {
      intervals.push_back(dt);
    }
  }
  if (intervals.empty())
  {
    throw std::runtime_error{"时间戳无法构成正间隔, 无法估计采样率."};
  }
  std::ranges::sort(intervals);
  return 1.0 / intervals[intervals.size() / 2];
}

void WriteImuData(const std::filesystem::path &path_imu_data,
                  const std::vector<std::string> &header_lines,
                  const std::vector<ImuSample> &samples)
{
  std::ofstream fout{path_imu_data};
  if (!fout)
  {
    throw std::runtime_error{
        std::format("无法写入 '{}'.", path_imu_data.string()),
    };
  }
  for (const std::string &header : header_lines)
  {
    std::println(fout, "{}", header);
  }
  for (const ImuSample &sample : samples)
  {
    std::print(fout,
               "{},{:.18g},{:.18g},{:.18g},{:.18g},{:.18g},{:.18g}\n",
               sample.timestamp_ns,                    //
               sample.angular_velocity.x(),            //
               sample.angular_velocity.y(),            //
               sample.angular_velocity.z(),            //
               sample.linear_acceleration.x(),         //
               sample.linear_acceleration.y(),         //
               sample.linear_acceleration.z());
  }
}

// 就地更新 imu0/sensor.yaml 的四个噪声密度字段, 其余内容 (T_BS 外参等) 原样保留
void UpdateImuSensorYaml(const std::filesystem::path &path_sensor_yaml,
                         const ImuNoiseModel<double>::Config &noise_config)
{
  std::ifstream fin{path_sensor_yaml};
  if (!fin)
  {
    throw std::runtime_error{
        std::format("无法打开 '{}'.", path_sensor_yaml.string()),
    };
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(fin, line))
  {
    if (!line.empty() && line.back() == '\r')
    {
      line.pop_back();
    }
    lines.push_back(line);
  }
  fin.close();

  const std::array<double, 4> noise_values{
      noise_config.gyro_noise_density,
      noise_config.gyro_random_walk,
      noise_config.accel_noise_density,
      noise_config.accel_random_walk,
  };
  std::array<bool, 4> updated{};
  for (std::string &current : lines)
  {
    const std::size_t key_start{current.find_first_not_of(" \t")};
    if (key_start == std::string::npos)
    {
      continue;
    }
    for (std::size_t i{0}; i < kImuNoiseKeys.size(); ++i)
    {
      const std::string_view key{kImuNoiseKeys[i]};
      if (current.compare(key_start, key.size(), key) != 0
          || key_start + key.size() >= current.size()
          || current[key_start + key.size()] != ':')
      {
        continue;
      }
      // 保留 '#' 之后的注释, 仅替换键值
      const std::size_t comment_start{current.find('#', key_start)};
      const std::string_view comment{
          comment_start == std::string::npos
              ? std::string_view{}
              : std::string_view{current}.substr(comment_start),
      };
      const std::string indent{current.substr(0, key_start)};
      current = comment.empty()
                    ? std::format("{}{}: {:.4e}", indent, key, noise_values[i])
                    : std::format("{}{}: {:.4e} {}", indent, key,
                                  noise_values[i], comment);
      updated[i] = true;
      break;
    }
  }
  for (std::size_t i{0}; i < kImuNoiseKeys.size(); ++i)
  {
    if (!updated[i])
    {
      lines.push_back(std::format("{}: {:.4e}", kImuNoiseKeys[i],
                                  noise_values[i]));
    }
  }

  std::ofstream fout{path_sensor_yaml};
  if (!fout)
  {
    throw std::runtime_error{
        std::format("无法写入 '{}'.", path_sensor_yaml.string()),
    };
  }
  for (const std::string &current : lines)
  {
    std::println(fout, "{}", current);
  }
}

int InjectImuNoise(const std::filesystem::path &path_mav0)
{
  const std::filesystem::path path_imu_data{path_mav0 / "imu0" / "data.csv"};
  const std::filesystem::path path_sensor_yaml{
      path_mav0 / "imu0" / "sensor.yaml",
  };
  if (!std::filesystem::exists(path_imu_data)
      || !std::filesystem::exists(path_sensor_yaml))
  {
    throw std::runtime_error{std::format(
        "无法找到 '{}' 或 '{}'.", path_imu_data.string(),
        path_sensor_yaml.string()
    )};
  }

  // 首次运行时备份原始数据与标定文件; 备份已存在说明可能重复注入过,
  // 拒绝执行以免噪声叠加
  const std::filesystem::path path_imu_backup{path_imu_data.string() + ".orig"};
  const std::filesystem::path path_sensor_backup{
      path_sensor_yaml.string() + ".orig",
  };
  if (std::filesystem::exists(path_imu_backup)
      || std::filesystem::exists(path_sensor_backup))
  {
    throw std::runtime_error{std::format(
        "备份 '{}' 或 '{}' 已存在, 原始数据疑似已注入过噪声; "
        "如需再次注入请先删除备份.",
        path_imu_backup.string(), path_sensor_backup.string()
    )};
  }
  std::filesystem::copy_file(path_imu_data, path_imu_backup);
  std::filesystem::copy_file(path_sensor_yaml, path_sensor_backup);

  std::vector<std::string> header_lines;
  std::vector<ImuSample> samples{ReadImuData(path_imu_data, header_lines)};
  const double sample_rate{EstimateSampleRateHz(samples)};

  // 噪声参数采用 EuRoC ADIS16448 规格 (ImuNoiseModel::Config 默认值),
  // 逐样本叠加 Bias 随机游走与高斯白噪声
  ImuNoiseModel<double> imu_noise_model{
      ImuNoiseModel<double>::Config{}, sample_rate, kDefaultSeed,
  };
  for (ImuSample &sample : samples)
  {
    const auto [gyro_measured, accel_measured]{
        imu_noise_model.Corrupt(sample.angular_velocity,
                                sample.linear_acceleration),
    };
    sample.angular_velocity    = gyro_measured;
    sample.linear_acceleration = accel_measured;
  }
  WriteImuData(path_imu_data, header_lines, samples);
  UpdateImuSensorYaml(path_sensor_yaml, imu_noise_model.GetConfig());

  const auto &noise_config{imu_noise_model.GetConfig()};
  std::println("完成: 已为 {} 个 IMU 样本注入噪声 (采样率 {:.1f} Hz, 种子 {})",
               samples.size(), sample_rate, kDefaultSeed);
  std::println("  陀螺仪白噪声密度     {:.4e} [rad/s/sqrt(Hz)]",
               noise_config.gyro_noise_density);
  std::println("  陀螺仪随机游走       {:.4e} [rad/s^2/sqrt(Hz)]",
               noise_config.gyro_random_walk);
  std::println("  加速度计白噪声密度   {:.4e} [m/s^2/sqrt(Hz)]",
               noise_config.accel_noise_density);
  std::println("  加速度计随机游走     {:.4e} [m/s^3/sqrt(Hz)]",
               noise_config.accel_random_walk);
  std::println("  数据输出: {}",
               std::filesystem::absolute(path_imu_data).string());
  std::println("  标定更新: {}",
               std::filesystem::absolute(path_sensor_yaml).string());
  std::println("  原始数据备份: {}",
               std::filesystem::absolute(path_imu_backup).string());
  std::println("  原始标定备份: {}",
               std::filesystem::absolute(path_sensor_backup).string());
  return 0;
}

} // namespace FastVIO::VisualSim

int main(int argc, char **argv)
{
  try
  {
    if (argc != 2)
    {
      std::println(stderr, "用法: {} <数据集mav0路径>", argv[0]);
      return 1;
    }
    return FastVIO::VisualSim::InjectImuNoise(std::filesystem::path{argv[1]});
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "错误: {}", ex.what());
    return 1;
  }
}
