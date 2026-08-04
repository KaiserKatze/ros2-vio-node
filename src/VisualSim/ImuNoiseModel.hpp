#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>

#include <Eigen/Core>

namespace FastVIO::VisualSim
{

/**
 * @brief IMU 测量误差模型 (加性 Bias 随机游走 + 白噪声)
 *
 * 数学模型:
 *   ω_m = ω + b_g + n_g
 *   a_m = a + b_a + n_a
 * 其中 b 满足随机游走 b_{k+1} = b_k + η, η ~ N(0, σ_rw²·Δt),
 * n 为每帧独立采样的零均值高斯白噪声 n ~ N(0, σ_noise²)。
 *
 * 对外参数为连续时间噪声密度 (与 EuRoC sensor.yaml 规格一致),
 * 连续 → 离散转换在构造时完成:
 *   白噪声离散标准差       σ_d = σ_c · √f
 *   Bias 游走离散增量标准差 σ_b = σ_rw · √Δt = σ_rw / √f
 */
template <typename value_type>
class ImuNoiseModel
{
public:
  using Vector3 = Eigen::Vector<value_type, 3>;
  using Seed    = std::uint32_t;
  // (陀螺仪测量值, 加速度计测量值)
  using Measurement = std::pair<Vector3, Vector3>;

  struct Config
  {
    // 默认值取自 EuRoC 数据集 ADIS16448 的连续时间噪声密度
    value_type gyro_noise_density{
        static_cast<value_type>(1.6968e-4)
    }; // rad/s/√Hz
    value_type gyro_random_walk{
        static_cast<value_type>(1.9393e-5)
    }; // rad/s²/√Hz
    value_type accel_noise_density{
        static_cast<value_type>(2.0000e-3)
    }; // m/s²/√Hz
    value_type accel_random_walk{
        static_cast<value_type>(3.0000e-3)
    }; // m/s³/√Hz
  };

  /**
   * @param config 连续时间噪声密度
   * @param sample_rate 采样率 (Hz)
   * @param seed 有值时可复现 (单元测试/可重复实验);
   *             为空时以硬件熵源播种 (蒙特卡洛仿真)
   */
  explicit ImuNoiseModel(const Config &config, value_type sample_rate,
                         std::optional<Seed> seed = std::nullopt) :
    config_{config}, sqrt_rate_{std::sqrt(sample_rate)},
    generator_{seed.has_value() ? seed.value() : std::random_device{}()},
    standard_normal_{static_cast<value_type>(0.0), static_cast<value_type>(1.0)}
  {
    UpdateNoiseModel();
  }

  /**
   * @brief 对一帧已离散采样的理想角速度/加速度叠加噪声, 每个采样步调用一次
   *
   * 先推进 Bias 随机游走, 再叠加白噪声; 输入真值不被修改,
   * 返回值满足 Measurement = GroundTruth + Bias + WhiteNoise。
   */
  Measurement Corrupt(const Vector3 &gyro_truth, const Vector3 &accel_truth)
  {
    gyro_bias_ += SampleIsotropicGaussian(sigma_gyro_bias_walk_);
    accel_bias_ += SampleIsotropicGaussian(sigma_accel_bias_walk_);
    return Measurement{
        gyro_truth + gyro_bias_ + SampleIsotropicGaussian(sigma_gyro_noise_),
        accel_truth + accel_bias_ + SampleIsotropicGaussian(sigma_accel_noise_),
    };
  }

  void SetConfig(const Config &config) noexcept
  {
    this->config_ = config;
    UpdateNoiseModel();
  }

  const Config &GetConfig() const
  {
    return config_;
  }

  const Vector3 &GetGyroBias() const
  {
    return gyro_bias_;
  }

  const Vector3 &GetAccelBias() const
  {
    return accel_bias_;
  }

private:
  void UpdateNoiseModel()
  {
    sigma_gyro_noise_      = config_.gyro_noise_density * sqrt_rate_;
    sigma_accel_noise_     = config_.accel_noise_density * sqrt_rate_;
    sigma_gyro_bias_walk_  = config_.gyro_random_walk / sqrt_rate_;
    sigma_accel_bias_walk_ = config_.accel_random_walk / sqrt_rate_;
  }

  Vector3 SampleIsotropicGaussian(value_type sigma)
  {
    return Vector3{
        sigma * standard_normal_(generator_),
        sigma * standard_normal_(generator_),
        sigma * standard_normal_(generator_),
    };
  }

  Config config_{};
  const value_type sqrt_rate_{200.0};
  std::mt19937 generator_;
  std::normal_distribution<value_type> standard_normal_;

  Vector3 gyro_bias_{Vector3::Zero()};
  Vector3 accel_bias_{Vector3::Zero()};

  // 白噪声离散标准差
  value_type sigma_gyro_noise_{0};
  value_type sigma_accel_noise_{0};
  // Bias 随机游走离散增量标准差
  value_type sigma_gyro_bias_walk_{0};
  value_type sigma_accel_bias_walk_{0};
};

} // namespace FastVIO::VisualSim
