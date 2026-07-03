#pragma once

#include <numeric>
#include <random>
#include <span>

#include <Eigen/Core>

/**
 * @brief 简易 3D Perlin 噪声实现，用于生成连续梯度的扰动
 * @note 可以用 [FastNoise2](https://github.com/Auburn/FastNoise2) 代替
 */
struct PerlinNoise
{
  int p[512];

  PerlinNoise()
  {
    std::iota(std::begin(p), std::begin(p) + 256, 0);
    std::shuffle(std::begin(p), std::begin(p) + 256,
                 std::default_random_engine(std::random_device{}()));
    for (int i = 0; i < 256; ++i)
    {
      p[256 + i] = p[i];
    }
  }

  ~PerlinNoise() = default;

  double fade(double t)
  {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  double lerp(double t, double a, double b)
  {
    return a + t * (b - a);
  }

  double grad(int hash, double x, double y, double z)
  {
    int h{hash & 15};
    double u{h < 8 ? x : y};
    double v{h < 4 ? y : h == 12 || h == 14 ? x : z};
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }

  double noise(double x, double y, double z)
  {
    int X{static_cast<int>(std::floor(x)) & 255};
    int Y{static_cast<int>(std::floor(y)) & 255};
    int Z{static_cast<int>(std::floor(z)) & 255};
    x -= std::floor(x);
    y -= std::floor(y);
    z -= std::floor(z);
    double u{fade(x)};
    double v{fade(y)};
    double w{fade(z)};
    int A{p[X] + Y};
    int AA{p[A] + Z};
    int AB{p[A + 1] + Z};
    int B{p[X + 1] + Y};
    int BA{p[B] + Z};
    int BB{p[B + 1] + Z};
    return lerp(w,
                lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                     lerp(u, grad(p[AB], x, y - 1, z),
                          grad(p[BB], x - 1, y - 1, z))),
                lerp(v,
                     lerp(u, grad(p[AA + 1], x, y, z - 1),
                          grad(p[BA + 1], x - 1, y, z - 1)),
                     lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                          grad(p[BB + 1], x - 1, y - 1, z - 1))));
  }

  template <typename value_type>
  static Eigen::Vector<value_type, 3> GetDisturbance(std::span<int> pt_idx)
  {
    static PerlinNoise pn;
    // 控制噪声的“频率”，值越小梯度越缓和
    static const value_type noise_scale{0.5};
    static const value_type min_disturb{0.01};
    static const value_type max_disturb{0.10};

    // 为 X, Y, Z 分别计算连续的噪声值 [-1.0, 1.0]
    // 输入坐标除以步长可以确保相邻点在噪声空间中也是相邻的
    const value_type nx{
        static_cast<value_type>(pn.noise(pt_idx[0] * noise_scale,
                                         pt_idx[1] * noise_scale,
                                         pt_idx[2] * noise_scale)),
    };
    const value_type ny{
        static_cast<value_type>(pn.noise(pt_idx[0] * noise_scale + 100.0,
                                         pt_idx[1] * noise_scale,
                                         pt_idx[2] * noise_scale)),
    };
    const value_type nz{
        static_cast<value_type>(pn.noise(pt_idx[0] * noise_scale,
                                         pt_idx[1] * noise_scale + 100.0,
                                         pt_idx[2] * noise_scale)),
    };

    // 将噪声映射到指定范围 [0.01, 0.10]
    static const auto map_noise = [&](value_type n)
    {
      // 将 [-1, 1] 映射到 [min, max]
      return min_disturb + (std::abs(n)) * (max_disturb - min_disturb);
    };

    return {map_noise(nx), map_noise(ny), map_noise(nz)};
  }
};
