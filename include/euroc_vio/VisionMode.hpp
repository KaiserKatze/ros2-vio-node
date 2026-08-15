#pragma once

#include <filesystem>

namespace FastVIO
{

// 视觉模式: 数据集相机数目
enum class VisionMode
{
  kStereo, // cam0 + cam1
  kMono    // 仅 cam0 或仅 cam1
};

// 检测数据集模式: cam1/data.csv 存在且非空 → 双目, 否则单目
inline VisionMode DetectVisionMode(const std::filesystem::path &dataset_root)
{
  const std::filesystem::path cam1_csv{dataset_root / "cam1" / "data.csv"};
  return std::filesystem::exists(cam1_csv)
                 && std::filesystem::file_size(cam1_csv) > 0
             ? VisionMode::kStereo
             : VisionMode::kMono;
}

// 单目模式的图像源目录: 优先 cam0, 缺失时回退 cam1
inline std::filesystem::path
SelectMonoCameraDirectory(const std::filesystem::path &dataset_root)
{
  if (std::filesystem::exists(dataset_root / "cam0" / "data.csv"))
  {
    return dataset_root / "cam0";
  }
  return dataset_root / "cam1";
}

} // namespace FastVIO
