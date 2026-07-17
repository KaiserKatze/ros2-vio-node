module;

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include <sophus/so3.hpp>

export module FastVIO:DatumFast;

// import std;

import :AbstractLoader;

using namespace std::chrono_literals;

export namespace FastVIO
{

struct DatumFast
{
  std::int64_t timestamp_;
  // 角位移向量
  Eigen::Vector3d angular_displacement_;
  // 单位化平移向量
  Eigen::Vector3d normalized_translation_;

  static std::vector<DatumFast>
  Load(const std::string &path_estimation_csv,
       const Sophus::SO3d &sensor_rotation_wrt_body)
  {
    std::vector<DatumFast> data;
    std::ifstream file{path_estimation_csv};
    std::string line;

    // 跳过表头
    std::getline(file, line);
    std::size_t line_num{0};
    while (std::getline(file, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
        // 读取时间戳
        const std::int64_t timestamp{
            AbstractLoader::get_item_as_int64(ss), // in nanoseconds
        };
        // 读取旋转角度
        const double wxt{AbstractLoader::get_item_as_double(ss)};
        const double wyt{AbstractLoader::get_item_as_double(ss)};
        const double wzt{AbstractLoader::get_item_as_double(ss)};
        // 读取位移方向
        const double tx{AbstractLoader::get_item_as_double(ss)};
        const double ty{AbstractLoader::get_item_as_double(ss)};
        const double tz{AbstractLoader::get_item_as_double(ss)};

        const Eigen::Vector3d wt{
            sensor_rotation_wrt_body * Eigen::Vector3d{wxt, wyt, wzt},
        };
        const Eigen::Vector3d t{
            sensor_rotation_wrt_body * Eigen::Vector3d{tx, ty, tz},
        };

        const DatumFast datum_fast{timestamp, wt, t};
        data.push_back(datum_fast);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_estimation_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};

} // namespace FastVIO
