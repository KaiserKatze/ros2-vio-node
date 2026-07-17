module;

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include <sophus/so3.hpp>

export module FastVIO:DatumTruth;

// import std;

import :AbstractLoader;

using namespace std::chrono_literals;

export namespace FastVIO
{

struct DatumTruth
{
  std::int64_t timestamp_;
  Eigen::Vector3d position_;
  Eigen::Quaterniond attitude_;
  Eigen::Vector3d velocity_;
  Eigen::Vector3d bias_gyro_;
  Eigen::Vector3d bias_accel_;

  static std::vector<DatumTruth>
  Load(const std::string &path_truth_csv,
       const Sophus::SO3d &sensor_rotation_wrt_body)
  {
    std::vector<DatumTruth> data;

    std::ifstream file{path_truth_csv};
    std::string line;
    size_t line_num{0};

    // 跳过表头
    std::getline(file, line);
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
        // 读取位置 (m)
        const double px{AbstractLoader::get_item_as_double(ss)};
        const double py{AbstractLoader::get_item_as_double(ss)};
        const double pz{AbstractLoader::get_item_as_double(ss)};
        // 读取朝向
        const double qw{AbstractLoader::get_item_as_double(ss)};
        const double qx{AbstractLoader::get_item_as_double(ss)};
        const double qy{AbstractLoader::get_item_as_double(ss)};
        const double qz{AbstractLoader::get_item_as_double(ss)};
        // 读取速度 (m s^-1)
        const double vx{AbstractLoader::get_item_as_double(ss)};
        const double vy{AbstractLoader::get_item_as_double(ss)};
        const double vz{AbstractLoader::get_item_as_double(ss)};
        // 读取陀螺仪偏差 (rad s^-1)
        const double bwx{AbstractLoader::get_item_as_double(ss)};
        const double bwy{AbstractLoader::get_item_as_double(ss)};
        const double bwz{AbstractLoader::get_item_as_double(ss)};
        // 读取加速度计偏差 (m s^-2)
        const double bax{AbstractLoader::get_item_as_double(ss)};
        const double bay{AbstractLoader::get_item_as_double(ss)};
        const double baz{AbstractLoader::get_item_as_double(ss)};

        const DatumTruth datum_truth{
            timestamp,                                              //
            sensor_rotation_wrt_body * Eigen::Vector3d{px, py, pz}, //
            (Eigen::Quaterniond{qw, qx, qy, qz}
             * sensor_rotation_wrt_body.unit_quaternion().conjugate())
                .normalized(),                                         //
            sensor_rotation_wrt_body * Eigen::Vector3d{vx, vy, vz},    //
            sensor_rotation_wrt_body * Eigen::Vector3d{bwx, bwy, bwz}, //
            sensor_rotation_wrt_body * Eigen::Vector3d{bax, bay, baz}, //
        };
        data.push_back(datum_truth);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_truth_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};

} // namespace FastVIO
