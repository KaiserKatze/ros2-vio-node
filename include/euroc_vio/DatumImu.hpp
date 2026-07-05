#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <meta>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

using namespace std::chrono_literals;

#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include <boost/numeric/odeint.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "yaml-cpp/yaml.h"

#include "euroc_vio/ImuState.hpp"
#include "euroc_vio/AbstractLoader.hpp"
#include "euroc_vio/Interpolation.hpp"
#include "euroc_vio/main.h"
#include "euroc_vio/zupt.hpp"

struct DatumImu
{
  std::int64_t timestamp_;
  Eigen::Vector3d angular_velocity_;
  Eigen::Vector3d linear_acceleration_;

  static std::vector<DatumImu>
  Load(const std::string &path_imu_csv,
       const Sophus::SO3d &sensor_rotation_wrt_body)
  {
    std::vector<DatumImu> data;

    std::ifstream file{path_imu_csv};
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
        // 读取旋转角度
        const double gx{AbstractLoader::get_item_as_double(ss)};
        const double gy{AbstractLoader::get_item_as_double(ss)};
        const double gz{AbstractLoader::get_item_as_double(ss)};
        // 读取位移方向
        const double ax{AbstractLoader::get_item_as_double(ss)};
        const double ay{AbstractLoader::get_item_as_double(ss)};
        const double az{AbstractLoader::get_item_as_double(ss)};

        const DatumImu datum_imu{
            timestamp,
            sensor_rotation_wrt_body * Eigen::Vector3d{gx, gy, gz},
            sensor_rotation_wrt_body * Eigen::Vector3d{ax, ay, az},
        };
        data.push_back(datum_imu);
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_imu_csv, line, //
                        ex.what()),
        };
      }
    } // end while
    return data;
  }
};
