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

#define DEBUG 0

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

#if (DEBUG)
    std::ofstream log_reflect{"Reflect-DatumFast.csv"};
#endif

    // 跳过表头
    std::getline(file, line);
    size_t line_num{0};
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

#if (DEBUG)
        std::print(log_reflect,
                   "{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n", //
                   wt.x(), wt.y(), wt.z(), t.x(), t.y(), t.z());
#endif
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

#if (DEBUG)
    log_reflect.flush();
    std::print(stderr,
               "[DEBUG] 回写世界坐标系下的角位移和单位化平移向量 ...\n");
#endif
    return data;
  }
};
