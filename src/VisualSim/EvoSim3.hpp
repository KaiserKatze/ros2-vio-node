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
#include <system_error>
#include <thread>

#include <Eigen/Dense>

#include "euroc_vio/AbstractLoader.hpp"

class EvoSim3
{
public:
  EvoSim3(bool do_clean_up = true) : do_clean_up_{do_clean_up} {}

  ~EvoSim3() noexcept
  {
    // 删除临时文件
    std::error_code ec;
    if (do_clean_up_)
    {
      std::filesystem::remove(path_temp_tum_file_, ec);
    }
  }

  void Write(std::int64_t timestamp,
             const Eigen::Vector3d &estimated_position_fast,
             const Eigen::Quaterniond &estimated_attitude_fast)
  {
    std::print(fout_temp_evo_sim3_,
               // 时间戳
               "{:020d}, "
               // 位置
               "{:.18f}, {:.18f}, {:.18f}, "
               // 朝向
               "{:.18f}, {:.18f}, {:.18f}, {:.18f}\n",
               timestamp, estimated_position_fast.x(),
               estimated_position_fast.y(), estimated_position_fast.z(),
               estimated_attitude_fast.w(), estimated_attitude_fast.x(),
               estimated_attitude_fast.y(), estimated_attitude_fast.z());
  }

  /**
   * @brief 根据给定的真实轨迹，对估计轨迹进行 SIM(3) 变换
   * @param path_truth_csv 真实轨迹
   */
  void TransformSim3(std::string_view path_truth_csv)
  {
    fout_temp_evo_sim3_.flush();
    fout_temp_evo_sim3_.close();
    std::print(stderr, "[INFO] 估计轨迹已写入 {}\n",
               std::filesystem::absolute(path_temp_tum_file_).string());
    // 利用 evo 提供的 SIM(3) 变换调整轨迹
    std::system(
        std::format("bash -c '"
                    "source .venv/bin/activate && "
                    "yes 'y' | evo_traj euroc {} --ref={} "
                    "--align --correct_scale --save_as_tum'",
                    std::filesystem::absolute(path_temp_tum_file_).string(),
                    std::filesystem::absolute(path_truth_csv).string())
            .c_str()
    );
  }

  template <typename Callback>
  void Read(Callback callback)
  {
    std::ifstream fin_temp_evo_sim3{path_temp_tum_file_};
    std::string line;
    std::size_t line_num{0};
    while (std::getline(fin_temp_evo_sim3, line))
    {
      ++line_num;
      std::stringstream ss(line);
      try
      {
        // 读取时间戳
        const std::int64_t timestamp{
            static_cast<std::int64_t>(
                AbstractLoader::get_item_as_double(ss, ' ')
            ), // in nanoseconds
        };
        // 读取位置
        const double px{AbstractLoader::get_item_as_double(ss, ' ')};
        const double py{AbstractLoader::get_item_as_double(ss, ' ')};
        const double pz{AbstractLoader::get_item_as_double(ss, ' ')};
        // 读取朝向
        const double qw{AbstractLoader::get_item_as_double(ss, ' ')};
        const double qx{AbstractLoader::get_item_as_double(ss, ' ')};
        const double qy{AbstractLoader::get_item_as_double(ss, ' ')};
        const double qz{AbstractLoader::get_item_as_double(ss, ' ')};
        callback(timestamp, Eigen::Quaterniond{qw, qx, qy, qz},
                 Eigen::Vector3d{px, py, pz});
      }
      catch (const std::runtime_error &ex)
      {
        throw std::runtime_error{
            std::format("Fail to parse line #{} of file '{}':\n{}.\n"
                        "Triggered by:\n{}",
                        line_num, path_temp_tum_file_.string(), line, //
                        ex.what()),
        };
      }
    } // end while
    std::print(stderr, "[INFO] 估计轨迹已缩放.\n");
  }

private:
  bool do_clean_up_{true};
  // 在工作目录下，用临时文件 path_temp_evo_sim3.tum 存储 TUM 格式的数据
  // 它是利用 python 模块 evo，经过 SIM(3) 变换得到的相机轨迹数据
  const std::filesystem::path path_temp_tum_file_{
      "path_temp_evo_sim3.tum",
  };
  std::ofstream fout_temp_evo_sim3_{path_temp_tum_file_};
};
