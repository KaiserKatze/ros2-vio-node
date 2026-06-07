#pragma once

#include <filesystem>
#include <fstream>
#include <print>
#include <vector>

#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include "DatumTruth.hpp"

class ErrorEvaluation
{
public:
  ErrorEvaluation(const std::filesystem::path &path_err_eval,
                  const std::vector<DatumTruth> &data_truth) :
    path_err_eval_{path_err_eval}, fout_err_eval_{path_err_eval},
    data_truth_{data_truth}
  {
    std::print(fout_err_eval_,
               "time [s],"
               "qw [],qx [],qy [],qz [],"
               "err(qx) [m],err(qy) [m],err(qz) [m],"
               "x [m],y [m],z [m],"
               "err(x) [m],err(y) [m],err(z) [m],"
               "vx [m s^-1],vy [m s^-1],vz [m s^-1],"
               "err(vx) [m s^-1],err(vy) [m s^-1],err(vz) [m s^-1]\n");
  }

  ~ErrorEvaluation()
  {
    fout_err_eval_.flush();
    std::print(stderr, "误差评估文件已写入 {}\n",
               std::filesystem::absolute(path_err_eval_).string());
  }

  void WriteErrorEvaluation(const std::int64_t timestamp,
                            const Sophus::SO3d &estimated_attitude,
                            const Eigen::Vector3d &estimated_position,
                            const Eigen::Vector3d &estimated_linear_velocity)
  {
    const DatumTruth datum_true{Interpolate(data_truth_, timestamp)};
    const Eigen::Vector3d angle_error{
        (Sophus::SO3d{datum_true.attitude_}.inverse() * estimated_attitude)
            .log()
    };
    const Eigen::Vector3d position_error{
        (estimated_position - datum_true.position_).cwiseAbs()
    };
    const Eigen::Vector3d linear_velocity_error{
        (estimated_linear_velocity - datum_true.velocity_).cwiseAbs()
    };
    // 更新统计信息
    std::print(
        fout_err_eval_,
        // 时间戳
        "{:020d}, "
        // 朝向
        "{:.18f},{:.18f},{:.18f},{:.18f},"
        // 朝向误差
        "{:.18f},{:.18f},{:.18f},"
        // 位置
        "{:.18f},{:.18f},{:.18f},"
        // 位置绝对误差
        "{:.18f},{:.18f},{:.18f},"
        // 线速度
        "{:.18f},{:.18f},{:.18f},"
        // 线速度绝对误差
        "{:.18f},{:.18f},{:.18f}\n",
        datum_imu.timestamp_,          // 时间戳
        estimated_quaternion.w(),      // 朝向
        estimated_quaternion.x(),      //
        estimated_quaternion.y(),      //
        estimated_quaternion.z(),      //
        angle_error.x(),               // 轴角误差
        angle_error.y(),               //
        angle_error.z(),               //
        estimated_position.x(),        // 位置
        estimated_position.y(),        //
        estimated_position.z(),        //
        position_error.x(),            // 位置误差
        position_error.y(),            //
        position_error.z(),            //
        estimated_linear_velocity.x(), // 线速度
        estimated_linear_velocity.y(), //
        estimated_linear_velocity.z(), //
        linear_velocity_error.x(),     // 线速度误差
        linear_velocity_error.y(),     //
        linear_velocity_error.z()      //
    );
  }

private:
  std::filesystem::path path_err_eval_;
  std::ofstream fout_err_eval_;
  std::vector<DatumTruth> data_truth_;
};
