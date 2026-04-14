// euroc_rk4_eval.cpp
// 读取 EuRoC MAV 数据集 IMU 和 Ground Truth 数据，利用 message_filter 同步，进行 RK4 积分与误差输出
// 参考 ros2-euroc2bag 及本地 imustate.hpp

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/imustate.hpp"
#include "euroc_vio/imuworker.hpp"
#include "euroc_vio/util.hpp"

struct ImuData
{
  double timestamp;
  double acc[3];
  double gyro[3];
};

struct GroundTruthData
{
  double timestamp;
  double position[3];
  double orientation[4]; // quaternion (w, x, y, z)
  double velocity[3];
  double acc_bias[3];
  double gyro_bias[3];
};

// 简单 CSV 解析函数
std::vector<GroundTruthData> read_groundtruth_csv(const std::string &filename)
{
  std::vector<GroundTruthData> data;
  std::ifstream file(filename);
  std::string line;
  // 跳过表头
  std::getline(file, line);
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string item;
    GroundTruthData gt;
    std::getline(ss, item, ',');
    gt.timestamp = std::stod(item);
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      gt.position[i] = std::stod(item);
    }
    for (int i = 0; i < 4; ++i)
    {
      std::getline(ss, item, ',');
      gt.orientation[i] = std::stod(item);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      gt.velocity[i] = std::stod(item);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      gt.acc_bias[i] = std::stod(item);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      gt.gyro_bias[i] = std::stod(item);
    }
    data.push_back(gt);
  }
  return data;
}

// 解析 EuRoC IMU 数据集 CSV
std::vector<ImuData> read_imu_csv(const std::string &filename)
{
  std::vector<ImuData> data;
  std::ifstream file(filename);
  std::string line;
  // 跳过表头
  std::getline(file, line);
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string item;
    ImuData imu;
    std::getline(ss, item, ',');
    imu.timestamp = std::stod(item); // 单位: ns
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      imu.gyro[i] = std::stod(item); // rad/s
    }
    for (int i = 0; i < 3; ++i)
    {
      std::getline(ss, item, ',');
      imu.acc[i] = std::stod(item); // m/s^2
    }
    data.push_back(imu);
  }
  return data;
}

// 线性插值查找最近的 GroundTruth 状态
GroundTruthData
interpolate_groundtruth(const std::vector<GroundTruthData> &gt_data,
                        double timestamp)
{
  // timestamp 单位: ns
  if (gt_data.empty())
  {
    throw std::runtime_error("GroundTruth 数据为空");
  }
  if (timestamp <= gt_data.front().timestamp)
  {
    return gt_data.front();
  }
  if (timestamp >= gt_data.back().timestamp)
  {
    return gt_data.back();
  }
  // 二分查找
  size_t left = 0, right = gt_data.size() - 1;
  while (left + 1 < right)
  {
    size_t mid = (left + right) / 2;
    if (gt_data[mid].timestamp < timestamp)
    {
      left = mid;
    }
    else
    {
      right = mid;
    }
  }
  const auto &gt0 = gt_data[left];
  const auto &gt1 = gt_data[right];
  double t0 = gt0.timestamp, t1 = gt1.timestamp;
  double alpha           = (timestamp - t0) / (t1 - t0);
  GroundTruthData interp = gt0;
  for (int i = 0; i < 3; ++i)
  {
    interp.position[i]
        = gt0.position[i] * (1 - alpha) + gt1.position[i] * alpha;
    interp.velocity[i]
        = gt0.velocity[i] * (1 - alpha) + gt1.velocity[i] * alpha;
    interp.acc_bias[i]
        = gt0.acc_bias[i] * (1 - alpha) + gt1.acc_bias[i] * alpha;
    interp.gyro_bias[i]
        = gt0.gyro_bias[i] * (1 - alpha) + gt1.gyro_bias[i] * alpha;
  }
  // 四元数球面线性插值
  Eigen::Quaterniond q0(gt0.orientation[0], gt0.orientation[1],
                        gt0.orientation[2], gt0.orientation[3]);
  Eigen::Quaterniond q1(gt1.orientation[0], gt1.orientation[1],
                        gt1.orientation[2], gt1.orientation[3]);
  Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
  interp.orientation[0]       = q_interp.w();
  interp.orientation[1]       = q_interp.x();
  interp.orientation[2]       = q_interp.y();
  interp.orientation[3]       = q_interp.z();
  interp.timestamp            = timestamp;
  return interp;
}

// 计算欧氏距离
double position_error(const double *est, const double *gt)
{
  double err = 0.0;
  for (int i = 0; i < 3; ++i)
  {
    err += (est[i] - gt[i]) * (est[i] - gt[i]);
  }
  return std::sqrt(err);
}

// 计算四元数角度误差（弧度）
double orientation_error(const double *est, const double *gt)
{
  Eigen::Quaterniond q_est(est[0], est[1], est[2], est[3]);
  Eigen::Quaterniond q_gt(gt[0], gt[1], gt[2], gt[3]);
  double angle = q_est.angularDistance(q_gt);
  return angle;
}

// 计算速度误差
double velocity_error(const double *est, const double *gt)
{
  double err = 0.0;
  for (int i = 0; i < 3; ++i)
  {
    err += (est[i] - gt[i]) * (est[i] - gt[i]);
  }
  return std::sqrt(err);
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "用法: " << argv[0] << " <imu.csv> <groundtruth.csv>"
              << std::endl;
    return 1;
  }
  std::string imu_file = argv[1];
  std::string gt_file  = argv[2];
  auto gt_data         = read_groundtruth_csv(gt_file);
  auto imu_data        = read_imu_csv(imu_file);
  std::cout << "GroundTruth 数据量: " << gt_data.size() << std::endl;
  std::cout << "IMU 数据量: " << imu_data.size() << std::endl;
  // 单位换算: ns -> s
  for (auto &imu : imu_data)
  {
    imu.timestamp *= 1e-9;
  }
  for (auto &gt : gt_data)
  {
    gt.timestamp *= 1e-9;
  }

  // 主循环: 以每一帧 GroundTruth 为基准，寻找其间的 IMU 数据段，做一次积分
  for (size_t i = 0; i + 1 < gt_data.size(); ++i)
  {
    const auto &gt0 = gt_data[i];
    const auto &gt1 = gt_data[i + 1];
    double t0 = gt0.timestamp, t1 = gt1.timestamp;
    // 找到区间内的 IMU 数据
    std::vector<ImuData> imu_segment;
    for (const auto &imu : imu_data)
    {
      if (imu.timestamp >= t0 && imu.timestamp < t1)
      {
        imu_segment.push_back(imu);
      }
    }
    if (imu_segment.size() < 2)
    {
      continue;
    }

    // 初始化 ImuState
    ImuState state;
    state.SetPosition(gt0.position[0], gt0.position[1], gt0.position[2]);
    state.SetVelocity(gt0.velocity[0], gt0.velocity[1], gt0.velocity[2]);
    state.SetQuaternion(gt0.orientation[0], gt0.orientation[1],
                        gt0.orientation[2], gt0.orientation[3]);
    state.NormalizeQuaternion();

    // RK4 积分
    double ode_time = t0;
    for (size_t k = 1; k < imu_segment.size(); ++k)
    {
      const auto &imu0 = imu_segment[k - 1];
      const auto &imu1 = imu_segment[k];
      // 对应时刻插值 GroundTruth Bias
      auto gt_bias0 = interpolate_groundtruth(gt_data, imu0.timestamp);
      auto gt_bias1 = interpolate_groundtruth(gt_data, imu1.timestamp);
      Eigen::Vector3d acc0(imu0.acc[0] - gt_bias0.acc_bias[0],
                           imu0.acc[1] - gt_bias0.acc_bias[1],
                           imu0.acc[2] - gt_bias0.acc_bias[2]);
      Eigen::Vector3d acc1(imu1.acc[0] - gt_bias1.acc_bias[0],
                           imu1.acc[1] - gt_bias1.acc_bias[1],
                           imu1.acc[2] - gt_bias1.acc_bias[2]);
      Eigen::Vector3d gyro0(imu0.gyro[0] - gt_bias0.gyro_bias[0],
                            imu0.gyro[1] - gt_bias0.gyro_bias[1],
                            imu0.gyro[2] - gt_bias0.gyro_bias[2]);
      Eigen::Vector3d gyro1(imu1.gyro[0] - gt_bias1.gyro_bias[0],
                            imu1.gyro[1] - gt_bias1.gyro_bias[1],
                            imu1.gyro[2] - gt_bias1.gyro_bias[2]);
      double t_0 = imu0.timestamp, t_1 = imu1.timestamp;
      double dt = t_1 - t_0;
      ImuKinematicsODE ode({acc0, acc1, gyro0, gyro1, t_0, t_1});
      ImuDerivative dxdt;
      // RK4 单步
      boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4;
      rk4.do_step(ode, state, ode_time, dt);
      ode_time += dt;
      state.NormalizeQuaternion();
    }

    // 与下一帧 GroundTruth 对比
    double pos_err = position_error(&state[0], gt1.position);
    double ori_err = orientation_error(&state[6], gt1.orientation);
    double vel_err = velocity_error(&state[3], gt1.velocity);
    std::cout << std::fixed << std::setprecision(6) << "t_gt1=" << gt1.timestamp
              << " pos_err=" << pos_err << " ori_err(rad)=" << ori_err
              << " vel_err=" << vel_err << std::endl;
  }
  return 0;
}
