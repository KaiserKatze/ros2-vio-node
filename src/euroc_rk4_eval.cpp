// euroc_rk4_eval.cpp
// 读取 EuRoC MAV 数据集 IMU 和 Ground Truth 数据，利用 message_filter 同步，进行 RK4 积分与误差输出
// 参考 ros2-euroc2bag 及本地 imustate.hpp

//=============================================================
// TO COMPILE THIS TEST, YOU NEED TO MODIFY CMAKE FILE:
//
// add_executable(euroc_rk4_eval test/euroc_rk4_eval.cpp)
// target_include_directories(euroc_rk4_eval PUBLIC
//   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
//   $<INSTALL_INTERFACE:include>
// )
// target_compile_features(euroc_rk4_eval PUBLIC cxx_std_20)
// install(TARGETS euroc_rk4_eval
//   DESTINATION debug/euroc_rk4_eval
// )
// target_link_libraries(euroc_rk4_eval
//   Eigen3::Eigen
//   Boost::boost
//   ${OpenCV_LIBS}
// )

//=============================================================
// RUN THIS TEST:
//
// ./$(find build -iname euroc_rk4_eval)

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>

#include "euroc_vio/ImuKinematics.hpp"
#include "euroc_vio/imustate.hpp"

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

double get_item(std::stringstream &ss)
{
  std::string item;
  std::getline(ss, item, ',');
  return std::stod(item);
}

// 简单 CSV 解析函数
std::vector<GroundTruthData> read_groundtruth_csv(const std::string &filename)
{
  std::vector<GroundTruthData> data;
  data.reserve(32767);
  std::ifstream file{filename};
  std::string line;
  // 跳过表头
  std::getline(file, line);
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    GroundTruthData gt;
    gt.timestamp = get_item(ss);
    for (int i = 0; i < 3; ++i)
    {
      gt.position[i] = get_item(ss);
    }
    for (int i = 0; i < 4; ++i)
    {
      gt.orientation[i] = get_item(ss);
    }
    for (int i = 0; i < 3; ++i)
    {
      gt.velocity[i] = get_item(ss);
    }
    for (int i = 0; i < 3; ++i)
    {
      gt.acc_bias[i] = get_item(ss);
    }
    for (int i = 0; i < 3; ++i)
    {
      gt.gyro_bias[i] = get_item(ss);
    }
    data.push_back(gt);
  }
  return data;
}

// 解析 EuRoC IMU 数据集 CSV
std::vector<ImuData> read_imu_csv(const std::string &filename)
{
  std::vector<ImuData> data;
  data.reserve(32767);
  std::ifstream file{filename};
  std::string line;
  // 跳过表头
  std::getline(file, line);
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    ImuData imu;
    imu.timestamp = get_item(ss); // 单位: ns
    for (int i = 0; i < 3; ++i)
    {
      imu.gyro[i] = get_item(ss); // rad/s
    }
    for (int i = 0; i < 3; ++i)
    {
      imu.acc[i] = get_item(ss); // m/s^2
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
  const double t0 = gt0.timestamp;
  const double t1 = gt1.timestamp;
  const double alpha
      = (t1 > t0) ? std::clamp((timestamp - t0) / (t1 - t0), 0.0, 1.0) : 0.0;
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

double orientation_error_angle(const double *est, const double *gt)
{
  Eigen::Quaterniond q_est(est[0], est[1], est[2], est[3]);
  Eigen::Quaterniond q_gt(gt[0], gt[1], gt[2], gt[3]);
  // 计算四元数角度误差（弧度）
  double angle = q_est.angularDistance(q_gt);
  return angle;
}

double orientation_error_frobenius(const double *est, const double *gt)
{
  Eigen::Quaterniond q_est(est[0], est[1], est[2], est[3]);
  Eigen::Quaterniond q_gt(gt[0], gt[1], gt[2], gt[3]);
  // 计算四元数对应的矩阵之差的 Frobenius 范数
  Eigen::Matrix3d R_est = q_est.toRotationMatrix();
  Eigen::Matrix3d R_gt  = q_gt.toRotationMatrix();
  double frob_norm      = (R_est - R_gt).norm();
  return frob_norm;
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

void test_rk4_piece_by_piece(const std::vector<GroundTruthData> &gt_data,
                             const std::vector<ImuData> &imu_data)
{
  // 主循环: 以每一帧 GroundTruth 为基准，寻找其间的 IMU 数据段，做一次积分
  for (size_t i = 0; i + 1 < gt_data.size(); ++i)
  {
    const auto &gt0 = gt_data[i];
    const auto &gt1 = gt_data[i + 1];
    const double t0 = gt0.timestamp;
    const double t1 = gt1.timestamp;
    // 找到区间内的 IMU 数据
    std::vector<ImuData> imu_segment;
    for (const auto &imu : imu_data)
    {
      if (t0 <= imu.timestamp && imu.timestamp < t1)
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

    // RK4 积分
    double ode_time = t0;
    boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4;

    std::cout << "time [s], "
                 "position error [m], "
                 "velocity error [m/s], "
                 "orientation error (by angle; rad), "
                 "orientation error (by Frobenius norm)\n";

    for (size_t k = 1; k < imu_segment.size(); ++k)
    {
      const auto &imu0 = imu_segment[k - 1];
      const auto &imu1 = imu_segment[k];
      // 对应时刻插值 GroundTruth Bias
      const auto gt_bias0 = interpolate_groundtruth(gt_data, imu0.timestamp);
      const auto gt_bias1 = interpolate_groundtruth(gt_data, imu1.timestamp);
      const Eigen::Vector3d acc0{
          imu0.acc[0] - gt_bias0.acc_bias[0],
          imu0.acc[1] - gt_bias0.acc_bias[1],
          imu0.acc[2] - gt_bias0.acc_bias[2],
      };
      const Eigen::Vector3d acc1{
          imu1.acc[0] - gt_bias1.acc_bias[0],
          imu1.acc[1] - gt_bias1.acc_bias[1],
          imu1.acc[2] - gt_bias1.acc_bias[2],
      };
      const Eigen::Vector3d gyro0{
          imu0.gyro[0] - gt_bias0.gyro_bias[0],
          imu0.gyro[1] - gt_bias0.gyro_bias[1],
          imu0.gyro[2] - gt_bias0.gyro_bias[2],
      };
      const Eigen::Vector3d gyro1{
          imu1.gyro[0] - gt_bias1.gyro_bias[0],
          imu1.gyro[1] - gt_bias1.gyro_bias[1],
          imu1.gyro[2] - gt_bias1.gyro_bias[2],
      };
      const double t_0 = imu0.timestamp;
      const double t_1 = imu1.timestamp;
      const double dt  = t_1 - t_0;
      ImuKinematicsODE ode({acc0, acc1, gyro0, gyro1, t_0, t_1});
      // RK4 单步
      rk4.do_step(ode, state, ode_time, dt);
      ode_time += dt;
      state.NormalizeQuaternion();
    }

    // 与下一帧 GroundTruth 对比
    std::cout << std::fixed << std::setprecision(6) << gt1.timestamp << ", "
              << position_error(&state[0], gt1.position) << ", "
              << velocity_error(&state[3], gt1.velocity) << ", "
              << orientation_error_angle(&state[6], gt1.orientation) << ", "
              << orientation_error_frobenius(&state[6], gt1.orientation)
              << "\n";
  }
}

// 只用真值提供的姿态、时间戳，不用真值位置、速度、零偏，不用 IMU 提供的角速度，目的是查看 EuRoC MAV 数据集的位置误差随 RK4 积分步数的变化幅度
void test_rk4_position(const std::vector<GroundTruthData> &gt_data,
                       const std::vector<ImuData> &imu_data)
{
  ImuState state;
  const auto &gt_first = gt_data[0];
  state.SetPosition(gt_first.position[0], gt_first.position[1],
                    gt_first.position[2]);
  state.SetVelocity(gt_first.velocity[0], gt_first.velocity[1],
                    gt_first.velocity[2]);

  double ode_time             = gt_first.timestamp;
  const Eigen::Vector3d gyro0 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d gyro1 = Eigen::Vector3d::Zero();
  boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4;

  std::cout << "time [s], "
               "position error [m], "
               "velocity error [m/s]\n";

  for (size_t i = 0; i + 1 < imu_data.size(); ++i)
  {
    const auto &imu0 = imu_data[i];

    // 跳过早于第一个真值数据的所有 IMU 数据
    if (imu0.timestamp < ode_time)
    {
      continue;
    }

    const auto &imu1 = imu_data[i + 1];
    const double t_0 = imu0.timestamp;
    const double t_1 = imu1.timestamp;
    const double dt  = t_1 - t_0;

    // 查找最接近 imu1.timestamp 的 GroundTruth 数据
    const auto &gt1 = interpolate_groundtruth(gt_data, imu1.timestamp);
    state.SetQuaternion(gt1.orientation[0], gt1.orientation[1],
                        gt1.orientation[2], gt1.orientation[3]);

    const Eigen::Vector3d acc0{
        imu0.acc[0], // - gt0.acc_bias[0],
        imu0.acc[1], // - gt0.acc_bias[1],
        imu0.acc[2], // - gt0.acc_bias[2],
    };
    const Eigen::Vector3d acc1{
        imu1.acc[0], // - gt1.acc_bias[0],
        imu1.acc[1], // - gt1.acc_bias[1],
        imu1.acc[2], // - gt1.acc_bias[2],
    };
    ImuKinematicsODE ode({acc0, acc1, gyro0, gyro1, t_0, t_1});
    // RK4 单步
    rk4.do_step(ode, state, ode_time, dt);
    ode_time += dt;

    // 与下一帧 GroundTruth 对比
    std::cout << std::fixed << std::setprecision(6) << gt1.timestamp << ", "
              << position_error(&state[0], gt1.position) << ", "
              << velocity_error(&state[3], gt1.velocity) << "\n";
  }
}

// 忽略位置积分，只对朝向导数进行积分，检查 RK4 积分的误差
void test_rk4_orientation(const std::vector<GroundTruthData> &gt_data,
                          const std::vector<ImuData> &imu_data)
{
  ImuState state;
  const auto &gt_first = gt_data[0];
  // state.SetPosition(gt_first.position[0], gt_first.position[1],
  //                   gt_first.position[2]);
  // state.SetVelocity(gt_first.velocity[0], gt_first.velocity[1],
  //                   gt_first.velocity[2]);
  state.SetQuaternion(gt_first.orientation[0], gt_first.orientation[1],
                      gt_first.orientation[2], gt_first.orientation[3]);

  double ode_time            = gt_first.timestamp;
  const Eigen::Vector3d acc0 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d acc1 = Eigen::Vector3d::Zero();
  boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4;

  std::cout << "time [s], "
               "orientation error (by angle; rad), "
               "orientation error (by Frobenius norm)\n";

  for (size_t i = 0; i + 1 < imu_data.size(); ++i)
  {
    const auto &imu0 = imu_data[i];

    // 跳过早于第一个真值数据的所有 IMU 数据
    if (imu0.timestamp < ode_time)
    {
      continue;
    }

    const auto &imu1 = imu_data[i + 1];
    const double t_0 = imu0.timestamp;
    const double t_1 = imu1.timestamp;
    const double dt  = t_1 - t_0;
    // 查找最接近 imu1.timestamp 的 GroundTruth 数据
    const auto &gt1 = interpolate_groundtruth(gt_data, imu1.timestamp);
    const Eigen::Vector3d gyro0{
        imu0.gyro[0], // - gt0.gyro_bias[0],
        imu0.gyro[1], // - gt0.gyro_bias[1],
        imu0.gyro[2], // - gt0.gyro_bias[2],
    };
    const Eigen::Vector3d gyro1{
        imu1.gyro[0], // - gt1.gyro_bias[0],
        imu1.gyro[1], // - gt1.gyro_bias[1],
        imu1.gyro[2], // - gt1.gyro_bias[2],
    };
    ImuKinematicsODE ode({acc0, acc1, gyro0, gyro1, t_0, t_1});
    // RK4 单步
    rk4.do_step(ode, state, ode_time, dt);
    ode_time += dt;
    state.NormalizeQuaternion();

    // 与下一帧 GroundTruth 对比
    std::cout << std::fixed << std::setprecision(6) << gt1.timestamp << ", "
              << orientation_error_angle(&state[6], gt1.orientation) << ", "
              << orientation_error_frobenius(&state[6], gt1.orientation)
              << "\n";
  }
}

// 忽略位置积分，将 IMU 提供的三轴角速度按照每一个分量分别积分，得到一个三维向量
void test_rk4_motionless_gyro_error_caused_by_bias(
    const std::vector<ImuData> &imu_data
)
{
  if (imu_data.empty())
  {
    return;
  }
  Eigen::Vector3d state = Eigen::Vector3d::Zero();
  auto itr              = imu_data.cbegin();
  if (itr == imu_data.cend())
  {
    return;
  }
  const double time_start = itr->timestamp;
  double time_prev        = time_start;

  std::cout << "time [s], x, y, z\n";

  for (; itr != imu_data.cend(); ++itr)
  {
    const ImuData &imu        = *itr;
    const double time_now     = imu.timestamp;
    const double time_elapsed = time_now - time_start;
    // 这里之所以取 2.5 秒，是因为根据 EuRoC MAV 数据集中相机拍摄的图像
    // 在第50个图像帧以后，无人机才开始运动，之前一直处于静止状态
    if (time_elapsed > 2.5)
    {
      std::cerr << "在 [t=" << time_elapsed << "] 时截断输出. "
                << "最终输出样本个数: " << std::distance(imu_data.cbegin(), itr)
                << "\n";
      break;
    }
    double time_delta = time_now - time_prev;
    time_prev         = time_now;
    Eigen::Vector3d gyro{
        imu.gyro[0],
        imu.gyro[1],
        imu.gyro[2],
    };
    gyro *= time_delta;
    state += gyro;

    std::cout << std::fixed << std::setprecision(6) << time_now << ", "
              << state(0) << ", " << state(1) << ", " << state(2) << "\n";
  }
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "用法: " << argv[0] << " <imu.csv> <groundtruth.csv>"
              << "\n";
    return 1;
  }
  std::string imu_file = argv[1];
  std::string gt_file  = argv[2];
  auto gt_data         = read_groundtruth_csv(gt_file);
  auto imu_data        = read_imu_csv(imu_file);
  std::cerr << "GroundTruth 数据量: " << gt_data.size() << "\n"
            << "IMU 数据量: " << imu_data.size() << "\n";
  // 单位换算: ns -> s
  double imu_timestamp_min = std::numeric_limits<double>::max();
  double imu_timestamp_max = std::numeric_limits<double>::min();
  for (auto &imu : imu_data)
  {
    imu.timestamp *= 1e-9;

    imu_timestamp_min = std::min(imu_timestamp_min, imu.timestamp);
    imu_timestamp_max = std::max(imu_timestamp_max, imu.timestamp);

    double ax   = imu.acc[0];
    double ay   = imu.acc[1];
    double az   = imu.acc[2];
    imu.acc[0]  = az;
    imu.acc[1]  = -ay;
    imu.acc[2]  = ax;
    double gx   = imu.gyro[0];
    double gy   = imu.gyro[1];
    double gz   = imu.gyro[2];
    imu.gyro[0] = gz;
    imu.gyro[1] = -gy;
    imu.gyro[2] = gx;
  }
  for (auto &gt : gt_data)
  {
    gt.timestamp *= 1e-9;
  }

  std::cerr << std::fixed << std::setprecision(6)
            << "IMU.TimeStamp[min]=" << imu_timestamp_min
            << " seconds since Unix epoch\n"
            << "IMU.TimeStamp[max]=" << imu_timestamp_max
            << " seconds since Unix epoch\n"
            << "IMU.TimeInterval=" << (imu_timestamp_max - imu_timestamp_min)
            << " seconds\n"
            << std::setprecision(2) << "IMU.AverageFrequency="
            << (static_cast<double>(imu_data.size())
                / (imu_timestamp_max - imu_timestamp_min))
            << " Hz\n"
            << "\n";

  // test_rk4_position(gt_data, imu_data);
  // test_rk4_orientation(gt_data, imu_data);
  test_rk4_motionless_gyro_error_caused_by_bias(imu_data);

  return 0;
}
