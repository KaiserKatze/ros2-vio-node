#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include "MeshPlot.hpp"
#include "Path.hpp"
#include "Room.hpp"
#include "StereoRig.hpp"
#include "euroc_vio/Reflect.hpp"

using namespace std::chrono_literals;

#define START_VISUALIZATION 0
#define OUTPUT_AS_EUROC 1

template <typename value_type>
struct VisualSim
{
  // 重力加速度
  const value_type gravity_world_norm_{9.81}; // m s^-2
  // 传入长宽高的划分段数
  Room<value_type> room_{};
  // 初始化专属绘制器
  MeshPlot<value_type> mesh_plot_{
      /* create_named_window */
      static_cast<bool>(START_VISUALIZATION),
  };
  // 仿真双目相机
  StereoRig<value_type> rig_{};
  // 仿真双目相机运动路径
  using OrientationMode = Path<value_type>::OrientationMode;
  OrientationMode orientation_mode_{OrientationMode::LookAtCenter};
  Path<value_type> path_{room_, orientation_mode_};
  const value_type time_limit_simulation_{
      // 计算匀速圆周运动恰好旋转两周所需的时间
      std::round(4 * std::numbers::pi_v<value_type> / path_.omega_
                 + path_.time_static_),
  };
  // 相机的时间步长 (单位: 秒) (采用 0.05 秒作为时间步长可以让仿真相机的采样率保持为 20 赫兹)
  const value_type step_{static_cast<value_type>(0.05)};
  // 仿真 IMU 与仿真相机的采样率之比
  const int rate_ratio_{10};
  // 真值和 IMU 的时间步长 (单位: 秒)
  const value_type imu_step_{step_ / rate_ratio_};

  using Point3     = Eigen::Vector<value_type, 3>;
  using Point2     = Eigen::Vector<value_type, 2>;
  using Attitude   = Eigen::Matrix<value_type, 3, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Frame      = typename StereoRig<value_type>::Frame;

  VisualSim() : mesh_plot_{room_}
  {
    // 只修改双目相机的基线长度
    rig_.camera_right_.translation_ = {-0.1, 0.0, 0.0};
  }

  static cv::Mat eigen2cv(const std::vector<Point2> &image_points)
  {
    cv::Mat cv_image_points(2, static_cast<int>(image_points.size()), CV_32F);

    for (size_t i = 0; i < image_points.size(); ++i)
    {
      const auto &image_point{image_points[i]};
      const value_type x{image_point.x()};
      const value_type y{image_point.y()};
      cv_image_points.at<float>(0, i) = x;
      cv_image_points.at<float>(1, i) = y;
    }

    return cv_image_points;
  }

#pragma region WRITE_EUROC_CONFIG

  void WriteCameraConfig(const std::filesystem::path &path_cam,
                         const Camera<value_type> &camera) const
  {
    std::ofstream fout_cam(path_cam / "sensor.yaml");
    std::print(fout_cam,
               "sensor_type: camera\n\n"
               "T_BS:\n"
               "  cols: 4\n"
               "  rows: 4\n"
               "  data: [{:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
               "         {:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
               "         {:.4f}, {:.4f}, {:.4f}, {:.4f},\n"
               "         0.0, 0.0, 0.0, 1.0]\n\n"
               "rate_hz: {}\n"
               "resolution: [{}, {}]\n"
               "camera_model: pinhole\n"
               "intrinsics: [{:.3f}, {:.3f}, {:.3f}, {:.3f}] # fu, fv, cu, cv\n"
               "distortion_model: radial-tangential\n"
               "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n",
               // 空间变换的齐次矩阵形式
               camera.rotation_(0, 0), camera.rotation_(0, 1),
               camera.rotation_(0, 2), camera.translation_(0),
               camera.rotation_(1, 0), camera.rotation_(1, 1),
               camera.rotation_(1, 2), camera.translation_(1),
               camera.rotation_(2, 0), camera.rotation_(2, 1),
               camera.rotation_(2, 2), camera.translation_(2),
               // 采样频率
               static_cast<value_type>(1.0) / step_,
               // 分辨率
               camera.width_, camera.height_,
               // 相机内参
               camera.intrinsic_(0, 0), camera.intrinsic_(1, 1),
               camera.intrinsic_(0, 2), camera.intrinsic_(1, 2));
  }

  void WriteImuConfig(const std::filesystem::path &path_imu) const
  {
    std::ofstream fout_imu(path_imu / "sensor.yaml");
    std::print(fout_imu,
               "sensor_type: imu\n\n"
               "T_BS:\n"
               "  cols: 4\n"
               "  rows: 4\n"
               "  data: [1.0, 0.0, 0.0, 0.0,\n"
               "         0.0, 1.0, 0.0, 0.0,\n"
               "         0.0, 0.0, 1.0, 0.0,\n"
               "         0.0, 0.0, 0.0, 1.0]\n"
               "rate_hz: {}\n\n"
               "# inertial sensor noise model parameters (static)\n"
               "gyroscope_noise_density:     {:.4e} "
               "# [ rad / s / sqrt(Hz) ]   ( gyro \"white noise\" )\n"
               "gyroscope_random_walk:       {:.4e} "
               "# [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )\n"
               "accelerometer_noise_density: {:.4e} "
               "# [ m / s^2 / sqrt(Hz) ]   ( accel \"white noise\" )\n"
               "accelerometer_random_walk:   {:.4e} "
               "# [ m / s^3 / sqrt(Hz) ]   ( accel bias diffusion )\n",
               // 采样频率
               static_cast<value_type>(rate_ratio_) / step_,
               // 陀螺仪白噪声功率密度
               0.0,
               // 陀螺仪随机游走
               0.0,
               // 加速度计白噪声功率密度
               0.0,
               // 加速度计随机游走
               0.0);
  }

  void
  WriteGroundTruthConfig(const std::filesystem::path &path_groundtruth) const
  {
    std::ofstream fout(path_groundtruth / "sensor.yaml");
    std::print(fout, "sensor_type: visual-inertial\n\n"
                     "sensor_type: camera\n\n"
                     "T_BS:\n"
                     "  cols: 4\n"
                     "  rows: 4\n"
                     "  data: [1.0, 0.0, 0.0, 0.0,\n"
                     "         0.0, 1.0, 0.0, 0.0,\n"
                     "         0.0, 0.0, 1.0, 0.0,\n"
                     "         0.0, 0.0, 0.0, 1.0]\n");
  }

#pragma endregion

  std::pair<Point3, Attitude> GetPose(value_type time) const
  {
    return path_.GetPose(time);
  }

  void Start()
  {
#pragma region WRITE_EUROC_CONFIG

#if (OUTPUT_AS_EUROC)
    // 创建 mav0 目录
    std::error_code ec;
    std::filesystem::path path_mav0{"mav0"};
    std::filesystem::remove_all(path_mav0, ec);
    if (std::filesystem::create_directories(path_mav0, ec))
    {
      std::print(stderr, "[INFO] 工作目录创建成功: {}\n",
                 std::filesystem::absolute(path_mav0).string());
    }
    else
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0 目录
    std::filesystem::path path_cam0{path_mav0 / "cam0"};
    if (!std::filesystem::create_directories(path_cam0, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0/data 目录
    std::filesystem::path path_cam0_data{path_mav0 / "cam0" / "data"};
    if (!std::filesystem::create_directories(path_cam0_data, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1 目录
    std::filesystem::path path_cam1{path_mav0 / "cam1"};
    if (!std::filesystem::create_directories(path_cam1, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1/data 目录
    std::filesystem::path path_cam1_data{path_mav0 / "cam1" / "data"};
    if (!std::filesystem::create_directories(path_cam1_data, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/state_groundtruth_estimate0 目录
    std::filesystem::path path_groundtruth{path_mav0
                                           / "state_groundtruth_estimate0"};
    if (!std::filesystem::create_directories(path_groundtruth, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/imu0 目录
    std::filesystem::path path_imu0{path_mav0 / "imu0"};
    if (!std::filesystem::create_directories(path_imu0, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }

    // 输出 mav0/cam0/sensor.yaml
    WriteCameraConfig(path_cam0, rig_.camera_left_);
    // 输出 mav0/cam1/sensor.yaml
    WriteCameraConfig(path_cam1, rig_.camera_right_);
    // 输出 mav0/state_groundtruth_estimate0/sensor.yaml
    WriteGroundTruthConfig(path_groundtruth);
    // 输出 mav0/imu0/sensor.yaml
    WriteImuConfig(path_imu0);

    // 输出 mav0/README.md 说明文件
    std::ofstream fout_readme(path_mav0 / "README.md");
    std::print(fout_readme,
               "Room:\n"
               "\tWidth:  {:.2f} [m]  <!-- 房间开间 -->\n"
               "\tDepth:  {:.2f} [m]  <!-- 房间进深 -->\n"
               "\tHeight: {:.2f} [m]  <!-- 房间层高 -->\n"
               "Movement Paradigm: {}  <!-- 无人机的运动范式 -->\n"
               "Movement Stage #1\n"
               "\tTime Length Before Takeoff: {:.2f} [s]  "
               "<!-- 无人机起飞前处于静止状态的时间长度 (单位: 秒) -->\n",
               room_.width_, room_.depth_,
               room_.height_,                     //                    //
               enum_to_string(orientation_mode_), //
               path_.time_static_);
    switch (orientation_mode_)
    {
    case OrientationMode::LookAtCenter:
    case OrientationMode::BackToCenter:
    case OrientationMode::Tangent:
    case OrientationMode::Upward:
    {
      // 做匀速圆周运动时的运动半径
      const value_type radius{path_.GetRadius()};
      // 做匀速圆周运动时的线速度大小
      const value_type v0{radius * path_.omega_};
      // 做匀速圆周运动之前，做匀加速直线运动所需的时长
      const value_type t0{static_cast<value_type>(2.0) * radius / v0};
      // 做匀加速直线运动时的线加速度大小
      const value_type a0{static_cast<value_type>(0.5) * v0 * path_.omega_};
      // 做匀速圆周运动时的线加速度大小
      const value_type a1{path_.omega_ * v0};
      std::print(
          fout_readme,
          "Movement Stage #2\n"
          "\tTime: {:.2f} [s]  <!-- 匀加速直线运动 时间 -->\n"
          "\tDistance: {:.2f} [m]  <!-- 匀加速直线运动 位移 -->\n"
          "\tAcceleration: {:.2f} [m s^-2]  <!-- 匀加速直线运动 线加速度 -->\n"
          "Movement Stage #3\n"
          "\tRadius: {:.2f} [m]  <!-- 匀速圆周运动 轨道半径 -->\n"
          "\tLinear Velocity: {:.2f} [m s^-1]  <!-- 匀速圆周运动 线速度 -->\n"
          "\tLinear Acceleration: {:.2f} [m s^-2]"
          "  <!-- 匀速圆周运动 线加速度 -->\n"
          "\tAngular Velocity: {:.2f} [rad s^-1]"
          "  <!-- 匀速圆周运动 角速度 -->\n",
          t0, radius, a0, //
          radius, v0, a1, path_.omega_
      );
      break;
    }
    case OrientationMode::StraightLine:
    case OrientationMode::Parabola:
      break;
    }

    // 输出 mav0/cam0/data.csv 表头
    std::ofstream fout_cam0_data_csv(path_cam0 / "data.csv");
    std::print(fout_cam0_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/cam1/data.csv 表头
    std::ofstream fout_cam1_data_csv(path_cam1 / "data.csv");
    std::print(fout_cam1_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/state_groundtruth_estimate0/data.csv 表头
    std::ofstream fout_groundtruth_csv(path_groundtruth / "data.csv");
    std::print(
        fout_groundtruth_csv,
        "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
        "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
        "a_RS_R_x [m s^-2], a_RS_R_y [m s^-2], a_RS_R_z [m s^-2], "
        "w_RS_R_x [rad s^-1], w_RS_R_y [rad s^-1], w_RS_R_z [rad s^-1]\n"
    );
    // 输出 mav0/imu0/data.csv 表头
    std::ofstream fout_imu0_data_csv(path_imu0 / "data.csv");
    std::print(fout_imu0_data_csv,
               "#timestamp [ns],"
               "w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
               "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n");
#endif

#pragma endregion

    const Point3 gravity_world{0.0, 0.0, -gravity_world_norm_};

    for (value_type time = 0.0; time < time_limit_simulation_; time += step_)
    {
      std::print(stderr, "[INFO] 时间 = ({:.3f}).\n", time);

#if (OUTPUT_AS_EUROC)
      const auto timestamp_ns{static_cast<std::int64_t>(time * 1e9)};
      std::print(fout_cam0_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
      std::print(fout_cam1_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
#endif

      const Frame frame{path_.GetImage(rig_, time)};

      Point3 true_current_position{Point3::Zero()};
      Quaternion true_current_attitude{Quaternion::Identity()};
      std::tie(true_current_position, true_current_attitude) = GetPose(time);

#pragma region GENERATE_IMU_AND_GROUNDTRUTH_DATA

#if (OUTPUT_AS_EUROC)
      // 保证 IMU 与 Ground Truth 的数据帧率相同
      for (int i = 0; i < rate_ratio_; ++i)
      {
        // IMU 的当前时间戳
        const value_type imu_time{time + i * imu_step_};
        const auto imu_timestamp_ns{
            static_cast<std::int64_t>(imu_time * 1e9),
        };
        // 角速度矢量 $\omega^{iv}_i$
        Point3 imu_angular_velocity_world{Point3::Zero()};
        // 线速度矢量 $\dot{r}^{iv}_i$
        Point3 imu_linear_velocity_world{Point3::Zero()};
        // 线加速度矢量 $\ddot{r}^{iv}_i$
        Point3 imu_linear_acceleration_world{Point3::Zero()};
        // 获取 IMU 在世界坐标系下的线速度、角速度、线加速度
        path_.GetKinematics(imu_time, imu_linear_velocity_world,
                            imu_angular_velocity_world,
                            imu_linear_acceleration_world);

        // 位置 $r^{vi}_i$
        Point3 imu_position{Point3::Zero()};
        // 朝向 $C_{iv}$
        Attitude imu_attitude{Attitude::Identity()};
        // 获取 IMU 在世界坐标系下的位置、朝向
        std::tie(imu_position, imu_attitude) = GetPose(imu_time);
        Quaternion imu_attitude_quat{imu_attitude};

        // 输出仿真 Ground Truth 数据
        std::print(fout_groundtruth_csv,
                   // 时间戳
                   "{:020d}, "
                   // 位置
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 朝向
                   "{:.18f}, {:.18f}, {:.18f}, {:.18f}, "
                   // 线速度
                   "{:.18f}, {:.18f}, {:.18f},"
                   // 线加速度
                   "{:.18f}, {:.18f}, {:.18f},"
                   // 角速度
                   "{:.18f}, {:.18f}, {:.18f}\n",
                   imu_timestamp_ns, //
                   imu_position.x(), imu_position.y(),
                   imu_position.z(), //
                   imu_attitude_quat.w(), imu_attitude_quat.x(),
                   imu_attitude_quat.y(),
                   imu_attitude_quat.z(), //
                   imu_linear_velocity_world.x(), imu_linear_velocity_world.y(),
                   imu_linear_velocity_world.z(), //
                   imu_linear_acceleration_world.x(),
                   imu_linear_acceleration_world.y(),
                   imu_linear_acceleration_world.z(), //
                   imu_angular_velocity_world.x(),
                   imu_angular_velocity_world.y(),
                   imu_angular_velocity_world.z());

        // 转换坐标系：从世界坐标系转为传感器坐标系

        Point3 imu_angular_velocity_sensor{
            // 假设传感器坐标系与载具坐标系重合
            // 那么在零误差的情况下 IMU 测得的传感器坐标系下的角速度
            // 就等于载具坐标系下的角速度
            imu_attitude.transpose() * imu_angular_velocity_world
        };
        Point3 imu_linear_acceleration_sensor{
            // 假设传感器坐标系与载具坐标系重合 (C_{sv} = 1, r^{sv}_v = 0)
            // 那么在零误差的情况下
            // IMU 测得的比力等于
            // 朝向矩阵的转置 * (世界坐标系下的真实线加速度 - 世界坐标系下的重力加速度)
            imu_attitude.transpose()
                * (imu_linear_acceleration_world - gravity_world),
        };

        // 输出仿真 IMU 数据
        std::print(fout_imu0_data_csv,
                   // 时间戳
                   "{:020d}, "
                   // 角速度
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 加速度
                   "{:.18f}, {:.18f}, {:.18f}\n",
                   imu_timestamp_ns, //
                   imu_angular_velocity_sensor.x(),
                   imu_angular_velocity_sensor.y(),
                   imu_angular_velocity_sensor.z(), //
                   imu_linear_acceleration_sensor.x(),
                   imu_linear_acceleration_sensor.y(),
                   imu_linear_acceleration_sensor.z());
      }
#endif

#pragma endregion

#pragma region GENERATE_STEREO_IMAGE_DATA

      // 绘制相机图像
      {
        const cv::Scalar background_gray{128, 128, 128};
        cv::Mat cv_image_left(rig_.camera_left_.height_,
                              rig_.camera_left_.width_, CV_8UC3,
                              background_gray);
        cv::Mat cv_image_right(rig_.camera_right_.height_,
                               rig_.camera_right_.width_, CV_8UC3,
                               background_gray);

        std::print("\t绘制双目图像 ...\n");

        // 核心绘制逻辑收口
        mesh_plot_.Draw(cv_image_left, cv_image_right, frame);

#if (START_VISUALIZATION)
        if (mesh_plot_.Render(cv_image_left, cv_image_right, 1000))
        {
          break;
        }
#endif
#if (OUTPUT_AS_EUROC)
        // 时间戳作为图片文件名称
        const std::string image_file_name{
            std::format("{:020d}.png", timestamp_ns),
        };
        cv::imwrite(std::filesystem::absolute(path_cam0_data / image_file_name),
                    cv_image_left);
        cv::imwrite(std::filesystem::absolute(path_cam1_data / image_file_name),
                    cv_image_right);
#endif
      }

#pragma endregion

    } /* end for */
  }
};

int main()
{
  try
  {
    VisualSim<float>{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }
  return 0;
}
