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
#include <memory>
#include <numbers>
#include <optional>
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
  using OrientationMode = PathCircle<value_type>::OrientationMode;
  OrientationMode orientation_mode_{OrientationMode::LookAtCenter};
  PathManager<value_type> path_manager_{room_};
  value_type time_static_{2.0};
  // 相机的时间步长 (单位: 秒) (采用 0.05 秒作为时间步长可以让仿真相机的采样率保持为 20 赫兹)
  const value_type step_{static_cast<value_type>(0.05)};
  // 仿真 IMU 与仿真相机的采样率之比
  const int rate_ratio_{10};
  // 真值和 IMU 的时间步长 (单位: 秒)
  const value_type imu_step_{step_ / rate_ratio_};

  const std::filesystem::path path_mav0_{"mav0"};
  const std::filesystem::path path_cam0_{path_mav0_ / "cam0"};
  const std::filesystem::path path_cam0_data_{path_mav0_ / "cam0" / "data"};
  const std::filesystem::path path_cam1_{path_mav0_ / "cam1"};
  const std::filesystem::path path_cam1_data_{path_mav0_ / "cam1" / "data"};
  const std::filesystem::path path_groundtruth_{
      path_mav0_ / "state_groundtruth_estimate0"
  };
  const std::filesystem::path path_imu0_{path_mav0_ / "imu0"};

  using Point3     = Eigen::Vector<value_type, 3>;
  using Point2     = Eigen::Vector<value_type, 2>;
  using Attitude   = Eigen::Matrix<value_type, 3, 3>;
  using Quaternion = Eigen::Quaternion<value_type>;
  using Frame      = typename StereoRig<value_type>::Frame;

  VisualSim() : mesh_plot_{room_}
  {
    // 只修改双目相机的基线长度
    rig_.camera_right_.translation_ = {-0.1, 0.0, 0.0};

#pragma region CONSTRUCT_PATH

    // 构建轨迹，依次执行以下三个运动状态
    // 1. 静止状态
    // 2. 匀加速直线运动状态
    // 3. 匀速圆周运动状态

    const value_type radius{static_cast<value_type>(0.4)
                            * std::min<value_type>(room_.depth_, room_.width_)};
    const Point3 pos_all_start{
        room_.center_.x() + radius,
        room_.center_.y() - radius,
        room_.center_.z(),
    };
    const Point3 pos_circle_start{
        room_.center_.x() + radius,
        room_.center_.y(),
        room_.center_.z(),
    };

    // 角速率
    const value_type omega{0.5}; // rad s^-1
    // 匀速圆周运动的持续时间 = 运行两周所需时间 (单位: 秒)
    const value_type duration{std::round(4 * std::numbers::pi_v<value_type>
                                         / omega)};
    auto path_circle{std::make_unique<PathCircle<value_type>>(
        duration, omega, room_.center_, pos_circle_start, Point3::UnitZ(),
        OrientationMode::LookAtCenter
    )};
    // 初始朝向
    const Attitude att_init{
        std::get<Attitude>(path_circle->GetPose(static_cast<value_type>(0.0)))
    };
    // 线速度大小
    const value_type linear_velocity_norm{omega * radius};
    // 线加速度大小
    const value_type linear_acceleration_norm{
        static_cast<value_type>(0.5) * linear_velocity_norm
        * linear_velocity_norm / (pos_all_start - pos_circle_start).norm()
    };
    auto path_stationary{std::make_unique<PathStationary<value_type>>(
        time_static_, pos_all_start, att_init
    )};
    auto path_acceleration{std::make_unique<PathAcceleration<value_type>>(
        std::numeric_limits<value_type>::infinity(), pos_all_start,
        pos_circle_start, static_cast<value_type>(0.0),
        linear_acceleration_norm, att_init
    )};

#pragma endregion

#pragma region WRITE_EUROC_CONFIG

#if (OUTPUT_AS_EUROC)
    // 创建 mav0 目录
    std::error_code ec;
    std::filesystem::remove_all(path_mav0_, ec);
    if (std::filesystem::create_directories(path_mav0_, ec))
    {
      std::print(stderr, "[INFO] 工作目录创建成功: {}\n",
                 std::filesystem::absolute(path_mav0_).string());
    }
    else
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0 目录
    if (!std::filesystem::create_directories(path_cam0_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam0/data 目录
    if (!std::filesystem::create_directories(path_cam0_data_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1 目录
    if (!std::filesystem::create_directories(path_cam1_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/cam1/data 目录
    if (!std::filesystem::create_directories(path_cam1_data_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/state_groundtruth_estimate0 目录
    if (!std::filesystem::create_directories(path_groundtruth_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }
    // 创建 mav0/imu0 目录
    if (!std::filesystem::create_directories(path_imu0_, ec))
    {
      std::print(stderr, "[ERROR] 工作目录创建失败!\n");
      return;
    }

    const auto total_duration{path_stationary->GetDuration()
                              + path_acceleration->GetDuration()
                              + path_circle->GetDuration()};

    // 输出 mav0/README.md 说明文件
    std::ofstream fout_readme{path_mav0_ / "README.txt"};
    std::print(
        fout_readme,
        "Simulation Time: {} [s]  <!-- 仿真时长 (单位: 秒) -->\n"
        "Local Gravity: {:.2f} [m s^-2]  <!-- 重力加速度 -->\n"
        "Room:\n"
        "\tWidth:  {:.2f} [m]  <!-- 房间开间 -->\n"
        "\tDepth:  {:.2f} [m]  <!-- 房间进深 -->\n"
        "\tHeight: {:.2f} [m]  <!-- 房间层高 -->\n"
        "Room Center: [{:.2f}, {:.2f}, {:.2f}]"
        "  <!-- 房间中心位置 (单位: 米) -->\n"
        "Radius: {:.2f} [m]  <!-- 无人机运动轨迹的半径 (单位: 米) -->\n"
        "Movement Stage #1\n"
        "\tTime: {:.2f} [s]  "
        "<!-- 无人机起飞前处于静止状态的时间长度 (单位: 秒) -->\n"
        "\t<!-- 无人机起飞前所处位置、朝向的世界坐标 -->\n"
        "\tPosition: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\tAttitude: [[{:.2f}, {:.2f}, {:.2f}],\n"
        "\t           [{:.2f}, {:.2f}, {:.2f}],\n"
        "\t           [{:.2f}, {:.2f}, {:.2f}]]\n"
        "Movement Stage #2\n"
        "\tTime: {:.2f} [s]\n"
        "\t<!-- 匀加速直线运动的起点的世界坐标 -->\n"
        "\tPosition Start: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\t<!-- 匀加速直线运动的终点的世界坐标 -->\n"
        "\tPosition End: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\t<!-- 匀加速直线运动的初始线速度大小 -->\n"
        "\tLinear Velocity Start Norm: {:.2f} [m s^-1]\n"
        "\t<!-- 匀加速直线运动的线加速度大小 -->\n"
        "\tLinear Acceleration Norm: {:.2f} [m s^-2]\n"
        "Movement Stage #3\n"
        "\tTime: {:.2f} [s]\n"
        "\t<!-- 匀速圆周运动的角速度大小 -->\n"
        "\tAngular Velocity Norm: {:.2f} [rad s^-1]\n"
        "\t<!-- 匀速圆周运动的线速度大小 -->\n"
        "\tLinear Velocity Norm: {:.2f} [m s^-1]\n"
        "\t<!-- 匀速圆周运动的轨迹圆圆心位置的世界坐标 -->\n"
        "\tPosition Center: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\t<!-- 匀速圆周运动的轨迹起点的世界坐标 -->\n"
        "\tPosition Start: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\t<!-- 匀速圆周运动的轨迹圆所在平面的法向量 -->\n"
        "\tTrajectory Norm: [{:.2f}, {:.2f}, {:.2f}]\n"
        "\tMovement Paradigm: {}  <!-- 无人机的运动范式 -->\n",
        total_duration,                                          //
        gravity_world_norm_,                                     //
        room_.width_, room_.depth_, room_.height_,               //
        room_.center_.x(), room_.center_.y(), room_.center_.z(), //
        radius,                                                  //
        time_static_,                                            //
        pos_all_start.x(), pos_all_start.y(), pos_all_start.z(), //
        att_init(0, 0), att_init(0, 1), att_init(0, 2),          //
        att_init(1, 0), att_init(1, 1), att_init(1, 2),          //
        att_init(2, 0), att_init(2, 1), att_init(2, 2),          //
        path_acceleration->GetDuration(),                        //
        path_acceleration->GetPositionStart().x(),               //
        path_acceleration->GetPositionStart().y(),               //
        path_acceleration->GetPositionStart().z(),               //
        path_acceleration->GetPositionEnd().x(),                 //
        path_acceleration->GetPositionEnd().y(),                 //
        path_acceleration->GetPositionEnd().z(),                 //
        path_acceleration->GetLinearVelocityStartNorm(),         //
        path_acceleration->GetLinearAccelerationNorm(),          //
        path_circle->GetDuration(),                              //
        path_circle->GetOmega(),                                 //
        path_circle->GetOmega()
            * path_circle->GetNorm()
                  .cross(path_circle->GetPositionStart()
                         - path_circle->GetPositionCenter())
                  .norm(),
        path_circle->GetPositionCenter().x(),             //
        path_circle->GetPositionCenter().y(),             //
        path_circle->GetPositionCenter().z(),             //
        path_circle->GetPositionStart().x(),              //
        path_circle->GetPositionStart().y(),              //
        path_circle->GetPositionStart().z(),              //
        path_circle->GetNorm().x(),                       //
        path_circle->GetNorm().y(),                       //
        path_circle->GetNorm().z(),                       //
        enum_to_string(path_circle->GetOrientationMode()) //
    );

#endif

#pragma endregion

    path_manager_.AddPath(std::move(path_stationary));
    path_manager_.AddPath(std::move(path_acceleration));
    path_manager_.AddPath(std::move(path_circle));
  }

#pragma region WRITE_EUROC_CONFIG

  void WriteCameraConfig(const std::filesystem::path &path_cam,
                         const Camera<value_type> &camera) const
  {
    std::ofstream fout_cam{path_cam / "sensor.yaml"};
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
    std::ofstream fout_imu{path_imu / "sensor.yaml"};
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
    std::ofstream fout{path_groundtruth / "sensor.yaml"};
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
    auto opt_pose{path_manager_.GetPose(time)};
    if (opt_pose.has_value())
    {
      return opt_pose.value();
    }
    throw std::runtime_error{
        std::format("Fail to get current pose (time={:.4f})", time)
    };
  }

  void Start()
  {
#pragma region WRITE_EUROC_CONFIG

#if (OUTPUT_AS_EUROC)
    std::ofstream fout_cam0_data_csv{path_cam0_ / "data.csv"};
    std::ofstream fout_cam1_data_csv{path_cam1_ / "data.csv"};
    std::ofstream fout_imu0_data_csv{path_imu0_ / "data.csv"};
    std::ofstream fout_groundtruth_csv{path_groundtruth_ / "data.csv"};

    // 输出 mav0/cam0/sensor.yaml
    WriteCameraConfig(path_cam0_, rig_.camera_left_);
    // 输出 mav0/cam1/sensor.yaml
    WriteCameraConfig(path_cam1_, rig_.camera_right_);
    // 输出 mav0/state_groundtruth_estimate0/sensor.yaml
    WriteGroundTruthConfig(path_groundtruth_);
    // 输出 mav0/imu0/sensor.yaml
    WriteImuConfig(path_imu0_);

    // 输出 mav0/cam0/data.csv 表头
    std::print(fout_cam0_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/cam1/data.csv 表头
    std::print(fout_cam1_data_csv, "#timestamp [ns],filename\n");
    // 输出 mav0/state_groundtruth_estimate0/data.csv 表头
    std::print(
        fout_groundtruth_csv,
        "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
        "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
        "b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], "
        "b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2], "
        // EuRoC 数据集只提供以上信息 (即位置、朝向、线速度、陀螺仪零偏、加速度计零偏)
        // 以下是仿真数据集额外提供的信息 (即线加速度、角速度)
        "a_RS_R_x [m s^-2], a_RS_R_y [m s^-2], a_RS_R_z [m s^-2], "
        "w_RS_R_x [rad s^-1], w_RS_R_y [rad s^-1], w_RS_R_z [rad s^-1]\n"
    );
    // 输出 mav0/imu0/data.csv 表头
    std::print(fout_imu0_data_csv,
               "#timestamp [ns],"
               "w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
               "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n");

    std::ofstream fout_landmarks{path_mav0_ / "landmarks.csv"};
    std::ofstream fout_cam0_pixels{path_cam0_ / "pixels.csv"};
    std::ofstream fout_cam1_pixels{path_cam1_ / "pixels.csv"};

    std::print(fout_landmarks, "#timestamp [ns],3-D points\n");
    std::print(fout_cam0_pixels, "#timestamp [ns],2-D points\n");
    std::print(fout_cam1_pixels, "#timestamp [ns],2-D points\n");

#endif

#pragma endregion

    const Point3 gravity_world{-gravity_world_norm_ * Point3::UnitZ()};

    for (value_type time = 0.0;; time += step_)
    {
      std::print(stderr, "[INFO] 时间 = ({:.3f}).\n", time);

#if (OUTPUT_AS_EUROC)
      const auto timestamp_ns{static_cast<std::int64_t>(time * 1e9)};
      std::print(fout_cam0_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
      std::print(fout_cam1_data_csv, "{0:020d},{0:020d}.png\n", timestamp_ns);
#endif

      auto opt_frame{path_manager_.GetImage(rig_, time)};
      if (!opt_frame.has_value())
      {
        break;
      }
      const Frame frame{opt_frame.value()};

      // 打印路标点
      std::print(fout_landmarks, "{0:020d},[", timestamp_ns);
      for (bool first_loop{true}; auto index_landmark : std::get<0>(frame))
      {
        auto object_point{
            room_.object_matrix_.col(static_cast<Eigen::Index>(index_landmark))
        };
        if (first_loop)
        {
          std::print(fout_landmarks, "({:.18f} {:.18f} {:.18f})",
                     object_point.x(), object_point.y(), object_point.z());
          first_loop = false;
        }
        else
        {
          std::print(fout_landmarks, ";({:.18f} {:.18f} {:.18f})",
                     object_point.x(), object_point.y(), object_point.z());
        }
      }
      std::print(fout_landmarks, "]\n");

      // 打印左目像素点
      std::print(fout_cam0_pixels, "{0:020d},[", timestamp_ns);
      for (bool first_loop{true}; auto pixel_point : std::get<1>(frame))
      {
        if (first_loop)
        {
          std::print(fout_cam0_pixels, "({:.18f} {:.18f})", pixel_point.x(),
                     pixel_point.y());
          first_loop = false;
        }
        else
        {
          std::print(fout_cam0_pixels, ";({:.18f} {:.18f})", pixel_point.x(),
                     pixel_point.y());
        }
      }
      std::print(fout_cam0_pixels, "]\n");

      // 打印右目像素点
      std::print(fout_cam1_pixels, "{0:020d},[", timestamp_ns);
      for (bool first_loop{true}; auto pixel_point : std::get<2>(frame))
      {
        if (first_loop)
        {
          std::print(fout_cam1_pixels, "({:.18f} {:.18f})", pixel_point.x(),
                     pixel_point.y());
          first_loop = false;
        }
        else
        {
          std::print(fout_cam1_pixels, ";({:.18f} {:.18f})", pixel_point.x(),
                     pixel_point.y());
        }
      }
      std::print(fout_cam1_pixels, "]\n");

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
        const auto opt_kinematics{path_manager_.GetKinematics(imu_time)};
        if (!opt_kinematics.has_value())
        {
          break;
        }
        std::tie(imu_linear_velocity_world, imu_linear_acceleration_world,
                 imu_angular_velocity_world) = opt_kinematics.value();

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
                   "{:.18f}, {:.18f}, {:.18f}, "
                   // 陀螺仪零偏
                   "0.0000, 0.0000, 0.0000, "
                   // 加速度计零偏
                   "0.0000, 0.0000, 0.0000, "
                   // 线加速度
                   "{:.18f}, {:.18f}, {:.18f}, "
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

        // std::print("\t绘制双目图像 ...\n");

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
        cv::imwrite(std::filesystem::absolute(path_cam0_data_
                                              / image_file_name),
                    cv_image_left);
        cv::imwrite(std::filesystem::absolute(path_cam1_data_
                                              / image_file_name),
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
    VisualSim<double>{}.Start();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }
  return 0;
}
