import std;

using namespace std::chrono_literals;

#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include <boost/numeric/odeint.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/core/eigen.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>

import FastVIO:AbstractLoader;
import FastVIO:ErrorStateKalmanFilter;
import FastVIO:Interpolation;
import FastVIO:Sensor;
import FastVIO:ZUPT;
import FastVIO:SensorYaml;
import FastVIO:DatumFast;
import FastVIO:DatumImu;
import FastVIO:DatumTruth;

namespace FastVIO
{

/**
 * @brief 统一的评估器配置与输入数据结构体。
 *        整合了传感器物理外参、置信度常数、初始配置以及估算所需的所有时序数据集，
 *        以便通过单一结构体实例对各估算器派生类进行统一构造。
 */
struct EstimatorConfig
{
  // 物理常数与初始化配置

  // 世界坐标系下的重力加速度大小 [m/s^2]
  double gravity_world_norm_{9.81};
  // 是否采用初始时刻真值作为状态初值
  bool use_true_init_pose_{false};

  // 算法置信度参数 (用于卡尔曼滤波观测噪声协方差)

  double confidence_angular_displacement_{1e-4};
  double confidence_normalized_translation_{1e-4};

  // 传感器外参及物理噪声等配置文件

  SensorYaml sensor_config_cam0_{};
  SensorYaml sensor_config_imu0_{};
  SensorYaml sensor_config_truth_{};

  using ProjectionMatrix = Eigen::Matrix<double, 3, 4>;
  // 经过立体矫正后，左目相机的 3x4 投影矩阵
  ProjectionMatrix proj_left_{
      {1.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0},
  };
  // 经过立体矫正后，右目相机的 3x4 投影矩阵
  ProjectionMatrix proj_right_{
      {1.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0},
  };

  // 评估所需的传感器时序输入数据集

  std::vector<DatumFast> data_fast_{};
  std::vector<DatumImu> data_imu_{};
  std::vector<DatumTruth> data_truth_{};

  // 系统文件输入输出配置

  // 估算结果 CSV 文件的输出保存目录路径
  std::string output_dir_{"."};
};

/**
 * @brief 轨迹估计抽象基类，规范了所有物理状态估算器的接口。
 */
class AbstractEstimator
{
public:
  AbstractEstimator(const EstimatorConfig &config) : config_{config} {}

  virtual ~AbstractEstimator() = default;

  /**
   * @brief 执行轨迹估算，并输出对应的运动轨迹数据。
   */
  virtual void Estimate() = 0;

  /**
   * @brief 获取当前评估器的名称（用作导出 CSV 的标识名）。
   * @return 估算器类名的字符串。
   */
  virtual std::string GetName() const = 0;

protected:
  const EstimatorConfig config_;

private:
  std::ofstream file_;

  void EnsureFileOpen()
  {
    if (file_.is_open())
    {
      return;
    }
    file_.open(std::filesystem::path{config_.output_dir_}
                   / std::format("{}.csv", GetName()),
               std::ios::trunc);
  }

protected:
  /**
   * @brief 初始化导出的 CSV 文件并写入统一的格式头部。
   */
  void InitializeCsv()
  {
    EnsureFileOpen();
    if (file_.is_open())
    {
      std::print(file_, "#timestamp [ns],"
                        "p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
                        "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n");
    }
  }

  /**
   * @brief 将当前状态数据追加入对应的导出 CSV 记录中。
   * @param timestamp 纳米级整型时间戳。
   * @param position 估计出的 3 维位置向量。
   * @param attitude 估计出的四元数旋转状态。
   */
  void AppendToCsv(std::int64_t timestamp, const Eigen::Vector3d &position,
                   const Eigen::Quaterniond &attitude)
  {
    if (file_.is_open())
    {
      std::print(file_,
                 // 时间戳
                 "{:020d},"
                 // 位置
                 "{:.18f},{:.18f},{:.18f},"
                 // 朝向
                 "{:.18f},{:.18f},{:.18f},{:.18f}\n",
                 timestamp, position.x(), position.y(), position.z(),
                 attitude.w(), attitude.x(), attitude.y(), attitude.z());
    }
  }
};

/**
 * @brief 单目视觉单纯采用角位移与单位化平移估计轨迹的算法类。
 */
class FastEstimator : public AbstractEstimator
{
public:
  FastEstimator(const EstimatorConfig &config) : AbstractEstimator(config) {}

  /**
   * @brief 通过单目估计相邻相机位姿实现位置与旋转递推。
   */
  void Estimate() override
  {
    InitializeCsv();

    Eigen::Vector3d estimated_position{Eigen::Vector3d::Zero()};
    Sophus::SO3d estimated_attitude{};

    if (config_.use_true_init_pose_ && !config_.data_truth_.empty())
    {
      estimated_position = config_.data_truth_[0].position_;
      estimated_attitude = Sophus::SO3d(config_.data_truth_[0].attitude_);
    }

    for (std::size_t i = 0; i + 1 < config_.data_fast_.size(); ++i)
    {
      const DatumFast &datum_fast{config_.data_fast_[i]};
      Sophus::SO3d delta_rotation{
          Sophus::SO3d::exp(datum_fast.angular_displacement_)
      };

      Eigen::Vector3d delta_position{datum_fast.normalized_translation_};

      // 因为数据集 path_estimation_csv 提供的旋转向量、平移向量是在相机坐标系下的表示
      // 所以应该使用以下状态更新方程
      estimated_position
          = estimated_position + estimated_attitude * delta_position;
      estimated_attitude = estimated_attitude * delta_rotation;

      AppendToCsv(datum_fast.timestamp_, estimated_position,
                  estimated_attitude.unit_quaternion());
    }
  }

  std::string GetName() const override
  {
    return "FastEstimator";
  }
};

/**
 * @brief 采用 Euler 梯形积分法实现 IMU 轨迹递推。
 */
class EulerEstimator : public AbstractEstimator
{
public:
  EulerEstimator(const EstimatorConfig &config) : AbstractEstimator(config) {}

  /**
   * @brief 一阶中值积分法估算状态转移。
   */
  void Estimate() override
  {
    InitializeCsv();

    // 世界坐标系下的重力加速度
    const Eigen::Vector3d gravity_world{-config_.gravity_world_norm_
                                        * Eigen::Vector3d::UnitZ()};

    Eigen::Vector3d estimated_position_imu{Eigen::Vector3d::Zero()};
    Sophus::SO3d estimated_attitude_imu{};
    Eigen::Vector3d estimated_linear_velocity_imu{Eigen::Vector3d::Zero()};

    if (config_.use_true_init_pose_ && !config_.data_truth_.empty())
    {
      estimated_position_imu = config_.data_truth_[0].position_;
      estimated_attitude_imu = Sophus::SO3d(config_.data_truth_[0].attitude_);
      estimated_linear_velocity_imu = config_.data_truth_[0].velocity_;
    }
    else
    {
      // 引入“零速更新”机制，检测起飞时刻
      ZUPT<double> zupt{};
      bool is_orientation_estimated{false};

      DatumImu datum_first;
      DatumImu datum_last;
      for (bool first_loop{true}; const DatumImu &datum_imu : config_.data_imu_)
      {
        if (first_loop)
        {
          datum_first = datum_imu;
          first_loop  = false;
          continue;
        }
        if (zupt.IsFull() && !is_orientation_estimated)
        { // 当样本足够多时，如果尚未预测过初始朝向，就立即进行预测
          estimated_attitude_imu   = Sophus::SO3d(zupt.EstimateOrientation());
          is_orientation_estimated = true;
        }
        if (!zupt.Update(datum_imu.linear_acceleration_,
                         datum_imu.angular_velocity_))
        {
          datum_last = datum_imu;
          break;
        }
      }
      if (!is_orientation_estimated)
      { // 如果尚未预测过初始朝向，就立即进行预测
        estimated_attitude_imu = Sophus::SO3d(zupt.EstimateOrientation());
      }
    }

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_imu : config_.data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_imu;
        first_loop = false;
        AppendToCsv(datum_imu.timestamp_, estimated_position_imu,
                    estimated_attitude_imu.unit_quaternion());
        continue;
      }

      // 时间步长
      const double dt{
          1e-9f
              * static_cast<double>(datum_imu.timestamp_
                                    - datum_prev.timestamp_),
      };

      // 载具参考系下的角速度 = 传感器参考系下的角速度
      // 载具参考系下前一帧角速度
      Eigen::Vector3d previous_angular_velocity_in_body_frame{
          datum_prev.angular_velocity_,
      };
      // 载具参考系下后一帧角速度
      Eigen::Vector3d current_angular_velocity_in_body_frame{
          datum_imu.angular_velocity_,
      };
      // 载具参考系下两帧角速度的平均值
      Eigen::Vector3d median_angular_velocity_in_body_frame{
          0.5
              * (previous_angular_velocity_in_body_frame
                 + current_angular_velocity_in_body_frame),
      };
      // 朝向变化量
      Sophus::SO3d delta_attitude{
          Sophus::SO3d::exp(
              median_angular_velocity_in_body_frame * dt
              + (dt * dt / 12.0)
                    * previous_angular_velocity_in_body_frame.cross(
                        current_angular_velocity_in_body_frame
                    )
          ),
      };
      // 新的朝向
      Sophus::SO3d estimated_new_attitude_imu{
          estimated_attitude_imu * delta_attitude,
      };

      // 惯性参考系下的线加速度
      Eigen::Vector3d linear_acceleration_in_world_frame{
          estimated_attitude_imu * datum_prev.linear_acceleration_
              + gravity_world,
      };
      // 线速度变化量
      Eigen::Vector3d delta_velocity{
          linear_acceleration_in_world_frame * dt,
      };
      // 位置变化量
      Eigen::Vector3d delta_position{
          (estimated_linear_velocity_imu + 0.5 * delta_velocity) * dt,
      };

      // 更新位置
      estimated_position_imu += delta_position;
      // 更新线速度
      estimated_linear_velocity_imu += delta_velocity;
      // 更新朝向
      estimated_attitude_imu = estimated_new_attitude_imu;

      AppendToCsv(datum_imu.timestamp_, estimated_position_imu,
                  estimated_attitude_imu.unit_quaternion());
      datum_prev = datum_imu;
    }
  }

  std::string GetName() const override
  {
    return "EulerEstimator";
  }
};

/**
 * @brief IMU 预积分估计器。
 */
class Preintegrator : public AbstractEstimator
{
public:
  Preintegrator(const EstimatorConfig &config) : AbstractEstimator(config) {}

  /**
   * @brief 通过固定某一时刻相对累积预积分完成位姿解算。
   */
  void Estimate() override
  {
    InitializeCsv();

    const Eigen::Vector3d gravity_world{-config_.gravity_world_norm_
                                        * Eigen::Vector3d::UnitZ()};

    // 预积分的“参考基准”（起始状态 P0, V0, R0）
    Eigen::Vector3d P0{Eigen::Vector3d::Zero()};
    Eigen::Vector3d V0{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond R0{Eigen::Quaterniond::Identity()};

    if (config_.use_true_init_pose_ && !config_.data_truth_.empty())
    {
      P0 = config_.data_truth_[0].position_;
      R0 = config_.data_truth_[0].attitude_;
      V0 = config_.data_truth_[0].velocity_;
    }

    // 预积分的相对变化量（从参考基准时刻到当前时刻）
    Eigen::Quaterniond delta_R{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d delta_p{Eigen::Vector3d::Zero()};
    Eigen::Vector3d delta_v{Eigen::Vector3d::Zero()};
    double delta_t{0.0};
    double t_prev{0.0};

    for (bool first_loop{true}; const DatumImu &datum_imu : config_.data_imu_)
    {
      double t_samp{1e-9f * static_cast<double>(datum_imu.timestamp_)};
      if (first_loop)
      {
        first_loop = false;
        t_prev     = t_samp;
        continue;
      }

      // 时间步长
      const double dt{t_samp - t_prev};
      auto drotvec{dt * datum_imu.angular_velocity_};
      double angle = drotvec.norm();

      Eigen::Quaterniond dR;
      if (angle > 1e-6)
      {
        dR = Eigen::Quaterniond{Eigen::AngleAxisd{angle, drotvec / angle}};
      }
      else
      {
        dR = Eigen::Quaterniond{1.0, 0.5 * drotvec.x(), 0.5 * drotvec.y(),
                                0.5 * drotvec.z()};
        dR.normalize();
      }

      auto dv{dt * datum_imu.linear_acceleration_};
      auto dp{0.5 * dt * dv};

      // 累积相对预积分量
      delta_t += dt;
      delta_p += delta_v * dt + delta_R * dp;
      delta_v += delta_R * dv;
      delta_R = delta_R * dR;
      t_prev  = t_samp;

      // 利用固定的初始态 (P0, V0, R0) 以及 预积分量 计算当前时刻的全局位姿。
      // 绝不要把计算出的 current_xxx 又反向赋值回去进行递归！
      Eigen::Vector3d current_position{P0 + V0 * delta_t
                                       + 0.5 * delta_t * delta_t * gravity_world
                                       + R0 * delta_p};
      Eigen::Quaterniond current_attitude{R0 * delta_R};

      AppendToCsv(datum_imu.timestamp_, current_position, current_attitude);
    }
  }

  std::string GetName() const override
  {
    return "Preintegrator";
  }
};

/**
 * @brief 四阶龙格库塔数值分析方法(RK4)处理常微分方程进行状态估计。
 */
class RK4Estimator : public AbstractEstimator
{
private:
  // 初始状态
  ImuState<double> state_;

  // 积分器
  boost::numeric::odeint::runge_kutta4<ImuState<double>, double,
                                       ImuDerivative<double>>
      rk4_;

  struct ImuKinematicsODE
  {
    const DatumImu &datum_prev_;
    const DatumImu &datum_next_;
    const Eigen::Vector3d &gravity_world_;

    void operator()(const ImuState<double> &x, ImuDerivative<double> &dxdt,
                    const double t) const
    {
      double alpha{
          (datum_next_.timestamp_ > datum_prev_.timestamp_)
              ? std::clamp(static_cast<double>((t - datum_prev_.timestamp_)
                                               / (datum_next_.timestamp_
                                                  - datum_prev_.timestamp_)),
                           0.0, 1.0)
              : 0.0,
      };
      const Eigen::Vector3d ang_vel_sensor{
          datum_prev_.angular_velocity_
              + (datum_next_.angular_velocity_ - datum_prev_.angular_velocity_)
                    * alpha,
      };
      Eigen::Quaterniond att_world{x.GetAttitude()};
      Eigen::Vector3d lin_vec_world{x.GetVelocity()};
      Eigen::Vector3d lin_acc_sensor{
          datum_prev_.linear_acceleration_
              + (datum_next_.linear_acceleration_
                 - datum_prev_.linear_acceleration_)
                    * alpha,
      };
      Eigen::Vector3d lin_acc_world{
          att_world * lin_acc_sensor + gravity_world_,
      };
      Eigen::Quaterniond half_rotation{
          0.0,
          0.5 * ang_vel_sensor.x(),
          0.5 * ang_vel_sensor.y(),
          0.5 * ang_vel_sensor.z(),
      };
      Eigen::Quaterniond att_derivative_world{att_world * half_rotation};

      // 位置导数 = 速度
      dxdt.SetVelocity(lin_vec_world);
      // 速度导数 = 加速度
      dxdt.SetAcceleration(lin_acc_world);
      // 朝向导数 = 0.5 * 朝向 ** 角速度
      dxdt.SetAttitudeDerivative(att_derivative_world);
    }
  };

public:
  RK4Estimator(const EstimatorConfig &config) : AbstractEstimator(config) {}

  /**
   * @brief 采用 Boost ODE 求解器实现高精度的姿态和位置推算。
   */
  void Estimate() override
  {
    if (config_.data_imu_.empty())
    {
      return;
    }
    InitializeCsv();

    const Eigen::Vector3d gravity_world{-config_.gravity_world_norm_
                                        * Eigen::Vector3d::UnitZ()};
    double ode_time{static_cast<double>(1e-9f
                                        * config_.data_imu_[0].timestamp_)};

    if (config_.use_true_init_pose_ && !config_.data_truth_.empty())
    {
      state_.SetPosition(config_.data_truth_[0].position_);
      state_.SetAttitude(config_.data_truth_[0].attitude_);
      state_.SetVelocity(config_.data_truth_[0].velocity_);
    }
    else
    {
      ZUPT<double> zupt{};
      bool is_orientation_estimated{false};
      Eigen::Quaterniond estimated_attitude_rk{Eigen::Quaterniond::Identity()};

      DatumImu datum_first;
      DatumImu datum_last;
      for (bool first_loop{true}; const DatumImu &datum_rk : config_.data_imu_)
      {
        if (first_loop)
        {
          datum_first = datum_rk;
          first_loop  = false;
          continue;
        }
        if (zupt.IsFull() && !is_orientation_estimated)
        { // 当样本足够多时，如果尚未预测过初始朝向，就立即进行预测
          estimated_attitude_rk    = zupt.EstimateOrientation();
          is_orientation_estimated = true;
        }
        if (!zupt.Update(datum_rk.linear_acceleration_,
                         datum_rk.angular_velocity_))
        {
          datum_last = datum_rk;
          break;
        }
      }
      if (!is_orientation_estimated)
      { // 如果尚未预测过初始朝向，就立即进行预测
        estimated_attitude_rk = zupt.EstimateOrientation();
      }
      state_.SetAttitude(estimated_attitude_rk);
    }

    DatumImu datum_prev;
    for (bool first_loop{true}; const DatumImu &datum_rk : config_.data_imu_)
    {
      if (first_loop)
      {
        datum_prev = datum_rk;
        first_loop = false;
        AppendToCsv(datum_rk.timestamp_, state_.GetPosition(),
                    state_.GetAttitude());
        continue;
      }

      const double dt{
          1e-9f
              * static_cast<double>(datum_rk.timestamp_
                                    - datum_prev.timestamp_),
      };

      ImuKinematicsODE ode{datum_prev, datum_rk, gravity_world};
      rk4_.do_step(ode, state_, ode_time, dt);
      ode_time += dt;
      state_.NormalizeAttitude();

      AppendToCsv(datum_rk.timestamp_, state_.GetPosition(),
                  state_.GetAttitude());
      datum_prev = datum_rk;
    }
  }

  std::string GetName() const override
  {
    return "RK4Estimator";
  }
};

/**
 * @brief 融合估计器，利用 ESKF 松耦合融合单目估计角位移/平移和惯导信息。
 */
class FuseEstimator : public AbstractEstimator
{
private:
  using ESKF = Filter::ErrorStateKalmanFilter<double>;

public:
  FuseEstimator(const EstimatorConfig &config) : AbstractEstimator(config) {}

  /**
   * @brief 采用时序异步对齐方式，进行 ESKF 标称状态前推以及视觉观测融合计算。
   */
  void Estimate() override
  {
    InitializeCsv();

    struct TimelineEvent
    {
      // 纳秒级全局统一时间戳
      std::int64_t timestamp;
      // 标识当前事件是否属于惯性测量单元
      bool is_imu;
      // 记录当前帧在各自容器(data_fast_或data_imu_)中的原始索引值
      std::size_t index;
    };

    std::vector<TimelineEvent> events;
    events.reserve(config_.data_fast_.size() + config_.data_imu_.size());

    // 填充单目视觉特征帧信息至时间轴中
    for (std::size_t i = 0; i < config_.data_fast_.size(); ++i)
    {
      events.push_back({config_.data_fast_[i].timestamp_, false, i});
    }
    // 填充高频惯性特征帧信息至时间轴中
    for (std::size_t i = 0; i < config_.data_imu_.size(); ++i)
    {
      events.push_back({config_.data_imu_[i].timestamp_, true, i});
    }

    // 针对混合时间轴事件根据时间戳升序排序，若时间戳相同则让 IMU 优先处理
    std::ranges::sort(events, std::less<>{}, [](const TimelineEvent &e)
                      { return std::make_tuple(e.timestamp, !e.is_imu); });

    typename ESKF::Config eskf_config;
    eskf_config.imu_rate_   = config_.sensor_config_imu0_.rate_hz_;
    eskf_config.proj_left_  = config_.proj_left_;
    eskf_config.proj_right_ = config_.proj_right_;
    ESKF eskf{eskf_config};
    // 初始化 ESKF 的标称状态
    typename ESKF::NominalStateVariable init_state;

    if (config_.use_true_init_pose_ && !config_.data_truth_.empty())
    {
      init_state.position_           = config_.data_truth_[0].position_;
      init_state.linear_velocity_    = config_.data_truth_[0].velocity_;
      init_state.attitude_           = config_.data_truth_[0].attitude_;
      init_state.accelerometer_bias_ = config_.data_truth_[0].bias_accel_;
      init_state.gyroscope_bias_     = config_.data_truth_[0].bias_gyro_;
      init_state.gravity_
          = -config_.gravity_world_norm_ * Eigen::Vector3d::UnitZ();
    }
    else
    {
      // TODO 尚未编码专用的初始姿态解算机制
      if (!config_.data_truth_.empty())
      {
        init_state.attitude_ = config_.data_truth_[0].attitude_;
      }
    }
    eskf.SetNominalState(init_state);

    // 将 YAML 中读取的传感器物理特征噪声传入 ESKF 进行精确的过程协方差计算
    eskf.SetGyroscopeNoiseDensity(
        config_.sensor_config_imu0_.gyroscope_noise_density_
    );
    eskf.SetGyroscopeRandomWalk(
        config_.sensor_config_imu0_.gyroscope_random_walk_
    );
    eskf.SetAccelerometerNoiseDensity(
        config_.sensor_config_imu0_.accelerometer_noise_density_
    );
    eskf.SetAccelerometerRandomWalk(
        config_.sensor_config_imu0_.accelerometer_random_walk_
    );

    eskf.confidence_angular_displacement_
        = config_.confidence_angular_displacement_;
    eskf.confidence_normalized_translation_
        = config_.confidence_normalized_translation_;

    // 顺序迭代离线混合时间轴上的所有传感器事件
    for (const auto &event : events)
    {
      if (event.is_imu)
      {
        // 传递高频 IMU 采样数据，执行 ESKF 标称状态前推以及误差状态协方差的时间传播
        const auto &datum_imu{config_.data_imu_[event.index]};
        eskf.ImuUpdate(&datum_imu);
      }
      else
      {
        // 加载当前帧低频单目视觉观测信息并调用 ESKF 的观测融合与后验误差校正
        const auto &datum_fast{config_.data_fast_[event.index]};
        eskf.MonocularUpdate(&datum_fast);

        // 获取融合后的最新名义状态
        auto state{eskf.GetNominalState()};
        AppendToCsv(datum_fast.timestamp_, state.GetPosition(),
                    state.GetAttitude());
      }
    }
  }

  std::string GetName() const override
  {
    return "FuseEstimator";
  }
};

/**
 * @brief 数据加载管理类，负责配置参数、解析数据集并将数据分发至特定评估器。
 */
class TrajectoryFactory : public rclcpp::Node
{
private:
  EstimatorConfig estimator_config_;
  std::vector<std::unique_ptr<AbstractEstimator>> estimators_;

public:
  /**
   * @brief 构造函数，声明 ROS 参数并初始化读取配置文件和数据集。
   */
  TrajectoryFactory() : Node("TrajectoryFactory")
  {
    this->declare_parameter("use_true_init_pose", false);
    estimator_config_.use_true_init_pose_
        = this->get_parameter("use_true_init_pose").as_bool();

    this->declare_parameter("path_estimation_csv", "estimated_motion.csv");
    const std::string path_estimation_csv{
        this->get_parameter("path_estimation_csv").as_string(),
    };

    this->declare_parameter("path_cam0_yaml", "");
    const std::string path_cam0_yaml{
        this->get_parameter("path_cam0_yaml").as_string(),
    };

    this->declare_parameter("path_imu_csv", "");
    const std::string path_imu_csv{
        this->get_parameter("path_imu_csv").as_string(),
    };

    this->declare_parameter("path_imu_yaml", "");
    const std::string path_imu_yaml{
        this->get_parameter("path_imu_yaml").as_string(),
    };

    this->declare_parameter("path_truth_csv", "");
    const std::string path_truth_csv{
        this->get_parameter("path_truth_csv").as_string(),
    };

    this->declare_parameter("path_truth_yaml", "");
    const std::string path_truth_yaml{
        this->get_parameter("path_truth_yaml").as_string(),
    };

    this->declare_parameter("confidence_angular_displacement", 1e-4);
    estimator_config_.confidence_angular_displacement_
        = this->get_parameter("confidence_angular_displacement").as_double();

    this->declare_parameter("confidence_normalized_translation", 1e-4);
    estimator_config_.confidence_normalized_translation_
        = this->get_parameter("confidence_normalized_translation").as_double();

    this->declare_parameter("output_dir", ".");
    estimator_config_.output_dir_
        = this->get_parameter("output_dir").as_string();

    this->declare_parameter("estimators", std::vector<std::string>{});
    const std::vector<std::string> active_estimators{
        this->get_parameter("estimators").as_string_array(),
    };

    this->declare_parameter<std::vector<double>>(
        "proj_left", std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                         0.0, 0.0, 1.0, 0.0}
    );
    auto proj_left_vec = this->get_parameter("proj_left").as_double_array();
    if (proj_left_vec.size() == 12)
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          estimator_config_.proj_left_(i, j) = proj_left_vec[i * 4 + j];
        }
      }
    }
    else
    {
      RCLCPP_WARN(
          this->get_logger(),
          "proj_left vector size is not 12, using default identity matrix."
      );
    }

    this->declare_parameter<std::vector<double>>(
        "proj_right", std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0, 0.0}
    );
    auto proj_right_vec = this->get_parameter("proj_right").as_double_array();
    if (proj_right_vec.size() == 12)
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          estimator_config_.proj_right_(i, j) = proj_right_vec[i * 4 + j];
        }
      }
    }
    else
    {
      RCLCPP_WARN(
          this->get_logger(),
          "proj_right vector size is not 12, using default identity matrix."
      );
    }

    if (path_estimation_csv.empty() || path_cam0_yaml.empty()
        || path_imu_csv.empty() || path_imu_yaml.empty()
        || path_truth_csv.empty() || path_truth_yaml.empty())
    {
      throw std::runtime_error{"Required configuration paths cannot be empty."};
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path_estimation_csv, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_estimation_csv
      )};
    }
    if (!std::filesystem::is_regular_file(path_cam0_yaml, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_cam0_yaml
      )};
    }
    if (!std::filesystem::is_regular_file(path_imu_csv, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_imu_csv
      )};
    }
    if (!std::filesystem::is_regular_file(path_imu_yaml, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_imu_yaml
      )};
    }
    if (!std::filesystem::is_regular_file(path_truth_csv, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_truth_csv
      )};
    }
    if (!std::filesystem::is_regular_file(path_truth_yaml, ec))
    {
      throw std::runtime_error{std::format(
          "Required configuration path '{}' not found.", path_truth_yaml
      )};
    }

    auto opt_sensor_config_cam0{SensorYaml::ReadSensorYaml(path_cam0_yaml)};
    if (opt_sensor_config_cam0.has_value())
    {
      estimator_config_.sensor_config_cam0_
          = std::move(opt_sensor_config_cam0.value());
      std::print(stderr,
                 "[INFO] T_BS_cam0 =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 estimator_config_.sensor_config_cam0_.transform_matrix_(0, 0),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(0, 1),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(0, 2),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(0, 3),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(1, 0),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(1, 1),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(1, 2),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(1, 3),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(2, 0),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(2, 1),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(2, 2),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(2, 3),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(3, 0),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(3, 1),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(3, 2),
                 estimator_config_.sensor_config_cam0_.transform_matrix_(3, 3));
    }
    else
    {
      throw std::runtime_error{std::format(
          "Failed to parse camera config yaml '{}'.", path_cam0_yaml
      )};
    }

    auto opt_sensor_config_imu0{SensorYaml::ReadSensorYaml(path_imu_yaml)};
    if (opt_sensor_config_imu0.has_value())
    {
      estimator_config_.sensor_config_imu0_
          = std::move(opt_sensor_config_imu0.value());
      std::print(stderr,
                 "[INFO] T_BS_imu0 =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 estimator_config_.sensor_config_imu0_.transform_matrix_(0, 0),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(0, 1),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(0, 2),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(0, 3),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(1, 0),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(1, 1),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(1, 2),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(1, 3),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(2, 0),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(2, 1),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(2, 2),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(2, 3),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(3, 0),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(3, 1),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(3, 2),
                 estimator_config_.sensor_config_imu0_.transform_matrix_(3, 3));
    }
    else
    {
      throw std::runtime_error{
          std::format("Failed to parse IMU config yaml '{}'.", path_imu_yaml)
      };
    }

    auto opt_sensor_config_truth{SensorYaml::ReadSensorYaml(path_truth_yaml)};
    if (opt_sensor_config_truth.has_value())
    {
      estimator_config_.sensor_config_truth_
          = std::move(opt_sensor_config_truth.value());
      std::print(stderr,
                 "[INFO] T_BS_truth =\n"
                 "\t[[{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}],\n"
                 "\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]]\n",
                 estimator_config_.sensor_config_truth_.transform_matrix_(0, 0),
                 estimator_config_.sensor_config_truth_.transform_matrix_(0, 1),
                 estimator_config_.sensor_config_truth_.transform_matrix_(0, 2),
                 estimator_config_.sensor_config_truth_.transform_matrix_(0, 3),
                 estimator_config_.sensor_config_truth_.transform_matrix_(1, 0),
                 estimator_config_.sensor_config_truth_.transform_matrix_(1, 1),
                 estimator_config_.sensor_config_truth_.transform_matrix_(1, 2),
                 estimator_config_.sensor_config_truth_.transform_matrix_(1, 3),
                 estimator_config_.sensor_config_truth_.transform_matrix_(2, 0),
                 estimator_config_.sensor_config_truth_.transform_matrix_(2, 1),
                 estimator_config_.sensor_config_truth_.transform_matrix_(2, 2),
                 estimator_config_.sensor_config_truth_.transform_matrix_(2, 3),
                 estimator_config_.sensor_config_truth_.transform_matrix_(3, 0),
                 estimator_config_.sensor_config_truth_.transform_matrix_(3, 1),
                 estimator_config_.sensor_config_truth_.transform_matrix_(3, 2),
                 estimator_config_.sensor_config_truth_.transform_matrix_(3,
                                                                          3));
    }
    else
    {
      throw std::runtime_error{std::format(
          "Failed to parse groundtruth config yaml '{}'.", path_truth_yaml
      )};
    }

    estimator_config_.data_fast_
        = DatumFast::Load(path_estimation_csv,
                          Sophus::SO3d{
                              estimator_config_.sensor_config_cam0_
                                  .transform_matrix_.template block<3, 3>(0, 0),
                          });
    estimator_config_.data_imu_
        = DatumImu::Load(path_imu_csv,
                         Sophus::SO3d{
                             estimator_config_.sensor_config_imu0_
                                 .transform_matrix_.template block<3, 3>(0, 0),
                         });
    estimator_config_.data_truth_ = DatumTruth::Load(
        path_truth_csv, Sophus::SO3d{
                            estimator_config_.sensor_config_truth_
                                .transform_matrix_.template block<3, 3>(0, 0),
                        }
    );

    // 根据 Launch 动态生成需要的具体评估器。
    for (const auto &name : active_estimators)
    {
      if (name == "FastEstimator")
      {
        estimators_.push_back(
            std::make_unique<FastEstimator>(estimator_config_)
        );
      }
      else if (name == "EulerEstimator")
      {
        estimators_.push_back(
            std::make_unique<EulerEstimator>(estimator_config_)
        );
      }
      else if (name == "RK4Estimator")
      {
        estimators_.push_back(
            std::make_unique<RK4Estimator>(estimator_config_)
        );
      }
      else if (name == "Preintegrator")
      {
        estimators_.push_back(
            std::make_unique<Preintegrator>(estimator_config_)
        );
      }
      else if (name == "FuseEstimator")
      {
        estimators_.push_back(
            std::make_unique<FuseEstimator>(estimator_config_)
        );
      }
      else
      {
        std::print(stderr, "[WARN] Unknown estimator specified: {}\n", name);
      }
    }
  }

  /**
   * @brief 分发已经读取的数据并执行对应的轨迹估计器。
   */
  void Run()
  {
    std::print(stderr, "[INFO] TrajectoryFactory estimation starting...\n");
    for (auto &estimator : estimators_)
    {
      std::print(stderr, "[INFO] Running estimator: {}\n",
                 estimator->GetName());
      estimator->Estimate();
    }
    std::print(stderr,
               "[INFO] Estimation finished. CSVs saved to directory: {}\n",
               estimator_config_.output_dir_);
  }
};

} // namespace FastVIO

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  try
  {
    auto factory{std::make_shared<FastVIO::TrajectoryFactory>()};
    factory->Run();
  }
  catch (const std::exception &ex)
  {
    std::println(stderr, "{}", ex.what());
  }
  rclcpp::shutdown();
  return 0;
}
