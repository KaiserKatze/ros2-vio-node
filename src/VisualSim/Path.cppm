module;

#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include <Eigen/Dense>

export module FastVIO.VisualSim:Path;

// import std;

import :Room;
import :StereoRig;

namespace FastVIO::VisualSim
{

template <typename value_type = double>
struct AbstractPath
{
  using Vector3            = Eigen::Vector<value_type, 3>;
  using Position           = Vector3;
  using LinearVelocity     = Vector3;
  using LinearAcceleration = Vector3;
  using Attitude           = Eigen::Matrix<value_type, 3, 3>;
  using AngularVelocity    = Vector3;
  using Pose               = std::pair<Position, Attitude>;
  using Kinematics
      = std::tuple<LinearVelocity, LinearAcceleration, AngularVelocity>;

  AbstractPath(value_type time_duration) : time_duration_{time_duration} {}

  virtual ~AbstractPath() {};

  virtual Pose GetPose(value_type time) const = 0;

  virtual Kinematics GetKinematics(value_type time) const = 0;

  virtual value_type GetDefaultDuration() const
  {
    return std::numeric_limits<value_type>::infinity();
  }

  value_type GetDuration() const;

  const value_type time_duration_;
};

template <typename value_type>
value_type AbstractPath<value_type>::GetDuration() const
{
  return std::min<value_type>(GetDefaultDuration(), time_duration_);
}

template <typename value_type = double>
class PathStationary : public AbstractPath<value_type>
{
public:
  using Vector3        = typename AbstractPath<value_type>::Vector3;
  using Position       = typename AbstractPath<value_type>::Position;
  using LinearVelocity = typename AbstractPath<value_type>::LinearVelocity;
  using LinearAcceleration =
      typename AbstractPath<value_type>::LinearAcceleration;
  using Attitude        = typename AbstractPath<value_type>::Attitude;
  using AngularVelocity = typename AbstractPath<value_type>::AngularVelocity;
  using Pose            = typename AbstractPath<value_type>::Pose;
  using Kinematics      = typename AbstractPath<value_type>::Kinematics;

  PathStationary(value_type time_duration, Position position,
                 Attitude attitude) :
    AbstractPath<value_type>{time_duration}, position_{position},
    attitude_{attitude}
  {
  }

  ~PathStationary() {}

  Pose GetPose(value_type time) const override
  {
    (void) time;
    return Pose{position_, attitude_};
  }

  Kinematics GetKinematics(value_type time) const override
  {
    (void) time;
    return Kinematics{
        LinearVelocity::Zero(),
        LinearAcceleration::Zero(),
        AngularVelocity::Zero(),
    };
  }

public:
  const Position &GetPosition() const
  {
    return position_;
  }

  const Attitude &GetAttitude() const
  {
    return attitude_;
  }

private:
  // 位置
  const Position position_;
  // 朝向
  const Attitude attitude_;
};

template <typename value_type = double>
class PathAcceleration : public AbstractPath<value_type>
{
public:
  using Vector3        = typename AbstractPath<value_type>::Vector3;
  using Position       = typename AbstractPath<value_type>::Position;
  using LinearVelocity = typename AbstractPath<value_type>::LinearVelocity;
  using LinearAcceleration =
      typename AbstractPath<value_type>::LinearAcceleration;
  using Attitude        = typename AbstractPath<value_type>::Attitude;
  using AngularVelocity = typename AbstractPath<value_type>::AngularVelocity;
  using Pose            = typename AbstractPath<value_type>::Pose;
  using Kinematics      = typename AbstractPath<value_type>::Kinematics;

  /**
   * @brief 构造函数
   * @param time_duration 轨迹持续时间 (单位: 秒)
   * @param position_start 匀加速直线运动位置起点 (单位: 米)
   * @param position_end 匀加速直线运动位置终点 (单位: 米)
   * @param linear_velocity_start_norm 匀加速直线运动初始线速度的大小 (单位: 米每秒，方向由 position_start 和 position_end 决定)
   * @param linear_acceleration_norm 匀加速直线运动线加速度的大小 (单位: 米每二次方秒，方向由 position_start 和 position_end 决定)
   * @param attitude 匀加速直线运动朝向 (单位: 无单位，函数内部会进行归一化处理)
   * @note 当 linear_acceleration_norm 足够小时，视作匀速直线运动，此时 time_duration 参数会被忽略，轨迹持续时间由 position_start、position_end 和 linear_velocity_start_norm 决定
   */
  PathAcceleration(value_type time_duration, Position position_start,
                   Position position_end, value_type linear_velocity_start_norm,
                   value_type linear_acceleration_norm, Attitude attitude) :
    AbstractPath<value_type>{time_duration}, position_start_{position_start},
    position_end_{position_end},
    linear_velocity_start_norm_{linear_velocity_start_norm},
    linear_acceleration_norm_{linear_acceleration_norm}, attitude_{attitude}
  {
  }

  ~PathAcceleration() {}

  Pose GetPose(value_type time) const override
  {
    // 运动方向
    Vector3 direction{(position_end_ - position_start_).normalized()};
    // 初始线速度向量
    LinearVelocity linear_velocity_start{linear_velocity_start_norm_
                                         * direction};
    // 线加速度
    LinearAcceleration linear_acceleration{linear_acceleration_norm_
                                           * direction};
    // 位置
    Position position{position_start_
                      + (linear_velocity_start
                         + static_cast<value_type>(0.5) * linear_acceleration
                               * time)
                            * time};
    return Pose{position, attitude_};
  }

  Kinematics GetKinematics(value_type time) const override
  {
    // 运动方向
    Vector3 direction{(position_end_ - position_start_).normalized()};
    // 初始线速度向量
    LinearVelocity linear_velocity_start{linear_velocity_start_norm_
                                         * direction};
    // 线加速度
    LinearAcceleration linear_acceleration{linear_acceleration_norm_
                                           * direction};
    // 当前线速度
    LinearVelocity linear_velocity_current{linear_velocity_start
                                           + linear_acceleration * time};
    return Kinematics{linear_velocity_current, linear_acceleration,
                      AngularVelocity::Zero()};
  }

  value_type GetDefaultDuration() const override
  {
    const value_type distance{(position_end_ - position_start_).norm()};
    if (linear_acceleration_norm_ < 1e-6)
    { // 当线加速度范数足够小时，视作匀速直线运动
      return distance / linear_velocity_start_norm_;
    }
    else
    {
      const value_type discriminant{
          linear_velocity_start_norm_ * linear_velocity_start_norm_
          + static_cast<value_type>(2.0) * linear_acceleration_norm_ * distance
      };
      return (std::sqrt(discriminant) - linear_velocity_start_norm_)
             / linear_acceleration_norm_;
    }
  }

public:
  const Position &GetPositionStart() const
  {
    return position_start_;
  }

  const Position &GetPositionEnd() const
  {
    return position_end_;
  }

  value_type GetLinearVelocityStartNorm() const
  {
    return linear_velocity_start_norm_;
  }

  value_type GetLinearAccelerationNorm() const
  {
    return linear_acceleration_norm_;
  }

  const Attitude &GetAttitude() const
  {
    return attitude_;
  }

private:
  // 匀加速直线运动位置起点
  const Position position_start_;
  // 匀加速直线运动位置终点
  const Position position_end_;
  // 匀加速直线运动初始线速度 (需要给出一个数值，表示线速度的大小，方向由 position_start_ 和 position_end_ 决定)
  const value_type linear_velocity_start_norm_;
  // 匀加速直线运动线加速度 (需要给出一个数值，表示线加速度的大小，方向由 position_start_ 和 position_end_ 决定)
  const value_type linear_acceleration_norm_;
  // 匀加速直线运动朝向
  const Attitude attitude_;
};

template <typename value_type = double>
class PathCircle : public AbstractPath<value_type>
{
public:
  using Vector3        = typename AbstractPath<value_type>::Vector3;
  using Position       = typename AbstractPath<value_type>::Position;
  using LinearVelocity = typename AbstractPath<value_type>::LinearVelocity;
  using LinearAcceleration =
      typename AbstractPath<value_type>::LinearAcceleration;
  using Attitude        = typename AbstractPath<value_type>::Attitude;
  using AngularVelocity = typename AbstractPath<value_type>::AngularVelocity;
  using Pose            = typename AbstractPath<value_type>::Pose;
  using Kinematics      = typename AbstractPath<value_type>::Kinematics;

  /**
   * @brief 相机朝向模式 (只影响朝向关于时间的函数)
   */
  enum class OrientationMode
  {
    LookAtCenter, // 朝向圆心
    Tangent,      // 沿切线方向 (线速度方向)
    Upward,       // 沿法线方向
  };

  /**
   * @brief 构造函数
   * @param time_duration 轨迹持续时间 (单位: 秒)
   * @param omega 匀速圆周运动的角速率 (单位: 弧度每秒)
   * @param center 匀速圆周运动的轨迹圆圆心 (单位: 米)
   * @param start 匀速圆周运动的轨迹出发点 (单位: 米)
   * @param norm 匀速圆周运动的轨迹所在平面的法向量 (单位: 无单位，函数内部会进行归一化处理)
   * @param mode 相机朝向模式
   */
  PathCircle(value_type time_duration, value_type omega, Position center,
             Position start, Vector3 norm, OrientationMode mode) :
    AbstractPath<value_type>{time_duration}, omega_{omega}, center_{center},
    start_{start}, norm_{norm}, mode_{mode}
  {
  }

  ~PathCircle() {}

  Pose GetPose(value_type time) const override
  {
    // 单位法向量
    Vector3 normalized_norm{norm_.normalized()};
    // 在轨迹圆所在平面内，构造两个正交向量
    Vector3 vec_u{start_ - center_};
    Vector3 vec_v{normalized_norm.cross(vec_u)};
    // 角位移
    value_type theta{omega_ * time};
    value_type cos_theta{std::cos(theta)};
    value_type sin_theta{std::sin(theta)};
    // 位置
    Position pos_body{center_ + vec_u * cos_theta + vec_v * sin_theta};

    // 相机画面中的水平方向
    Vector3 basis_x{Vector3::UnitX()};
    // 相机画面中的竖直方向
    Vector3 basis_y{Vector3::UnitY()};
    // 相机朝向
    Vector3 basis_z{Vector3::UnitZ()};

    switch (mode_)
    {
    case OrientationMode::LookAtCenter:
    {
      basis_z = (center_ - pos_body).normalized();   // 朝向圆心
      basis_y = -normalized_norm;                    // 朝向下方
      basis_x = basis_y.cross(basis_z).normalized(); // 朝向右侧
      break;
    }
    case OrientationMode::Tangent:
    {
      // 逆时针运动的切线方向
      basis_z = -sin_theta * vec_u + cos_theta * vec_v;
      basis_y = -normalized_norm;                    // 朝向下方
      basis_x = basis_y.cross(basis_z).normalized(); // 朝向右侧
      break;
    }
    case OrientationMode::Upward:
    {
      basis_z = normalized_norm;                     // 指向天花板
      basis_x = (center_ - pos_body).normalized();   // 朝向圆心
      basis_y = basis_z.cross(basis_x).normalized(); // 朝向运动的反方向
      break;
    }
    }

    // 朝向
    Attitude att_body;
    att_body.col(0) = basis_x;
    att_body.col(1) = basis_y;
    att_body.col(2) = basis_z;

    return Pose{pos_body, att_body};
  }

  Kinematics GetKinematics(value_type time) const override
  {
    // 单位法向量
    Vector3 normalized_norm{norm_.normalized()};
    // 角速度
    AngularVelocity angular_velocity{omega_ * normalized_norm};
    // 在轨迹圆所在平面内，构造两个正交向量
    Vector3 vec_u{start_ - center_};
    Vector3 vec_v{normalized_norm.cross(vec_u)};
    // 角位移
    value_type theta{omega_ * time};
    value_type cos_theta{std::cos(theta)};
    value_type sin_theta{std::sin(theta)};
    // 线速度
    LinearVelocity linear_velocity{omega_
                                   * (vec_u * -sin_theta + vec_v * cos_theta)};
    // 线加速度
    LinearAcceleration linear_acceleration{
        omega_ * omega_ * (vec_u * -cos_theta + vec_v * -sin_theta)
    };
    return Kinematics{linear_velocity, linear_acceleration, angular_velocity};
  }

public:
  value_type GetOmega() const
  {
    return omega_;
  }

  const Position &GetPositionCenter() const
  {
    return center_;
  }

  const Position &GetPositionStart() const
  {
    return start_;
  }

  const Vector3 &GetNorm() const
  {
    return norm_;
  }

  OrientationMode GetOrientationMode() const
  {
    return mode_;
  }

private:
  // 匀速圆周运动的角速率
  const value_type omega_;
  // 匀速圆周运动的轨迹圆圆心
  const Position center_;
  // 匀速圆周运动的轨迹出发点
  const Position start_;
  // 匀速圆周运动的轨迹所在平面的法向量
  const Vector3 norm_;
  // 相机朝向
  const OrientationMode mode_;
};

template <typename value_type = double>
class PathManager
{
public:
  using Pose       = typename AbstractPath<value_type>::Pose;
  using Kinematics = typename AbstractPath<value_type>::Kinematics;

  PathManager(const Room<value_type> &room) : room_{room} {}

  void AddPath(std::unique_ptr<AbstractPath<value_type>> &&path);

  /**
   * @brief 获取世界坐标系下的位姿参数 (位置 $r^{vi}_i$、朝向 $C_{iv}$)
   */
  std::optional<Pose> GetPose(value_type time) const;

  /**
   * @brief 获取世界坐标系下的动力学参数 (线速度 $\dot{r}^{vi}_i$、角速度 $\omega^{iv}_i$、线加速度 $\ddot{r}^{vi}_i$)
   */
  std::optional<Kinematics> GetKinematics(value_type time) const;

  /**
   * @brief 让双目相机 rig 绕着房间的几何中心，在平行于地板的平面内，做匀速圆周运动
   */
  std::optional<typename StereoRig<value_type>::Frame>
  GetImage(const StereoRig<value_type> &rig, value_type time) const;

  auto operator[](std::size_t index) const;

  value_type GetDuration() const;

private:
  const Room<value_type> &room_;
  std::vector<std::unique_ptr<AbstractPath<value_type>>> queue_path_;
};

template <typename value_type>
void PathManager<value_type>::AddPath(
    std::unique_ptr<AbstractPath<value_type>> &&path
)
{
  queue_path_.push_back(std::move(path));
}

template <typename value_type>
std::optional<typename PathManager<value_type>::Pose>
PathManager<value_type>::GetPose(value_type time) const
{
  value_type time_elapsed{0.0};
  for (const std::unique_ptr<AbstractPath<value_type>> &path : queue_path_)
  {
    value_type duration{path->GetDuration()};
    if (time_elapsed <= time && time < time_elapsed + duration)
    {
      return path->GetPose(time - time_elapsed);
    }
    time_elapsed += duration;
  }
  return std::nullopt;
}

template <typename value_type>
std::optional<typename PathManager<value_type>::Kinematics>
PathManager<value_type>::GetKinematics(value_type time) const
{
  value_type time_elapsed{0.0};
  for (const std::unique_ptr<AbstractPath<value_type>> &path : queue_path_)
  {
    value_type duration{path->GetDuration()};
    if (time_elapsed <= time && time < time_elapsed + duration)
    {
      return path->GetKinematics(time - time_elapsed);
    }
    time_elapsed += duration;
  }
  return std::nullopt;
}

template <typename value_type>
std::optional<typename StereoRig<value_type>::Frame>
PathManager<value_type>::GetImage(const StereoRig<value_type> &rig,
                                  value_type time) const
{
  const auto opt_pose{GetPose(time)};
  if (!opt_pose.has_value())
  {
    return std::nullopt;
  }
  const typename PathManager<value_type>::Pose &pose{opt_pose.value()};
  const auto &pos_body{std::get<0>(pose)};
  const auto &att_body{std::get<1>(pose)};
  return std::make_optional(rig.Project(room_.object_matrix_,
                                        att_body.transpose(),
                                        -att_body.transpose() * pos_body));
}

template <typename value_type>
auto PathManager<value_type>::operator[](std::size_t index) const
{
  return queue_path_[index];
}

template <typename value_type>
value_type PathManager<value_type>::GetDuration() const
{
  value_type total_duration{0.0};
  for (const std::unique_ptr<AbstractPath<value_type>> &path : queue_path_)
  {
    value_type duration{path->GetDuration()};
    total_duration += duration;
  }
  return total_duration;
}

} // namespace FastVIO::VisualSim
