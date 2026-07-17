module;

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>

#include <Eigen/Dense>

export module FastVIO:Sensor;

// import std;

template <typename T, typename Idx = std::size_t>
concept VectorLike = requires(T t, Idx i) {
  { t[i] } -> std::convertible_to<double>;
};

inline constexpr std::size_t index_px{0};
inline constexpr std::size_t index_py{1};
inline constexpr std::size_t index_pz{2};
inline constexpr std::size_t index_vx{3};
inline constexpr std::size_t index_vy{4};
inline constexpr std::size_t index_vz{5};
inline constexpr std::size_t index_qw{6};
inline constexpr std::size_t index_qx{7};
inline constexpr std::size_t index_qy{8};
inline constexpr std::size_t index_qz{9};

export namespace FastVIO
{

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
template <typename value_type>
struct ImuState : public std::array<value_type, 10>
{
  ImuState() :
    std::array<value_type, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
  {
  }

  ImuState(std::initializer_list<value_type> &&data) :
    std::array<value_type, 10>{data}
  {
  }

  value_type &GetPositionX()
  {
    return (*this)[index_px];
  }

  value_type &GetPositionY()
  {
    return (*this)[index_py];
  }

  value_type &GetPositionZ()
  {
    return (*this)[index_pz];
  }

  value_type &GetVelocityX()
  {
    return (*this)[index_vx];
  }

  value_type &GetVelocityY()
  {
    return (*this)[index_vy];
  }

  value_type &GetVelocityZ()
  {
    return (*this)[index_vz];
  }

  value_type &GetAttitudeW()
  {
    return (*this)[index_qw];
  }

  value_type &GetAttitudeX()
  {
    return (*this)[index_qx];
  }

  value_type &GetAttitudeY()
  {
    return (*this)[index_qy];
  }

  value_type &GetAttitudeZ()
  {
    return (*this)[index_qz];
  }

  value_type GetPositionX() const
  {
    return (*this)[index_px];
  }

  value_type GetPositionY() const
  {
    return (*this)[index_py];
  }

  value_type GetPositionZ() const
  {
    return (*this)[index_pz];
  }

  Eigen::Vector<value_type, 3> GetPosition() const
  {
    Eigen::Vector<value_type, 3> pos{GetPositionX(), GetPositionY(),
                                     GetPositionZ()};
    return pos;
  }

  value_type GetVelocityX() const
  {
    return (*this)[index_vx];
  }

  value_type GetVelocityY() const
  {
    return (*this)[index_vy];
  }

  value_type GetVelocityZ() const
  {
    return (*this)[index_vz];
  }

  Eigen::Vector<value_type, 3> GetVelocity() const
  {
    Eigen::Vector<value_type, 3> vel{GetVelocityX(), GetVelocityY(),
                                     GetVelocityZ()};
    return vel;
  }

  value_type GetAttitudeW() const
  {
    return (*this)[index_qw];
  }

  value_type GetAttitudeX() const
  {
    return (*this)[index_qx];
  }

  value_type GetAttitudeY() const
  {
    return (*this)[index_qy];
  }

  value_type GetAttitudeZ() const
  {
    return (*this)[index_qz];
  }

  Eigen::Quaternion<value_type> GetAttitude() const
  {
    Eigen::Quaternion<value_type> q{GetAttitudeW(), GetAttitudeX(),
                                    GetAttitudeY(), GetAttitudeZ()};
    q.normalize();
    return q;
  }

  void SetPosition(value_type px, value_type py, value_type pz)
  {
    (*this)[index_px] = px;
    (*this)[index_py] = py;
    (*this)[index_pz] = pz;
  }

  void SetPosition(const VectorLike auto &pos)
  {
    this->SetPosition(pos[0], pos[1], pos[2]);
  }

  void SetVelocity(value_type vx, value_type vy, value_type vz)
  {
    (*this)[index_vx] = vx;
    (*this)[index_vy] = vy;
    (*this)[index_vz] = vz;
  }

  void SetVelocity(const VectorLike auto &vel)
  {
    this->SetVelocity(vel[0], vel[1], vel[2]);
  }

  void SetAttitude(value_type qw, value_type qx, value_type qy, value_type qz)
  {
    (*this)[index_qw] = qw;
    (*this)[index_qx] = qx;
    (*this)[index_qy] = qy;
    (*this)[index_qz] = qz;
    // 必须对四元数进行归一化，保证单位模长约束
    this->NormalizeAttitude();
  }

  void SetAttitude(const VectorLike auto &quat)
  {
    this->SetAttitude(quat[0], quat[1], quat[2], quat[3]);
  }

  void SetAttitude(const Eigen::Quaternion<value_type> &quat)
  {
    this->SetAttitude(quat.w(), quat.x(), quat.y(), quat.z());
  }

  void NormalizeAttitude()
  {
    value_type &qw{GetAttitudeW()};
    value_type &qx{GetAttitudeX()};
    value_type &qy{GetAttitudeY()};
    value_type &qz{GetAttitudeZ()};
    const value_type q_norm{std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz)};
    qw /= q_norm;
    qx /= q_norm;
    qy /= q_norm;
    qz /= q_norm;
  }
};

// 状态量定义: [vx, vy, vz, ax, ay, az, qw', qx', qy', qz'] (大小为 10)
template <typename value_type>
struct ImuDerivative : public std::array<value_type, 10>
{
  ImuDerivative() :
    std::array<value_type, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  {
  }

  void SetVelocity(value_type vx, value_type vy, value_type vz)
  {
    (*this)[index_px] = vx;
    (*this)[index_py] = vy;
    (*this)[index_pz] = vz;
  }

  void SetVelocity(const VectorLike auto &velocity)
  {
    this->SetVelocity(velocity[0], velocity[1], velocity[2]);
  }

  void SetAcceleration(value_type ax, value_type ay, value_type az)
  {
    (*this)[index_vx] = ax;
    (*this)[index_vy] = ay;
    (*this)[index_vz] = az;
  }

  void SetAcceleration(const VectorLike auto &acceleration)
  {
    this->SetAcceleration(acceleration[0], acceleration[1], acceleration[2]);
  }

  void SetAttitudeDerivative(value_type qw, value_type qx, value_type qy,
                             value_type qz)
  {
    (*this)[index_qw] = qw;
    (*this)[index_qx] = qx;
    (*this)[index_qy] = qy;
    (*this)[index_qz] = qz;
  }

  void SetAttitudeDerivative(const VectorLike auto &quat_derivative)
  {
    this->SetAttitudeDerivative(quat_derivative[0], quat_derivative[1],
                                quat_derivative[2], quat_derivative[3]);
  }

  void
  SetAttitudeDerivative(const Eigen::Quaternion<value_type> &quat_derivative)
  {
    this->SetAttitudeDerivative(quat_derivative.w(), quat_derivative.x(),
                                quat_derivative.y(), quat_derivative.z());
  }
};

} // namespace FastVIO
