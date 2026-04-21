#ifndef IMUSTATE_HPP
#define IMUSTATE_HPP

#include <array>
#include <concepts>

template <typename T, typename Idx = std::size_t>
concept VectorLike = requires(T t, Idx i) {
  { t[i] } -> std::convertible_to<double>;
};

#include <Eigen/Dense>

static constexpr size_t index_px = 0;
static constexpr size_t index_py = 1;
static constexpr size_t index_pz = 2;
static constexpr size_t index_vx = 3;
static constexpr size_t index_vy = 4;
static constexpr size_t index_vz = 5;
static constexpr size_t index_qw = 6;
static constexpr size_t index_qx = 7;
static constexpr size_t index_qy = 8;
static constexpr size_t index_qz = 9;

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
struct ImuState : public std::array<double, 10>
{
  ImuState() :
    std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
  {
  }

  double &GetPositionX()
  {
    return (*this)[index_px];
  }

  double &GetPositionY()
  {
    return (*this)[index_py];
  }

  double &GetPositionZ()
  {
    return (*this)[index_pz];
  }

  double &GetVelocityX()
  {
    return (*this)[index_vx];
  }

  double &GetVelocityY()
  {
    return (*this)[index_vy];
  }

  double &GetVelocityZ()
  {
    return (*this)[index_vz];
  }

  double &GetQuaternionW()
  {
    return (*this)[index_qw];
  }

  double &GetQuaternionX()
  {
    return (*this)[index_qx];
  }

  double &GetQuaternionY()
  {
    return (*this)[index_qy];
  }

  double &GetQuaternionZ()
  {
    return (*this)[index_qz];
  }

  double GetPositionX() const
  {
    return (*this)[index_px];
  }

  double GetPositionY() const
  {
    return (*this)[index_py];
  }

  double GetPositionZ() const
  {
    return (*this)[index_pz];
  }

  Eigen::Vector3d GetPosition() const
  {
    Eigen::Vector3d pos;
    pos << GetPositionX(), GetPositionY(), GetPositionZ();
    return pos;
  }

  double GetVelocityX() const
  {
    return (*this)[index_vx];
  }

  double GetVelocityY() const
  {
    return (*this)[index_vy];
  }

  double GetVelocityZ() const
  {
    return (*this)[index_vz];
  }

  Eigen::Vector3d GetVelocity() const
  {
    Eigen::Vector3d vel;
    vel << GetVelocityX(), GetVelocityY(), GetVelocityZ();
    return vel;
  }

  double GetQuaternionW() const
  {
    return (*this)[index_qw];
  }

  double GetQuaternionX() const
  {
    return (*this)[index_qx];
  }

  double GetQuaternionY() const
  {
    return (*this)[index_qy];
  }

  double GetQuaternionZ() const
  {
    return (*this)[index_qz];
  }

  Eigen::Quaterniond GetQuaternion() const
  {
    Eigen::Quaterniond q{GetQuaternionW(), GetQuaternionX(), GetQuaternionY(),
                         GetQuaternionZ()};
    q.normalize();
    return q;
  }

  void SetPosition(double px, double py, double pz)
  {
    (*this)[index_px] = px;
    (*this)[index_py] = py;
    (*this)[index_pz] = pz;
  }

  void SetPosition(const VectorLike auto &pos)
  {
    this->SetPosition(pos[0], pos[1], pos[2]);
  }

  void SetVelocity(double vx, double vy, double vz)
  {
    (*this)[index_vx] = vx;
    (*this)[index_vy] = vy;
    (*this)[index_vz] = vz;
  }

  void SetVelocity(const VectorLike auto &vel)
  {
    this->SetVelocity(vel[0], vel[1], vel[2]);
  }

  void SetQuaternion(double qw, double qx, double qy, double qz)
  {
    (*this)[index_qw] = qw;
    (*this)[index_qx] = qx;
    (*this)[index_qy] = qy;
    (*this)[index_qz] = qz;
    // 必须对四元数进行归一化，保证单位模长约束
    this->NormalizeQuaternion();
  }

  void SetQuaternion(const VectorLike auto &quat)
  {
    this->SetQuaternion(quat[0], quat[1], quat[2], quat[3]);
  }

  void SetQuaternion(const Eigen::Quaterniond &quat)
  {
    this->SetQuaternion(quat.w(), quat.x(), quat.y(), quat.z());
  }

  void NormalizeQuaternion()
  {
    double &qw{GetQuaternionW()};
    double &qx{GetQuaternionX()};
    double &qy{GetQuaternionY()};
    double &qz{GetQuaternionZ()};
    const double q_norm{std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz)};
    qw /= q_norm;
    qx /= q_norm;
    qy /= q_norm;
    qz /= q_norm;
  }
};

// 状态量定义: [vx, vy, vz, ax, ay, az, qw', qx', qy', qz'] (大小为 10)
struct ImuDerivative : public std::array<double, 10>
{
  ImuDerivative() :
    std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  {
  }

  void SetVelocity(double vx, double vy, double vz)
  {
    (*this)[index_px] = vx;
    (*this)[index_py] = vy;
    (*this)[index_pz] = vz;
  }

  void SetVelocity(const VectorLike auto &velocity)
  {
    this->SetVelocity(velocity[0], velocity[1], velocity[2]);
  }

  void SetAcceleration(double ax, double ay, double az)
  {
    (*this)[index_vx] = ax;
    (*this)[index_vy] = ay;
    (*this)[index_vz] = az;
  }

  void SetAcceleration(const VectorLike auto &acceleration)
  {
    this->SetAcceleration(acceleration[0], acceleration[1], acceleration[2]);
  }

  void SetQuaternionDerivative(double qw, double qx, double qy, double qz)
  {
    (*this)[index_qw] = qw;
    (*this)[index_qx] = qx;
    (*this)[index_qy] = qy;
    (*this)[index_qz] = qz;
  }

  void SetQuaternionDerivative(const VectorLike auto &quat_derivative)
  {
    this->SetQuaternionDerivative(quat_derivative[0], quat_derivative[1],
                                  quat_derivative[2], quat_derivative[3]);
  }

  void SetQuaternionDerivative(const Eigen::Quaterniond &quat_derivative)
  {
    this->SetQuaternionDerivative(quat_derivative.w(), quat_derivative.x(),
                                  quat_derivative.y(), quat_derivative.z());
  }
};

#endif /* IMUSTATE_HPP */
