#ifndef IMUSTATE_HPP
#define IMUSTATE_HPP

#include <array>
#include <type_traits>

#include <opencv2/core.hpp>

#include <Eigen/Dense>

template <typename T, typename E, size_t S>
concept VectorLike = std::is_same_v<T, typename cv::Vec<E, S>>
                     || std::is_same_v<T, typename Eigen::Matrix<E, S, 1>>;

template <typename T>
concept VectorLike3d = VectorLike<T, double, 3>;

template <typename T>
concept VectorLike4d = VectorLike<T, double, 4>;

// 状态量定义: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (大小为 10)
struct ImuState : public std::array<double, 10>
{
  ImuState()
      : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
  {
  }

  double &GetPositionX()
  {
    return (*this)[0];
  }

  double &GetPositionY()
  {
    return (*this)[1];
  }

  double &GetPositionZ()
  {
    return (*this)[2];
  }

  double &GetVelocityX()
  {
    return (*this)[3];
  }

  double &GetVelocityY()
  {
    return (*this)[4];
  }

  double &GetVelocityZ()
  {
    return (*this)[5];
  }

  double &GetQuaternionW()
  {
    return (*this)[6];
  }

  double &GetQuaternionX()
  {
    return (*this)[7];
  }

  double &GetQuaternionY()
  {
    return (*this)[8];
  }

  double &GetQuaternionZ()
  {
    return (*this)[9];
  }

  double GetPositionX() const
  {
    return (*this)[0];
  }

  double GetPositionY() const
  {
    return (*this)[1];
  }

  double GetPositionZ() const
  {
    return (*this)[2];
  }

  Eigen::Vector3d GetPosition() const
  {
    Eigen::Vector3d pos;
    pos << GetPositionX(), GetPositionY(), GetPositionZ();
    return pos;
  }

  double GetVelocityX() const
  {
    return (*this)[3];
  }

  double GetVelocityY() const
  {
    return (*this)[4];
  }

  double GetVelocityZ() const
  {
    return (*this)[5];
  }

  Eigen::Vector3d GetVelocity() const
  {
    Eigen::Vector3d vel;
    vel << GetVelocityX(), GetVelocityY(), GetVelocityZ();
    return vel;
  }

  double GetQuaternionW() const
  {
    return (*this)[6];
  }

  double GetQuaternionX() const
  {
    return (*this)[7];
  }

  double GetQuaternionY() const
  {
    return (*this)[8];
  }

  double GetQuaternionZ() const
  {
    return (*this)[9];
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
    (*this)[0] = px;
    (*this)[1] = py;
    (*this)[2] = pz;
  }

  void SetPosition(const VectorLike3d auto &pos)
  {
    (*this)[0] = pos[0];
    (*this)[1] = pos[1];
    (*this)[2] = pos[2];
  }

  void SetVelocity(double vx, double vy, double vz)
  {
    (*this)[3] = vx;
    (*this)[4] = vy;
    (*this)[5] = vz;
  }

  void SetVelocity(const VectorLike3d auto &vel)
  {
    (*this)[3] = vel[0];
    (*this)[4] = vel[1];
    (*this)[5] = vel[2];
  }

  void SetQuaternion(double qw, double qx, double qy, double qz)
  {
    (*this)[6] = qw;
    (*this)[7] = qx;
    (*this)[8] = qy;
    (*this)[9] = qz;
  }

  void SetQuaternion(const VectorLike4d auto &quat)
  {
    (*this)[6] = quat[0];
    (*this)[7] = quat[1];
    (*this)[8] = quat[2];
    (*this)[9] = quat[3];
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

struct ImuDerivative : public std::array<double, 10>
{
  ImuDerivative()
      : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  {
  }

  void SetVelocity(const VectorLike3d auto &velocity)
  {
    (*this)[0] = velocity[0];
    (*this)[1] = velocity[1];
    (*this)[2] = velocity[2];
  }

  void SetAcceleration(const VectorLike3d auto &acceleration)
  {
    (*this)[3] = acceleration[0];
    (*this)[4] = acceleration[1];
    (*this)[5] = acceleration[2];
  }

  void SetQuaternionDerivative(const VectorLike4d auto &quat_derivative)
  {
    (*this)[6] = quat_derivative[0];
    (*this)[7] = quat_derivative[1];
    (*this)[8] = quat_derivative[2];
    (*this)[9] = quat_derivative[3];
  }

  void SetQuaternionDerivative(const Eigen::Quaterniond &quat_derivative)
  {
    (*this)[6] = quat_derivative.w();
    (*this)[7] = quat_derivative.x();
    (*this)[8] = quat_derivative.y();
    (*this)[9] = quat_derivative.z();
  }
};

#endif /* IMUSTATE_HPP */
