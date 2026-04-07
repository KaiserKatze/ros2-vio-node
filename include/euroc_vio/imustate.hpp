#ifndef IMUSTATE_HPP
#define IMUSTATE_HPP

#include <array>

#include <opencv2/core.hpp>

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

  cv::Vec3d GetPosition() const
  {
    return cv::Vec3d(GetPositionX(), GetPositionY(), GetPositionZ());
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

  cv::Vec3d GetVelocity() const
  {
    return cv::Vec3d(GetVelocityX(), GetVelocityY(), GetVelocityZ());
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

  void SetPosition(double px, double py, double pz)
  {
    (*this)[0] = px;
    (*this)[1] = py;
    (*this)[2] = pz;
  }

  void SetPosition(const cv::Vec3d &pos)
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

  void SetVelocity(const cv::Vec3d &vel)
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

  void SetQuaternion(const cv::Vec4f &quat)
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

  void SetVelocity(const cv::Vec3d &velocity)
  {
    (*this)[0] = velocity[0];
    (*this)[1] = velocity[1];
    (*this)[2] = velocity[2];
  }

  void SetAcceleration(const cv::Vec3d &acceleration)
  {
    (*this)[3] = acceleration[0];
    (*this)[4] = acceleration[1];
    (*this)[5] = acceleration[2];
  }

  void SetQuaternionDerivative(const cv::Vec4d &quat_derivative)
  {
    (*this)[6] = quat_derivative[0];
    (*this)[7] = quat_derivative[1];
    (*this)[8] = quat_derivative[2];
    (*this)[9] = quat_derivative[3];
  }
};

#endif /* IMUSTATE_HPP */
