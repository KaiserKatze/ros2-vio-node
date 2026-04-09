//=====================================================================================================
// ahrs.hpp
//=====================================================================================================
//
// Madgwick's implementation of Mayhony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//
// Date			Author			Notes
// 29/09/2011	SOH Madgwick    Initial release
// 02/10/2011	SOH Madgwick	Optimised for reduced CPU load
// 19/02/2012	SOH Madgwick	Magnetometer measurement is normalised
//
//=====================================================================================================

#ifndef AHRS_HPP
#define AHRS_HPP

//---------------------------------------------------------------------------------------------------
// Header files

#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include <opencv2/core.hpp>

static constexpr float betaDef{0.1f};         // 2 * proportional gain
static constexpr float twoKpDef{2.0f * 0.5f}; // 2 * proportional gain
static constexpr float twoKiDef{2.0f * 0.0f}; // 2 * integral gain

template <std::floating_point T> struct FloatInt
{
  using type = std::conditional_t<
      sizeof(T) == 1, std::int8_t,
      std::conditional_t<
          sizeof(T) == 2, std::int16_t,
          std::conditional_t<
              sizeof(T) == 4, std::int32_t,
              std::conditional_t<sizeof(T) == 8, std::int64_t, void>>>>;
  static_assert(!std::is_same_v<type, void>, "Unsupported float type");
};

template <std::floating_point T> struct MagicNumber
{
};

template <> struct MagicNumber<double>
{
  static constexpr typename FloatInt<double>::type value{0x5fe6eb50c7b537a9ll};
};

template <> struct MagicNumber<float>
{
  static constexpr typename FloatInt<float>::type value{0x5f3759df};
};

/**
 * @brief Fast inverse square-root calculation
 * @param x Input value
 * @return Inverse square-root of x
 * @note http://en.wikipedia.org/wiki/Fast_inverse_square_root
 */
template <std::floating_point FloatType> FloatType invSqrt(FloatType x)
{
  using fint = typename FloatInt<FloatType>::type;
  union
  {
    FloatType f;
    fint i;
  } fi;
  fi.f = x;
  FloatType y{static_cast<FloatType>(0.5) * x};
  fi.i = MagicNumber<FloatType>::value - (fi.i >> 1);
  FloatType z{fi.f};
  z = z * (static_cast<FloatType>(1.5) - (y * z * z));
  return z;
}

struct AbstractAHRS
{
  virtual void Update(cv::Vec3f &gyro, cv::Vec3f &accel, float dt) = 0;
  virtual void Update(cv::Vec3f &gyro, cv::Vec3f &accel, cv::Vec3f &mag,
                      float dt)                                    = 0;
  virtual cv::Vec4f GetQuaternion() const                          = 0;
};

struct MahonyAHRS : public AbstractAHRS
{
  volatile float twoKp{twoKpDef}; // 2 * proportional gain (Kp)
  volatile float twoKi{twoKiDef}; // 2 * integral gain (Ki)
  volatile float beta{betaDef};   // 2 * proportional gain (Kp)

  // quaternion of sensor frame relative to auxiliary frame
  cv::Vec4f q{1.0f, 0.0f, 0.0f, 0.0f};
  // integral error terms scaled by Ki
  cv::Vec3f integralFB{0.0f, 0.0f, 0.0f};

  void Update(cv::Vec3f &gyro, cv::Vec3f &accel, float dt) override
  {
    float recipNorm;
    float halfvx, halfvy, halfvz;
    float halfex, halfey, halfez;
    float qa, qb, qc;

    float &gx{gyro[0]}, &gy{gyro[1]}, &gz{gyro[2]};
    float &ax{accel[0]}, &ay{accel[1]}, &az{accel[2]};
    float &q0{q[0]}, &q1{q[1]}, &q2{q[2]}, &q3{q[3]};

    // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {

      // Normalise accelerometer measurement
      recipNorm = invSqrt(ax * ax + ay * ay + az * az);
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      // Estimated direction of gravity and vector perpendicular to magnetic flux
      halfvx = q1 * q3 - q0 * q2;
      halfvy = q0 * q1 + q2 * q3;
      halfvz = q0 * q0 - 0.5f + q3 * q3;

      // Error is sum of cross product between estimated and measured direction of gravity
      halfex = (ay * halfvz - az * halfvy);
      halfey = (az * halfvx - ax * halfvz);
      halfez = (ax * halfvy - ay * halfvx);

      // Compute and apply integral feedback if enabled
      if (twoKi > 0.0f)
      {
        // integral error scaled by Ki
        integralFB += twoKi * cv::Vec3f{halfex, halfey, halfez} * dt;
        // apply integral feedback
        gyro += integralFB;
      }
      else
      {
        // prevent integral windup
        integralFB = cv::Vec3f{0.0f, 0.0f, 0.0f};
      }

      // Apply proportional feedback
      gx += twoKp * halfex;
      gy += twoKp * halfey;
      gz += twoKp * halfez;
    }

    // Integrate rate of change of quaternion
    gx *= (0.5f * dt); // pre-multiply common factors
    gy *= (0.5f * dt);
    gz *= (0.5f * dt);
    qa = q0;
    qb = q1;
    qc = q2;
    q0 += (-qb * gx - qc * gy - q3 * gz);
    q1 += (qa * gx + qc * gz - q3 * gy);
    q2 += (qa * gy - qb * gz + q3 * gx);
    q3 += (qa * gz + qb * gy - qc * gx);

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
  }

  void Update(cv::Vec3f &gyro, cv::Vec3f &accel, cv::Vec3f &mag,
              float dt) override
  {
    float recipNorm;
    float q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3;
    float hx, hy, bx, bz;
    float halfvx, halfvy, halfvz, halfwx, halfwy, halfwz;
    float halfex, halfey, halfez;
    float qa, qb, qc;

    float &gx{gyro[0]}, &gy{gyro[1]}, &gz{gyro[2]};
    float &ax{accel[0]}, &ay{accel[1]}, &az{accel[2]};
    float &mx{mag[0]}, &my{mag[1]}, &mz{mag[2]};
    float &q0{q[0]}, &q1{q[1]}, &q2{q[2]}, &q3{q[3]};

    // Use IMU algorithm if magnetometer measurement invalid (avoids NaN in magnetometer normalisation)
    if ((mx == 0.0f) && (my == 0.0f) && (mz == 0.0f))
    {
      Update(gyro, accel, dt);
      return;
    }

    // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {

      // Normalise accelerometer measurement
      recipNorm = invSqrt(ax * ax + ay * ay + az * az);
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      // Normalise magnetometer measurement
      recipNorm = invSqrt(mx * mx + my * my + mz * mz);
      mx *= recipNorm;
      my *= recipNorm;
      mz *= recipNorm;

      // Auxiliary variables to avoid repeated arithmetic
      q0q0 = q0 * q0;
      q0q1 = q0 * q1;
      q0q2 = q0 * q2;
      q0q3 = q0 * q3;
      q1q1 = q1 * q1;
      q1q2 = q1 * q2;
      q1q3 = q1 * q3;
      q2q2 = q2 * q2;
      q2q3 = q2 * q3;
      q3q3 = q3 * q3;

      // Reference direction of Earth's magnetic field
      hx = 2.0f
           * (mx * (0.5f - q2q2 - q3q3) + my * (q1q2 - q0q3)
              + mz * (q1q3 + q0q2));
      hy = 2.0f
           * (mx * (q1q2 + q0q3) + my * (0.5f - q1q1 - q3q3)
              + mz * (q2q3 - q0q1));
      bx = sqrt(hx * hx + hy * hy);
      bz = 2.0f
           * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1)
              + mz * (0.5f - q1q1 - q2q2));

      // Estimated direction of gravity and magnetic field
      halfvx = q1q3 - q0q2;
      halfvy = q0q1 + q2q3;
      halfvz = q0q0 - 0.5f + q3q3;
      halfwx = bx * (0.5f - q2q2 - q3q3) + bz * (q1q3 - q0q2);
      halfwy = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3);
      halfwz = bx * (q0q2 + q1q3) + bz * (0.5f - q1q1 - q2q2);

      // Error is sum of cross product between estimated direction and measured direction of field vectors
      halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy);
      halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz);
      halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx);

      // Compute and apply integral feedback if enabled
      if (twoKi > 0.0f)
      {
        // integral error scaled by Ki
        integralFB += twoKi * cv::Vec3f{halfex, halfey, halfez} * dt;
        // apply integral feedback
        gyro += integralFB;
      }
      else
      {
        // prevent integral windup
        integralFB = cv::Vec3f{0.0f, 0.0f, 0.0f};
      }

      // Apply proportional feedback
      gx += twoKp * halfex;
      gy += twoKp * halfey;
      gz += twoKp * halfez;
    }

    // Integrate rate of change of quaternion
    gx *= (0.5f * dt); // pre-multiply common factors
    gy *= (0.5f * dt);
    gz *= (0.5f * dt);
    qa = q0;
    qb = q1;
    qc = q2;
    q0 += (-qb * gx - qc * gy - q3 * gz);
    q1 += (qa * gx + qc * gz - q3 * gy);
    q2 += (qa * gy - qb * gz + q3 * gx);
    q3 += (qa * gz + qb * gy - qc * gx);

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
  }

  cv::Vec4f GetQuaternion() const override
  {
    return q;
  }
};

struct MadgwickAHRS : public AbstractAHRS
{
  volatile float beta{betaDef}; // 2 * proportional gain (Kp)

  // quaternion of sensor frame relative to auxiliary frame
  cv::Vec4f q{1.0f, 0.0f, 0.0f, 0.0f};

  void Update(cv::Vec3f &gyro, cv::Vec3f &accel, float dt) override
  {
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;
    float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1,
        q2q2, q3q3;

    float &gx{gyro[0]}, &gy{gyro[1]}, &gz{gyro[2]};
    float &ax{accel[0]}, &ay{accel[1]}, &az{accel[2]};
    float &q0{q[0]}, &q1{q[1]}, &q2{q[2]}, &q3{q[3]};

    // Rate of change of quaternion from gyroscope
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {

      // Normalise accelerometer measurement
      recipNorm = invSqrt(ax * ax + ay * ay + az * az);
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      // Auxiliary variables to avoid repeated arithmetic
      _2q0 = 2.0f * q0;
      _2q1 = 2.0f * q1;
      _2q2 = 2.0f * q2;
      _2q3 = 2.0f * q3;
      _4q0 = 4.0f * q0;
      _4q1 = 4.0f * q1;
      _4q2 = 4.0f * q2;
      _8q1 = 8.0f * q1;
      _8q2 = 8.0f * q2;
      q0q0 = q0 * q0;
      q1q1 = q1 * q1;
      q2q2 = q2 * q2;
      q3q3 = q3 * q3;

      // Gradient decent algorithm corrective step
      s0        = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
      s1        = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1
                  + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
      s2        = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2
                  + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
      s3        = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
      recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2
                          + s3 * s3); // normalise step magnitude
      s0 *= recipNorm;
      s1 *= recipNorm;
      s2 *= recipNorm;
      s3 *= recipNorm;

      // Apply feedback step
      qDot1 -= beta * s0;
      qDot2 -= beta * s1;
      qDot3 -= beta * s2;
      qDot4 -= beta * s3;
    }

    // Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
  }

  void Update(cv::Vec3f &gyro, cv::Vec3f &accel, cv::Vec3f &mag,
              float dt) override
  {
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;
    float hx, hy;
    float _2q0mx, _2q0my, _2q0mz, _2q1mx, _2bx, _2bz, _4bx, _4bz, _2q0, _2q1,
        _2q2, _2q3, _2q0q2, _2q2q3, q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3,
        q2q2, q2q3, q3q3;

    float &gx{gyro[0]}, &gy{gyro[1]}, &gz{gyro[2]};
    float &ax{accel[0]}, &ay{accel[1]}, &az{accel[2]};
    float &mx{mag[0]}, &my{mag[1]}, &mz{mag[2]};
    float &q0{q[0]}, &q1{q[1]}, &q2{q[2]}, &q3{q[3]};

    // Use IMU algorithm if magnetometer measurement invalid (avoids NaN in magnetometer normalisation)
    if ((mx == 0.0f) && (my == 0.0f) && (mz == 0.0f))
    {
      Update(gyro, accel, dt);
      return;
    }

    // Rate of change of quaternion from gyroscope
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {

      // Normalise accelerometer measurement
      recipNorm = invSqrt(ax * ax + ay * ay + az * az);
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      // Normalise magnetometer measurement
      recipNorm = invSqrt(mx * mx + my * my + mz * mz);
      mx *= recipNorm;
      my *= recipNorm;
      mz *= recipNorm;

      // Auxiliary variables to avoid repeated arithmetic
      _2q0mx = 2.0f * q0 * mx;
      _2q0my = 2.0f * q0 * my;
      _2q0mz = 2.0f * q0 * mz;
      _2q1mx = 2.0f * q1 * mx;
      _2q0   = 2.0f * q0;
      _2q1   = 2.0f * q1;
      _2q2   = 2.0f * q2;
      _2q3   = 2.0f * q3;
      _2q0q2 = 2.0f * q0 * q2;
      _2q2q3 = 2.0f * q2 * q3;
      q0q0   = q0 * q0;
      q0q1   = q0 * q1;
      q0q2   = q0 * q2;
      q0q3   = q0 * q3;
      q1q1   = q1 * q1;
      q1q2   = q1 * q2;
      q1q3   = q1 * q3;
      q2q2   = q2 * q2;
      q2q3   = q2 * q3;
      q3q3   = q3 * q3;

      // Reference direction of Earth's magnetic field
      hx   = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2
             + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
      hy   = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1
             + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
      _2bx = sqrt(hx * hx + hy * hy);
      _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1
             + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
      _4bx = 2.0f * _2bx;
      _4bz = 2.0f * _2bz;

      // Gradient decent algorithm corrective step
      s0 = -_2q2 * (2.0f * q1q3 - _2q0q2 - ax)
           + _2q1 * (2.0f * q0q1 + _2q2q3 - ay)
           - _2bz * q2
                 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
           + (-_2bx * q3 + _2bz * q1)
                 * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
           + _2bx * q2
                 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
      s1 = _2q3 * (2.0f * q1q3 - _2q0q2 - ax)
           + _2q0 * (2.0f * q0q1 + _2q2q3 - ay)
           - 4.0f * q1 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az)
           + _2bz * q3
                 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
           + (_2bx * q2 + _2bz * q0)
                 * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
           + (_2bx * q3 - _4bz * q1)
                 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
      s2 = -_2q0 * (2.0f * q1q3 - _2q0q2 - ax)
           + _2q3 * (2.0f * q0q1 + _2q2q3 - ay)
           - 4.0f * q2 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az)
           + (-_4bx * q2 - _2bz * q0)
                 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
           + (_2bx * q1 + _2bz * q3)
                 * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
           + (_2bx * q0 - _4bz * q2)
                 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
      s3 = _2q1 * (2.0f * q1q3 - _2q0q2 - ax)
           + _2q2 * (2.0f * q0q1 + _2q2q3 - ay)
           + (-_4bx * q3 + _2bz * q1)
                 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
           + (-_2bx * q0 + _2bz * q2)
                 * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
           + _2bx * q1
                 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
      recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2
                          + s3 * s3); // normalise step magnitude
      s0 *= recipNorm;
      s1 *= recipNorm;
      s2 *= recipNorm;
      s3 *= recipNorm;

      // Apply feedback step
      qDot1 -= beta * s0;
      qDot2 -= beta * s1;
      qDot3 -= beta * s2;
      qDot4 -= beta * s3;
    }

    // Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
  }

  cv::Vec4f GetQuaternion() const override
  {
    return q;
  }
};

#endif /* AHRS_HPP */
