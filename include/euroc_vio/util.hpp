#ifndef UTIL_HPP
#define UTIL_HPP

#include <Eigen/Dense>

struct Util
{
  /**
   * @brief 用向量 rVec 对应的单位化向量表示旋转轴 $a$，用 rVec 的范数表示旋转角度 $\phi$
   * @note 旋转矩阵为
   *    $ C_{21} = \cos\phi I + (1 - \cos\phi) a a^T - \sin\phi a^\times $,
   * 其中 $a^\times$ 表示单位向量 $a = (a_1,a_2,a_3)^T$ 对应的叉乘矩阵
   *    $$
   *        \begin{bmatrix}
   *             0 & -a_3 & a_2 \\
   *             a_3 & 0 & -a_1 \\
   *             -a_2 & a_1 & 0 \\
   *        \end{bmatrix}
   *    $$
   */
  static Eigen::Matrix3d Rodrigues(const Eigen::Vector3d &rVec)
  {
    double angle{rVec.norm()};
    if (angle < 1e-8)
    {
      return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rVec.normalized()).toRotationMatrix();
  }

  static Eigen::Vector3d InvRodrigues(const Eigen::Matrix3d &R)
  {
    Eigen::AngleAxisd angleAxis(R);
    return angleAxis.angle() * angleAxis.axis();
  }

  static Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d skew;
    double x{v.x()}, y{v.y()}, z{v.z()};
    skew << 0, -z, y, z, 0, -x, -y, x, 0;
    return skew;
  }
};

#endif /* UTIL_HPP */
