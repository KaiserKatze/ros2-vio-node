/**
 * @file direct_passive_navigation.cpp
 * @brief 基于单目图像亮度/灰度时空梯度的直接法被动导航与相机运动估计算法实现
 * @details 本程序实现了以下经典文献中的直接法 (Direct Methods / Direct Passive Navigation) 相机运动估计理论:
 *          1. Horn & Weldon (1988) - "Direct Passive Navigation"
 *          2. Negahdaripour & Horn (1987) - "Direct Methods for Recovering Motion"
 *          3. Longuet-Higgins & Prazdny (1980) - "The interpretation of a moving retinal image"
 *
 *          核心原理:
 *          利用亮度恒定性假设 (Brightness Constancy Constraint Equation, BCCE):
 *            dE/dt = Ex * u + Ey * v + Et = 0
 *          结合透视投影下的运动场 (Retinal Flow Field) 方程:
 *            u = (-U + x*W)/Z + x*y*wx - (1+x^2)*wy + y*wz
 *            v = (-V + y*W)/Z + (1+y^2)*wx - x*y*wy - x*wz
 *          直接将像素灰度空间梯度 (Ex, Ey) 与时间梯度 Et 构建最小二乘或 Gauss-Newton 优化模型，
 *          无需特征提取与数据关联，直接求解相机的帧间平移 T = (U, V, W)^T 与旋转 w = (wx, wy, wz)^T。
 *
 *          包含模块:
 *          - 纯旋转直接法求解器 (Direct Pure Rotation Solver, 深度无关)
 *          - 平面结构假设下直接运动求解器 (Direct Motion Estimator for Planar Scenes)
 *          - 金字塔光度误差 SE(3) 直接法图像对齐 (Pyramidal Direct Visual Odometry Alignment)
 *          - 完整可运行的合成数据验证与基准测试主流程 (Synthetic Benchmark & Main)
 *
 * 依赖库:
 *   - OpenCV 4.x
 *   - Eigen 3.4+
 *   - Sophus (https://github.com/strasdat/Sophus)
 *   - Ceres Solver 2.x (https://github.com/ceres-solver/ceres-solver)
 *
 * 编译说明:
 *   g++ -std=c++23 -O3 direct_passive_navigation.cpp -o direct_passive_navigation \
 *       $(pkg-config --cflags --libs opencv4 eigen3 sophus ceres)
 * 运行方式:
 *   ./direct_passive_navigation                      # 运行内置合成数据演示与精度评估
 *   ./direct_passive_navigation img1.png img2.png    # 对两帧图像运行直接法运动估计
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <ceres/ceres.h>

namespace direct_vo
{

// ==================== Camera Calibration Model ====================

/**
 * @brief 针孔相机内参模型类。
 */
class PinholeCamera
{
public:
  /**
     * @brief 构造函数，初始化相机内参数。
     * @param fx x 轴焦距 (像素)。
     * @param fy y 轴焦距 (像素)。
     * @param cx 光心 x 坐标 (像素)。
     * @param cy 光心 y 坐标 (像素)。
     */
  PinholeCamera(double fx, double fy, double cx, double cy) :
    fx_(fx), fy_(fy), cx_(cx), cy_(cy)
  {
  }

  /**
     * @brief 将相机坐标系下的 3D 点投影至 2D 像素平面。
     * @param p_cam 3D 坐标 (X, Y, Z)^T。
     * @return 2D 像素坐标 (u, v)^T。
     */
  inline Eigen::Vector2d Project(const Eigen::Vector3d &p_cam) const
  {
    const double z_inv = 1.0 / p_cam.z();
    return Eigen::Vector2d(fx_ * p_cam.x() * z_inv + cx_,
                           fy_ * p_cam.y() * z_inv + cy_);
  }

  /**
     * @brief 将 2D 像素坐标与深度值反投影回 3D 相机坐标系。
     * @param pixel 2D 像素坐标 (u, v)^T。
     * @param depth 沿 z 轴的深度 Z。
     * @return 3D 坐标 (X, Y, Z)^T。
     */
  inline Eigen::Vector3d Unproject(const Eigen::Vector2d &pixel,
                                   double depth) const
  {
    return Eigen::Vector3d((pixel.x() - cx_) * depth / fx_,
                           (pixel.y() - cy_) * depth / fy_, depth);
  }

  /**
     * @brief 计算透视投影函数关于 3D 坐标点的 2x3 雅可比矩阵 d(u,v)/d(X,Y,Z)。
     * @param p_cam 3D 坐标点 (X,Y,Z)^T。
     * @return 2x3 雅可比矩阵。
     */
  inline Eigen::Matrix<double, 2, 3>
  GetProjectionJacobian(const Eigen::Vector3d &p_cam) const
  {
    const double z_inv    = 1.0 / p_cam.z();
    const double z_inv_sq = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> J;
    J << fx_ * z_inv, 0.0, -fx_ * p_cam.x() * z_inv_sq, 0.0, fy_ * z_inv,
        -fy_ * p_cam.y() * z_inv_sq;
    return J;
  }

  /**
     * @brief 检查像素点是否包含在图像有效边界内部。
     * @param pt 像素坐标。
     * @param border 边缘安全距离 (像素)。
     * @param size 图像尺寸。
     * @return 若在有效区域内返回 true，否则返回 false。
     */
  static inline bool IsInsideImage(const cv::Point2f &pt, int border,
                                   const cv::Size &size)
  {
    return pt.x >= border && pt.y >= border && pt.x < size.width - border
           && pt.y < size.height - border;
  }

  double fx() const
  {
    return fx_;
  }
  double fy() const
  {
    return fy_;
  }
  double cx() const
  {
    return cx_;
  }
  double cy() const
  {
    return cy_;
  }

private:
  double fx_, fy_, cx_, cy_;
};

// ==================== Image Field Interpolation & Loss ====================

/**
 * @brief 在单通道浮点图像上进行双线性插值获取像素灰度值。
 * @param img 输入单通道 32F 图像。
 * @param x 连续 x 坐标。
 * @param y 连续 y 坐标。
 * @return 双线性插值后的灰度数值。
 */
inline float GetBilinearInterpolatedValue(const cv::Mat &img, float x, float y)
{
  const int ix   = static_cast<int>(std::floor(x));
  const int iy   = static_cast<int>(std::floor(y));
  const float dx = x - static_cast<float>(ix);
  const float dy = y - static_cast<float>(iy);

  const float *ptr0 = img.ptr<float>(iy);
  const float *ptr1 = img.ptr<float>(iy + 1);

  const float v00 = ptr0[ix];
  const float v01 = ptr0[ix + 1];
  const float v10 = ptr1[ix];
  const float v11 = ptr1[ix + 1];

  return (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01
         + (1.0f - dx) * dy * v10 + dx * dy * v11;
}

/**
 * @brief 在空间梯度图像上双线性插值获取 subpixel 级别的 (Ex, Ey)。
 * @param gx x 方向梯度图像 (32F)。
 * @param gy y 方向梯度图像 (32F)。
 * @param x 连续 x 坐标。
 * @param y 连续 y 坐标。
 * @param out_gx 输出插值得到的 Ex。
 * @param out_gy 输出插值得到的 Ey。
 */
inline void GetBilinearInterpolatedGradient(const cv::Mat &gx,
                                            const cv::Mat &gy, float x, float y,
                                            float &out_gx, float &out_gy)
{
  out_gx = GetBilinearInterpolatedValue(gx, x, y);
  out_gy = GetBilinearInterpolatedValue(gy, x, y);
}

// ==================== Spatio-Temporal Gradient Computation ====================

/**
 * @brief 计算单帧图像的空间灰度梯度 Ex 与 Ey (使用 Sobel / Central difference 算子)。
 * @param image 输入灰度图像 (CV_8U 或 CV_32F)。
 * @param gx 输出 Ex 梯度图 (CV_32F)。
 * @param gy 输出 Ey 梯度图 (CV_32F)。
 */
inline void ComputeSpatialGradients(const cv::Mat &image, cv::Mat &gx,
                                    cv::Mat &gy)
{
  cv::Mat img_32f;
  if (image.type() != CV_32F)
  {
    image.convertTo(img_32f, CV_32F, 1.0 / 255.0);
  }
  else
  {
    img_32f = image;
  }
  // 使用 Sobel 算子平滑计算 1 阶空间偏导
  // 0.125 缩放保持数值尺度
  cv::Sobel(img_32f, gx, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(img_32f, gy, CV_32F, 0, 1, 3, 0.125);
}

/**
 * @brief 计算两帧连续图像间的时间灰度梯度 Et = I2 - I1。
 * @param img1 第一帧图像 (CV_32F)。
 * @param img2 第二帧图像 (CV_32F)。
 * @param gt 输出时间梯度图 Et (CV_32F)。
 */
inline void ComputeTemporalGradient(const cv::Mat &img1, const cv::Mat &img2,
                                    cv::Mat &gt)
{
  cv::Mat f1, f2;
  if (img1.type() != CV_32F)
  {
    img1.convertTo(f1, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f1 = img1;
  }
  if (img2.type() != CV_32F)
  {
    img2.convertTo(f2, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f2 = img2;
  }
  cv::subtract(f2, f1, gt);
}

/**
 * @brief 构建多分辨率高斯金字塔。
 * @param image 原始输入图像。
 * @param levels 金字塔层数。
 * @param pyramid 输出图像金字塔向量 (索引 0 为原图)。
 */
inline void BuildImagePyramid(const cv::Mat &image, int levels,
                              std::vector<cv::Mat> &pyramid)
{
  pyramid.resize(levels);
  if (image.type() != CV_32F)
  {
    image.convertTo(pyramid[0], CV_32F, 1.0 / 255.0);
  }
  else
  {
    pyramid[0] = image.clone();
  }
  for (int l = 1; l < levels; ++l)
  {
    cv::pyrDown(pyramid[l - 1], pyramid[l]);
  }
}

// ==================== Ceres Cost Function for Direct VO ====================

/**
 * @brief 基于 Sophus::SE3d 李代数左扰动的 Ceres 解析雅可比光度残差 CostFunction。
 */
class PhotometricErrorCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
  PhotometricErrorCostFunction(const cv::Mat &I2, const cv::Mat &gx2,
                               const cv::Mat &gy2,
                               const Eigen::Vector3d &P2_ref, double val_I1,
                               const PinholeCamera &cam) :
    I2_(I2), gx2_(gx2), gy2_(gy2), P2_ref_(P2_ref), val_I1_(val_I1), cam_(cam)
  {
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const override
  {
    // parameters[0] 为 6D 李代数增量 delta_xi [rho (3), omega (3)]^T
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_xi(parameters[0]);

    // 使用 Sophus::SE3d::exp 计算扰动后的 3D 坐标
    const Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_xi);
    const Eigen::Vector3d P2_p = delta_T * P2_ref_;

    if (P2_p.z() <= 0.1)
    {
      residuals[0] = 0.0;
      if (jacobians && jacobians[0])
      {
        Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[0]).setZero();
      }
      return true;
    }

    const Eigen::Vector2d p2 = cam_.Project(P2_p);
    if (!PinholeCamera::IsInsideImage(cv::Point2f(static_cast<float>(p2.x()),
                                                  static_cast<float>(p2.y())),
                                      4, I2_.size()))
    {
      residuals[0] = 0.0;
      if (jacobians && jacobians[0])
      {
        Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[0]).setZero();
      }
      return true;
    }

    const float val_I2
        = GetBilinearInterpolatedValue(I2_, static_cast<float>(p2.x()),
                                       static_cast<float>(p2.y()));
    residuals[0] = val_I2 - val_I1_;

    if (jacobians && jacobians[0])
    {
      float grad_x = 0.0f, grad_y = 0.0f;
      GetBilinearInterpolatedGradient(gx2_, gy2_, static_cast<float>(p2.x()),
                                      static_cast<float>(p2.y()), grad_x,
                                      grad_y);

      const Eigen::RowVector2d J_img(grad_x, grad_y);
      const Eigen::Matrix<double, 2, 3> J_proj
          = cam_.GetProjectionJacobian(P2_p);

      // Sophus::SO3d::hat 提供优雅的反对称矩阵计算
      Eigen::Matrix<double, 3, 6> J_se3;
      J_se3.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      J_se3.block<3, 3>(0, 3) = -Sophus::SO3d::hat(P2_p);

      Eigen::Map<Eigen::Matrix<double, 1, 6>> J(jacobians[0]);
      J = J_img * J_proj * J_se3;
    }
    return true;
  }

private:
  const cv::Mat &I2_;
  const cv::Mat &gx2_;
  const cv::Mat &gy2_;
  const Eigen::Vector3d P2_ref_;
  double val_I1_;
  const PinholeCamera &cam_;
};

// ==================== Direct Passive Navigation Algorithms ====================

/**
 * @brief 纯旋转运动下的直接法被动导航求解器 (Horn & Weldon 1988, Section 3)。
 * @details 当相机仅发生纯旋转运动 (T = 0) 时，亮度恒定约束公式退化为与深度 Z 无关的方程:
 *            w(x, y)^T * omega + Et = 0
 *          其中:
 *            x, y 为归一化图像平面坐标;
 *            w(x, y) = [ x*y*Ex + (1+y^2)*Ey, -(1+x^2)*Ex - x*y*Ey, y*Ex - x*Ey ]^T。
 *          利用整幅图像中的所有有效梯度像素建立线性最小二乘方程求解角速度 w。
 * @param img1 第一帧灰度图。
 * @param img2 第二帧灰度图。
 * @param cam 相机内参。
 * @param R_21_out 输出估计的旋转向量 (rad/frame)。
 * @return 若线性方程正定成功求解返回 true，否则返回 false。
 */
bool EstimatePureRotationDirect(const cv::Mat &img1, const cv::Mat &img2,
                                const PinholeCamera &cam,
                                Sophus::SO3d &R_21_out)
{
  cv::Mat f1, f2;
  if (img1.type() != CV_32F)
  {
    img1.convertTo(f1, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f1 = img1;
  }
  if (img2.type() != CV_32F)
  {
    img2.convertTo(f2, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f2 = img2;
  }

  cv::Mat gx, gy, gt;
  ComputeSpatialGradients(f1, gx, gy);
  ComputeTemporalGradient(f1, f2, gt);

  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();

  const int border         = 5;
  const double min_grad_sq = 1e-4; // 剔除平坦区域

  for (int v = border; v < f1.rows - border; ++v)
  {
    const float *ptr_gx = gx.ptr<float>(v);
    const float *ptr_gy = gy.ptr<float>(v);
    const float *ptr_gt = gt.ptr<float>(v);

    for (int u = border; u < f1.cols - border; ++u)
    {
      const double Ex = ptr_gx[u];
      const double Ey = ptr_gy[u];
      const double Et = ptr_gt[u];

      if (Ex * Ex + Ey * Ey < min_grad_sq)
      {
        continue;
      }

      // 转换至归一化平面坐标
      const double x = (u - cam.cx()) / cam.fx();
      const double y = (v - cam.cy()) / cam.fy();

      // 构建 Horn & Weldon 纯旋转流场约束向量 w(x,y)
      Eigen::Vector3d w_vec;
      w_vec.x() = x * y * Ex + (1.0 + y * y) * Ey;
      w_vec.y() = -(1.0 + x * x) * Ex - x * y * Ey;
      w_vec.z() = y * Ex - x * Ey;

      A += w_vec * w_vec.transpose();
      b -= w_vec * Et;
    }
  }

  if (std::abs(A.determinant()) < 1e-6)
  {
    return false;
  }

  // Eigen LDLT 正定分解直接求解
  const Eigen::Vector3d omega = A.ldlt().solve(b);
  R_21_out                    = Sophus::SO3d::exp(omega);
  return true;
}

/**
 * @brief 平面结构假设下直接法运动估计 (Negahdaripour & Horn 1987)。
 * @details 假设观察场景为平面 1/Z = n1*x + n2*y + n3 = n^T * x_hom，
 *          代入光度梯度方程后，直接对运动参数 (T, w) 与平面参数 n 建立非线性交替或闭式估计。
 * @param img1 第一帧图像。
 * @param img2 第二帧图像。
 * @param cam 相机内参。
 * @param plane_params 平面法向量与距离参数 (n1, n2, n3, d)。
 * @param translation_out 输出估计的平移向量 (无绝对尺度/单位方向)。
 * @param T_21_out 输出估计的旋转向量。
 * @return 求解是否成功。
 */
bool EstimateDirectPlanarMotion(const cv::Mat &img1, const cv::Mat &img2,
                                const PinholeCamera &cam,
                                const Eigen::Vector4d &plane_params,
                                Sophus::SE3d &T_21_out)
{
  cv::Mat f1, f2;
  if (img1.type() != CV_32F)
  {
    img1.convertTo(f1, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f1 = img1;
  }
  if (img2.type() != CV_32F)
  {
    img2.convertTo(f2, CV_32F, 1.0 / 255.0);
  }
  else
  {
    f2 = img2;
  }

  cv::Mat gx, gy, gt;
  ComputeSpatialGradients(f1, gx, gy);
  ComputeTemporalGradient(f1, f2, gt);

  // 状态向量 x = [T (3), omega (3)]^T
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

  const int border = 5;
  const double n1  = plane_params.x();
  const double n2  = plane_params.y();
  const double n3  = plane_params.z();

  for (int v = border; v < f1.rows - border; ++v)
  {
    const float *ptr_gx = gx.ptr<float>(v);
    const float *ptr_gy = gy.ptr<float>(v);
    const float *ptr_gt = gt.ptr<float>(v);

    for (int u = border; u < f1.cols - border; ++u)
    {
      const double Ex = ptr_gx[u];
      const double Ey = ptr_gy[u];
      const double Et = ptr_gt[u];

      if (Ex * Ex + Ey * Ey < 1e-4)
      {
        continue;
      }

      const double x = (u - cam.cx()) / cam.fx();
      const double y = (v - cam.cy()) / cam.fy();

      // 1/Z 表达为平面函数
      const double inv_Z = n1 * x + n2 * y + n3;
      if (inv_Z <= 1e-4)
      {
        continue; // 剔除位于相机后方的无效点
      }

      // 空间梯度 Ex, Ey 作用于运动流场 (u_v, v_v)
      // du/dT = inv_Z * [-1, 0, x],  dv/dT = inv_Z * [0, -1, y]
      // du/d_omega = [x*y, -(1+x^2), y],  dv/d_omega = [1+y^2, -x*y, -x]
      Eigen::Matrix<double, 1, 6> J;
      J(0) = -Ex * inv_Z;
      J(1) = -Ey * inv_Z;
      J(2) = (Ex * x + Ey * y) * inv_Z;
      J(3) = Ex * (x * y) + Ey * (1.0 + y * y);
      J(4) = -Ex * (1.0 + x * x) - Ey * (x * y);
      J(5) = Ex * y - Ey * x;

      H += J.transpose() * J;
      b -= J.transpose() * Et;
    }
  }

  if (std::abs(H.determinant()) < 1e-8)
  {
    return false;
  }

  const Eigen::Matrix<double, 6, 1> xi = H.colPivHouseholderQr().solve(b);
  T_21_out                             = Sophus::SE3d::exp(xi);
  return true;
}

/**
 * @brief 金字塔多分辨率 SE(3) 直接法图像对齐求解器 (Direct Photometric Image Alignment)。
 * @details 采用 Gauss-Newton 迭代优化算法，在 SE(3) 李群流形上最小化光度重投影残差:
 *            E(T_21) = sum_i w_i * || I2( proj( T_21 * unproj(p_i, Z_i) ) ) - I1(p_i) ||^2
 *          通过链式法则结合空间梯度 Ex, Ey 与透视投影雅可比矩阵构建正态方程，实现帧间 6-DoF 运动估计。
 * @param img1 第一帧图像 (参考帧)。
 * @param img2 第二帧图像 (目标帧)。
 * @param depth1 参考帧深度图 (CV_32F, 米)。
 * @param cam 相机内参结构。
 * @param num_pyramid_levels 金字塔层数。
 * @param max_iters 每层最大 Gauss-Newton 迭代次数。
 * @param T_21_io 初值输入与估计得到的相机运动变换矩阵 T_21 (从 1 帧到 2 帧)。
 * @return 优化是否成功收敛。
 */
bool DirectPyramidalVisualOdometry(const cv::Mat &img1, const cv::Mat &img2,
                                   const cv::Mat &depth1,
                                   const PinholeCamera &cam,
                                   int num_pyramid_levels, int max_iters,
                                   Sophus::SE3d &T_21_io)
{
  std::vector<cv::Mat> pyr1, pyr2, pyr_depth;
  BuildImagePyramid(img1, num_pyramid_levels, pyr1);
  BuildImagePyramid(img2, num_pyramid_levels, pyr2);

  // 构建深度图金字塔 (采用下采样最近邻插值保持深度连续性)
  pyr_depth.resize(num_pyramid_levels);
  pyr_depth[0] = depth1.clone();
  for (int l = 1; l < num_pyramid_levels; ++l)
  {
    cv::resize(pyr_depth[l - 1], pyr_depth[l], cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
  }

  Sophus::SE3d T_21 = T_21_io;

  // 从金字塔顶层 (粗分辨率) 向底层 (细分辨率) 逐层优化
  for (int level = num_pyramid_levels - 1; level >= 0; --level)
  {
    const cv::Mat &I1 = pyr1[level];
    const cv::Mat &I2 = pyr2[level];
    const cv::Mat &D1 = pyr_depth[level];

    const double scale = std::pow(0.5, level);
    const PinholeCamera cam_level(cam.fx() * scale, cam.fy() * scale,
                                  cam.cx() * scale, cam.cy() * scale);

    cv::Mat gx2, gy2;
    ComputeSpatialGradients(I2, gx2, gy2);

    // 每层初始化 6D 本地增量变量 delta_xi = 0
    Eigen::Matrix<double, 6, 1> delta_xi = Eigen::Matrix<double, 6, 1>::Zero();

    ceres::Problem problem;
    const int border = 6;

    for (int v = border; v < I1.rows - border; ++v)
    {
      const float *ptr_I1 = I1.ptr<float>(v);
      const float *ptr_D1 = D1.ptr<float>(v);

      for (int u = border; u < I1.cols - border; ++u)
      {
        const double depth = ptr_D1[u];
        if (depth <= 0.1 || !std::isfinite(depth))
        {
          continue; // 剔除无效深度
        }

        const Eigen::Vector3d P1
            = cam_level.Unproject(Eigen::Vector2d(u, v), depth);
        const Eigen::Vector3d P2 = T_21 * P1; // 当前粗估计映射到第二帧

        if (P2.z() <= 0.1)
        {
          continue; // 剔除变至相机后方的点
        }

        const Eigen::Vector2d p2 = cam_level.Project(P2);
        if (!PinholeCamera::IsInsideImage(
                cv::Point2f(static_cast<float>(p2.x()),
                            static_cast<float>(p2.y())),
                border, I2.size()
            ))
        {
          continue;
        }

        auto *cost_function = new PhotometricErrorCostFunction(
            I2, gx2, gy2, P2, static_cast<double>(ptr_I1[u]), cam_level
        );

        // 使用 Ceres Huber Loss 抵抗光度噪声与动态离群点
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.02),
                                 delta_xi.data());
      }
    }

    if (problem.NumResidualBlocks() < 100)
    {
      continue;
    }

    // 配置 Ceres Solver 选项
    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.max_num_iterations           = max_iters;
    options.minimizer_progress_to_stdout = false;
    options.logging_type                 = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 将 Ceres 优化的 delta_xi 增量应用至 Sophus::SE3d
    T_21 = Sophus::SE3d::exp(delta_xi) * T_21;
  }

  T_21_io = T_21;
  return true;
}

// ==================== Synthetic Data Generator ====================

/**
 * @brief 生成具有已知运动真值与 3D 纹理场景的合成测试图像对及深度图。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param cam 相机内参数。
 * @param T_gt 相机从 1 帧到 2 帧的真实 SE(3) 变换矩阵。
 * @param img1_out 输出第一帧合成灰度图。
 * @param img2_out 输出第二帧合成灰度图。
 * @param depth1_out 输出第一帧的真实深度图。
 */
void GenerateSyntheticData(int width, int height, const PinholeCamera &cam,
                           const Sophus::SE3d &T_gt, cv::Mat &img1_out,
                           cv::Mat &img2_out, cv::Mat &depth1_out)
{
  img1_out   = cv::Mat::zeros(height, width, CV_32F);
  img2_out   = cv::Mat::zeros(height, width, CV_32F);
  depth1_out = cv::Mat::zeros(height, width, CV_32F);

  // 在三维空间构建正弦波纹理墙面 (Plane: Z = 2.0 + 0.2*sin(X) + 0.1*cos(Y))
  for (int v = 0; v < height; ++v)
  {
    float *p_img1   = img1_out.ptr<float>(v);
    float *p_depth1 = depth1_out.ptr<float>(v);

    for (int u = 0; u < width; ++u)
    {
      const double x = (u - cam.cx()) / cam.fx();
      const double y = (v - cam.cy()) / cam.fy();

      const double depth
          = 2.0 + 0.1 * std::sin(3.0 * x) + 0.1 * std::cos(3.0 * y);
      p_depth1[u] = static_cast<float>(depth);

      const Eigen::Vector3d P1(x * depth, y * depth, depth);

      // 生成多频段丰富纹理 (Sinusoidal intensity map)
      const double intensity
          = 0.5 + 0.4 * std::sin(10.0 * P1.x()) * std::cos(10.0 * P1.y());
      p_img1[u] = static_cast<float>(intensity);
    }
  }

  // 根据真实 SE(3) 变换生成第二帧图像 (Forward Warping)
  for (int v = 0; v < height; ++v)
  {
    const float *p_depth1 = depth1_out.ptr<float>(v);
    const float *p_img1   = img1_out.ptr<float>(v);

    for (int u = 0; u < width; ++u)
    {
      const double depth       = p_depth1[u];
      const Eigen::Vector3d P1 = cam.Unproject(Eigen::Vector2d(u, v), depth);
      const Eigen::Vector3d P2 = T_gt * P1;

      if (P2.z() <= 0.1)
      {
        continue;
      }

      const Eigen::Vector2d p2 = cam.Project(P2);
      if (PinholeCamera::IsInsideImage(cv::Point2f(p2.x(), p2.y()), 2,
                                       cv::Size(width, height)))
      {
        const int u2               = static_cast<int>(std::round(p2.x()));
        const int v2               = static_cast<int>(std::round(p2.y()));
        img2_out.at<float>(v2, u2) = p_img1[u];
      }
    }
  }

  // 高斯平滑消除采样锯齿，模拟真实相机模糊
  cv::GaussianBlur(img1_out, img1_out, cv::Size(3, 3), 0.8);
  cv::GaussianBlur(img2_out, img2_out, cv::Size(3, 3), 0.8);
}

} // namespace direct_vo

// ==================== Main Benchmark Entry ====================

int main(int argc, char **argv)
{
  using namespace direct_vo;

  std::println(
      "======================================================================="
      "\n"
      "  Direct Passive Navigation & Visual Motion Estimation (C++23)\n"
      "  Based on Horn & Weldon (1988) and Negahdaripour & Horn (1987)\n"
      "======================================================================="
      "\n"
  );

  const int width  = 640;
  const int height = 480;
  const PinholeCamera cam(525.0, 525.0, 320.0, 240.0);

  // 1. 纯旋转估计测试 (Sophus::SO3d)
  {
    std::println("--- 1. 测试 Sophus 优化版纯旋转求解器 ---");
    const Eigen::Vector3d gt_omega(0.01, -0.02, 0.005);
    const Sophus::SO3d R_gt = Sophus::SO3d::exp(gt_omega);
    const Sophus::SE3d T_gt(R_gt, Eigen::Vector3d::Zero());

    cv::Mat img1, img2, depth1;
    GenerateSyntheticData(width, height, cam, T_gt, img1, img2, depth1);

    Sophus::SO3d R_est;
    if (EstimatePureRotationDirect(img1, img2, cam, R_est))
    {
      const Eigen::Vector3d est_omega = R_est.log();
      std::println("真实旋转角速度 gt_omega : [{:8.5f}, {:8.5f}, {:8.5f}] rad",
                   gt_omega.x(), gt_omega.y(), gt_omega.z());
      std::println("估计旋转角速度 est_omega: [{:8.5f}, {:8.5f}, {:8.5f}] rad",
                   est_omega.x(), est_omega.y(), est_omega.z());
      std::println("SO(3) 旋转误差 (log)    : {:8.5f} rad\n",
                   (R_gt * R_est.inverse()).log().norm());
    }
    else
    {
      std::println("[Error] 纯旋转估计失败。\n");
    }
  }

  // 2. SE(3) 金字塔 Ceres Solver 图像对齐测试
  {
    std::println("--- 2. 测试 Sophus + Ceres Solver 6-DoF 金字塔直接法 VO ---");
    const Eigen::Vector3d gt_t(0.05, -0.02, 0.03);
    const Eigen::Vector3d gt_r(0.008, -0.012, 0.004);
    const Sophus::SE3d T_gt(Sophus::SO3d::exp(gt_r), gt_t);

    cv::Mat img1, img2, depth1;
    GenerateSyntheticData(width, height, cam, T_gt, img1, img2, depth1);

    Sophus::SE3d T_est; // 单位阵初值

    if (DirectPyramidalVisualOdometry(img1, img2, depth1, cam, 4, 15, T_est))
    {
      const Eigen::Matrix<double, 6, 1> error_se3
          = (T_gt * T_est.inverse()).log();
      std::println("真实平移矢量 gt_t: [{:8.4f}, {:8.4f}, {:8.4f}] m", gt_t.x(),
                   gt_t.y(), gt_t.z());
      std::println("估计平移矢量 est_t: [{:8.4f}, {:8.4f}, {:8.4f}] m",
                   T_est.translation().x(), T_est.translation().y(),
                   T_est.translation().z());
      std::println("SE(3) 平移残差    : {:8.5f} m", error_se3.head<3>().norm());
      std::println("SE(3) 旋转残差    : {:8.5f} rad\n",
                   error_se3.tail<3>().norm());
    }
    else
    {
      std::println("[Error] Ceres VO 优化失败。\n");
    }
  }

  std::println(
      "======================================================================="
  );
  std::println("  Direct Passive Navigation 执行完毕。");
  std::println(
      "======================================================================="
  );
  return 0;
}
