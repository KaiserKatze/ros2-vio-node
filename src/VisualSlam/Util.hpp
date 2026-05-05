#pragma once

#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/types.hpp>

struct Util
{
  /**
 * @brief 使用 Eigen 原生 Map 进行高性能转换（极力推荐）
 *
 * OpenCV 的 cv::Point2f 在内存中是连续的[x, y, x, y...] 结构。
 * Eigen 默认是列主序(Column-Major)，其 2xN 矩阵在内存中也是 [row0_col0, row1_col0, ...]
 * 两者的内存布局完全等价，可以直接进行零拷贝映射！
 */
  static Eigen::Matrix3Xd
  ConvertToMatrix3Xd_EigenMap(const std::vector<cv::Point2f> &points)
  {
    if (points.empty())
    {
      return Eigen::Matrix3Xd(3, 0);
    }

    // 1. 零拷贝映射为 Eigen::Matrix2Xf (2行 N列的 float 矩阵)
    Eigen::Map<const Eigen::Matrix2Xf> mapped_pts(
        reinterpret_cast<const float *>(points.data()), 2, points.size());

    // 2. 初始化目标矩阵
    Eigen::Matrix3Xd mat(3, points.size());

    // 3. 利用 Eigen 内部的 SIMD 向量化指令批量执行类型转换和赋值
    mat.topRows<2>() = mapped_pts.cast<double>();
    mat.bottomRows<1>().setOnes(); // 填充齐次坐标 1.0

    return mat;
  }
};
