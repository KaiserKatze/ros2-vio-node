#pragma once

#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

struct Util
{
  /**
   * @brief 打印 cv::Mat 对象的数据量、通道数、形状等信息
   */
  static std::string FormatCvMatInfo(std::string_view mat_name,
                                     const cv::Mat &mat)
  {
    const auto mat_type{mat.type()};
    const auto mat_depth{mat.depth()};
    return std::format("\t{} = {{ "
                       "data = {}, "
                       "total = {}, "
                       "dim = {}, "
                       "shape = ({}, {}), "
                       "continuous = {}, "
                       "channel = {}, "
                       "depth = {} ({}), "
                       "type = {} ({}) }}\n",
                       mat_name,                    //
                       static_cast<bool>(mat.data), //
                       mat.total(),                 //
                       mat.dims,                    //
                       mat.rows,
                       mat.cols,                               //
                       mat.isContinuous(),                     //
                       mat.channels(),                         //
                       cv::typeToString(mat_depth), mat_depth, //
                       cv::typeToString(mat_type), mat_type);
  }

  /**
   * @brief 使用 Eigen 原生 Map 进行高性能转换（极力推荐）
   *
   * OpenCV 的 PointType 在内存中是连续的[x, y, x, y...] 结构。
   * Eigen 默认是列主序(Column-Major)，其 2xN 矩阵在内存中也是 [row0_col0, row1_col0, ...]
   * 两者的内存布局完全等价，可以直接进行零拷贝映射！
   */
  template <typename PointType = cv::Point2f>
  static Eigen::Matrix<typename PointType::value_type, 3, Eigen::Dynamic>
  ConvertToMatrix3X_EigenMap(const std::vector<PointType> &points)
  {
    if (points.empty())
    {
      return Eigen::Matrix<typename PointType::value_type, 3, Eigen::Dynamic>(
          3, 0
      );
    }

    // 1. 零拷贝映射为 Eigen::Matrix2Xf (2行 N列的 floating_point 矩阵)
    Eigen::Map<const Eigen::Matrix<typename PointType::value_type, 2,
                                   Eigen::Dynamic>>
        mapped_pts(reinterpret_cast<const typename PointType::value_type *>(
                       points.data()
                   ),
                   2, points.size());

    // 2. 初始化目标矩阵
    Eigen::Matrix<typename PointType::value_type, 3, Eigen::Dynamic> mat{
        3, points.size()
    };

    // 3. 利用 Eigen 内部的 SIMD 向量化指令批量执行类型转换和赋值
    mat.template topRows<2>()
        = mapped_pts.template cast<typename PointType::value_type>();
    mat.template bottomRows<1>().setOnes(); // 填充齐次坐标 1.0

    return mat;
  }
};
