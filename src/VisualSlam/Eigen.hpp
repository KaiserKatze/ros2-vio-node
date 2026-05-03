#pragma once

#include <Eigen/Dense>

namespace Eigen
{
using Vector6d    = Eigen::Matrix<double, 6, 1>;
using RowVector6d = Eigen::Matrix<double, 1, 6>;
using MatrixX6d   = Eigen::Matrix<double, Eigen::Dynamic, 6>;
using Vector9d    = Eigen::Matrix<double, 9, 1>;
using MatrixX9d   = Eigen::Matrix<double, Eigen::Dynamic, 9>;
using Matrix34d   = Eigen::Matrix<double, 3, 4>;
} // namespace Eigen
