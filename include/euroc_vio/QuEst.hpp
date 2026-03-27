#ifndef QUEST_HPP
#define QUEST_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <armadillo>

using VecIndex = std::vector<size_t>;

struct ResultType
{
  struct
  {
    double x;
    double y;
    double z;
  } translation;
  struct
  {
    double w;
    double x;
    double y;
    double z;
  } rotation;
};

void coefsNum(Eigen::MatrixXd &coefsN, const Eigen::VectorXd &mx1,
              const Eigen::VectorXd &mx2, const Eigen::VectorXd &my1,
              const Eigen::VectorXd &my2, const Eigen::VectorXd &nx2,
              const Eigen::VectorXd &ny2, const Eigen::VectorXd &r2,
              const Eigen::VectorXd &s1, const Eigen::VectorXd &s2)
{
  auto numPts{mx1.rows()};

  Eigen::VectorXd t2(numPts, 1);
  Eigen::VectorXd t3(numPts, 1);
  Eigen::VectorXd t4(numPts, 1);
  Eigen::VectorXd t5(numPts, 1);
  Eigen::VectorXd t6(numPts, 1);
  Eigen::VectorXd t7(numPts, 1);
  Eigen::VectorXd t8(numPts, 1);
  Eigen::VectorXd t9(numPts, 1);
  Eigen::VectorXd t10(numPts, 1);
  Eigen::VectorXd t11(numPts, 1);
  Eigen::VectorXd t12(numPts, 1);
  Eigen::VectorXd t13(numPts, 1);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    t2(i, 0)  = mx1(i, 0) * my2(i, 0) * r2(i, 0);
    t3(i, 0)  = mx2(i, 0) * ny2(i, 0) * s1(i, 0);
    t4(i, 0)  = my1(i, 0) * nx2(i, 0) * s2(i, 0);
    t5(i, 0)  = mx1(i, 0) * nx2(i, 0) * s2(i, 0) * 2.0;
    t6(i, 0)  = my1(i, 0) * ny2(i, 0) * s2(i, 0) * 2.0;
    t7(i, 0)  = mx1(i, 0) * my2(i, 0) * nx2(i, 0) * 2.0;
    t8(i, 0)  = my2(i, 0) * r2(i, 0) * s1(i, 0) * 2.0;
    t9(i, 0)  = mx2(i, 0) * my1(i, 0) * r2(i, 0);
    t10(i, 0) = mx1(i, 0) * ny2(i, 0) * s2(i, 0);
    t11(i, 0) = mx2(i, 0) * my1(i, 0) * ny2(i, 0) * 2.0;
    t12(i, 0) = mx2(i, 0) * r2(i, 0) * s1(i, 0) * 2.0;
    t13(i, 0) = my2(i, 0) * nx2(i, 0) * s1(i, 0);

    coefsN(i, 0)
        = t2(i, 0) + t3(i, 0) + t4(i, 0) - mx2(i, 0) * my1(i, 0) * r2(i, 0)
          - mx1(i, 0) * ny2(i, 0) * s2(i, 0) - my2(i, 0) * nx2(i, 0) * s1(i, 0);
    coefsN(i, 1) = t11(i, 0) + t12(i, 0)
                   - mx1(i, 0) * my2(i, 0) * ny2(i, 0) * 2.0
                   - mx1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    coefsN(i, 2) = t7(i, 0) + t8(i, 0) - mx2(i, 0) * my1(i, 0) * nx2(i, 0) * 2.0
                   - my1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    coefsN(i, 3) = t5(i, 0) + t6(i, 0) - mx2(i, 0) * nx2(i, 0) * s1(i, 0) * 2.0
                   - my2(i, 0) * ny2(i, 0) * s1(i, 0) * 2.0;
    coefsN(i, 4) = -t2(i, 0) - t3(i, 0) + t4(i, 0) + t9(i, 0) + t10(i, 0)
                   - my2(i, 0) * nx2(i, 0) * s1(i, 0);
    coefsN(i, 5) = -t5(i, 0) + t6(i, 0) + mx2(i, 0) * nx2(i, 0) * s1(i, 0) * 2.0
                   - my2(i, 0) * ny2(i, 0) * s1(i, 0) * 2.0;
    coefsN(i, 6) = t7(i, 0) - t8(i, 0) - mx2(i, 0) * my1(i, 0) * nx2(i, 0) * 2.0
                   + my1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    coefsN(i, 7)
        = -t2(i, 0) + t3(i, 0) - t4(i, 0) + t9(i, 0) - t10(i, 0) + t13(i, 0);
    coefsN(i, 8) = -t11(i, 0) + t12(i, 0)
                   + mx1(i, 0) * my2(i, 0) * ny2(i, 0) * 2.0
                   - mx1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    coefsN(i, 9)
        = t2(i, 0) - t3(i, 0) - t4(i, 0) - t9(i, 0) + t10(i, 0) + t13(i, 0);
  }
}

void coefsDen(Eigen::MatrixXd &coefsD, const Eigen::VectorXd &mx2,
              const Eigen::VectorXd &my2, const Eigen::VectorXd &nx1,
              const Eigen::VectorXd &nx2, const Eigen::VectorXd &ny1,
              const Eigen::VectorXd &ny2, const Eigen::VectorXd &r1,
              const Eigen::VectorXd &r2, const Eigen::VectorXd &s2)
{
  auto numPts{mx2.rows()};

  Eigen::VectorXd t2_D(numPts, 1);
  Eigen::VectorXd t3_D(numPts, 1);
  Eigen::VectorXd t4_D(numPts, 1);
  Eigen::VectorXd t5_D(numPts, 1);
  Eigen::VectorXd t6_D(numPts, 1);
  Eigen::VectorXd t7_D(numPts, 1);
  Eigen::VectorXd t8_D(numPts, 1);
  Eigen::VectorXd t9_D(numPts, 1);
  Eigen::VectorXd t10_D(numPts, 1);
  Eigen::VectorXd t11_D(numPts, 1);
  Eigen::VectorXd t12_D(numPts, 1);
  Eigen::VectorXd t13_D(numPts, 1);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    t2_D(i, 0)  = mx2(i, 0) * ny1(i, 0) * r2(i, 0);
    t3_D(i, 0)  = my2(i, 0) * nx2(i, 0) * r1(i, 0);
    t4_D(i, 0)  = nx1(i, 0) * ny2(i, 0) * s2(i, 0);
    t5_D(i, 0)  = mx2(i, 0) * nx2(i, 0) * r1(i, 0) * 2.0;
    t6_D(i, 0)  = my2(i, 0) * ny2(i, 0) * r1(i, 0) * 2.0;
    t7_D(i, 0)  = mx2(i, 0) * nx2(i, 0) * ny1(i, 0) * 2.0;
    t8_D(i, 0)  = ny1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    t9_D(i, 0)  = my2(i, 0) * nx1(i, 0) * r2(i, 0);
    t10_D(i, 0) = nx2(i, 0) * ny1(i, 0) * s2(i, 0);
    t11_D(i, 0) = my2(i, 0) * nx1(i, 0) * ny2(i, 0) * 2.0;
    t12_D(i, 0) = nx1(i, 0) * r2(i, 0) * s2(i, 0) * 2.0;
    t13_D(i, 0) = mx2(i, 0) * ny2(i, 0) * r1(i, 0);

    coefsD(i, 0) = t2_D(i, 0) + t3_D(i, 0) + t4_D(i, 0)
                   - mx2(i, 0) * ny2(i, 0) * r1(i, 0)
                   - my2(i, 0) * nx1(i, 0) * r2(i, 0)
                   - nx2(i, 0) * ny1(i, 0) * s2(i, 0);
    coefsD(i, 1) = t11_D(i, 0) + t12_D(i, 0)
                   - my2(i, 0) * nx2(i, 0) * ny1(i, 0) * 2.0
                   - nx2(i, 0) * r1(i, 0) * s2(i, 0) * 2.0;
    coefsD(i, 2) = t7_D(i, 0) + t8_D(i, 0)
                   - mx2(i, 0) * nx1(i, 0) * ny2(i, 0) * 2.0
                   - ny2(i, 0) * r1(i, 0) * s2(i, 0) * 2.0;
    coefsD(i, 3) = t5_D(i, 0) + t6_D(i, 0)
                   - mx2(i, 0) * nx1(i, 0) * r2(i, 0) * 2.0
                   - my2(i, 0) * ny1(i, 0) * r2(i, 0) * 2.0;
    coefsD(i, 4) = t2_D(i, 0) - t3_D(i, 0) - t4_D(i, 0) + t9_D(i, 0)
                   + t10_D(i, 0) - mx2(i, 0) * ny2(i, 0) * r1(i, 0);
    coefsD(i, 5) = t5_D(i, 0) - t6_D(i, 0)
                   - mx2(i, 0) * nx1(i, 0) * r2(i, 0) * 2.0
                   + my2(i, 0) * ny1(i, 0) * r2(i, 0) * 2.0;
    coefsD(i, 6) = -t7_D(i, 0) + t8_D(i, 0)
                   + mx2(i, 0) * nx1(i, 0) * ny2(i, 0) * 2.0
                   - ny2(i, 0) * r1(i, 0) * s2(i, 0) * 2.0;
    coefsD(i, 7) = -t2_D(i, 0) + t3_D(i, 0) - t4_D(i, 0) - t9_D(i, 0)
                   + t10_D(i, 0) + t13_D(i, 0);
    coefsD(i, 8) = t11_D(i, 0) - t12_D(i, 0)
                   - my2(i, 0) * nx2(i, 0) * ny1(i, 0) * 2.0
                   + nx2(i, 0) * r1(i, 0) * s2(i, 0) * 2.0;
    coefsD(i, 9) = -t2_D(i, 0) - t3_D(i, 0) + t4_D(i, 0) + t9_D(i, 0)
                   - t10_D(i, 0) + t13_D(i, 0);
  }
}

void coefsNumDen(Eigen::MatrixXd &coefsND, const Eigen::VectorXd &a1,
                 const Eigen::VectorXd &a2, const Eigen::VectorXd &a3,
                 const Eigen::VectorXd &a4, const Eigen::VectorXd &a5,
                 const Eigen::VectorXd &a6, const Eigen::VectorXd &a7,
                 const Eigen::VectorXd &a8, const Eigen::VectorXd &a9,
                 const Eigen::VectorXd &a10, const Eigen::VectorXd &b1,
                 const Eigen::VectorXd &b2, const Eigen::VectorXd &b3,
                 const Eigen::VectorXd &b4, const Eigen::VectorXd &b5,
                 const Eigen::VectorXd &b6, const Eigen::VectorXd &b7,
                 const Eigen::VectorXd &b8, const Eigen::VectorXd &b9,
                 const Eigen::VectorXd &b10)
{
  auto numPts{a1.rows()};
  for (auto i{0}; i < numPts; ++i)
  {
    coefsND(i, 0) = a1(i, 0) * b1(i, 0);
    coefsND(i, 1) = a1(i, 0) * b2(i, 0) + a2(i, 0) * b1(i, 0);
    coefsND(i, 2)
        = a2(i, 0) * b2(i, 0) + a1(i, 0) * b5(i, 0) + a5(i, 0) * b1(i, 0);
    coefsND(i, 3) = a2(i, 0) * b5(i, 0) + a5(i, 0) * b2(i, 0);
    coefsND(i, 4) = a5(i, 0) * b5(i, 0);
    coefsND(i, 5) = a1(i, 0) * b3(i, 0) + a3(i, 0) * b1(i, 0);
    coefsND(i, 6) = a2(i, 0) * b3(i, 0) + a3(i, 0) * b2(i, 0)
                    + a1(i, 0) * b6(i, 0) + a6(i, 0) * b1(i, 0);
    coefsND(i, 7) = a2(i, 0) * b6(i, 0) + a3(i, 0) * b5(i, 0)
                    + a5(i, 0) * b3(i, 0) + a6(i, 0) * b2(i, 0);
    coefsND(i, 8) = a5(i, 0) * b6(i, 0) + a6(i, 0) * b5(i, 0);
    coefsND(i, 9)
        = a3(i, 0) * b3(i, 0) + a1(i, 0) * b8(i, 0) + a8(i, 0) * b1(i, 0);
    coefsND(i, 10) = a3(i, 0) * b6(i, 0) + a6(i, 0) * b3(i, 0)
                     + a2(i, 0) * b8(i, 0) + a8(i, 0) * b2(i, 0);
    coefsND(i, 11)
        = a6(i, 0) * b6(i, 0) + a5(i, 0) * b8(i, 0) + a8(i, 0) * b5(i, 0);
    coefsND(i, 12) = a3(i, 0) * b8(i, 0) + a8(i, 0) * b3(i, 0);
    coefsND(i, 13) = a6(i, 0) * b8(i, 0) + a8(i, 0) * b6(i, 0);
    coefsND(i, 14) = a8(i, 0) * b8(i, 0);
    coefsND(i, 15) = a1(i, 0) * b4(i, 0) + a4(i, 0) * b1(i, 0);
    coefsND(i, 16) = a2(i, 0) * b4(i, 0) + a4(i, 0) * b2(i, 0)
                     + a1(i, 0) * b7(i, 0) + a7(i, 0) * b1(i, 0);
    coefsND(i, 17) = a2(i, 0) * b7(i, 0) + a4(i, 0) * b5(i, 0)
                     + a5(i, 0) * b4(i, 0) + a7(i, 0) * b2(i, 0);
    coefsND(i, 18) = a5(i, 0) * b7(i, 0) + a7(i, 0) * b5(i, 0);
    coefsND(i, 19) = a3(i, 0) * b4(i, 0) + a4(i, 0) * b3(i, 0)
                     + a1(i, 0) * b9(i, 0) + a9(i, 0) * b1(i, 0);
    coefsND(i, 20) = a3(i, 0) * b7(i, 0) + a4(i, 0) * b6(i, 0)
                     + a6(i, 0) * b4(i, 0) + a7(i, 0) * b3(i, 0)
                     + a2(i, 0) * b9(i, 0) + a9(i, 0) * b2(i, 0);
    coefsND(i, 21) = a6(i, 0) * b7(i, 0) + a7(i, 0) * b6(i, 0)
                     + a5(i, 0) * b9(i, 0) + a9(i, 0) * b5(i, 0);
    coefsND(i, 22) = a3(i, 0) * b9(i, 0) + a4(i, 0) * b8(i, 0)
                     + a8(i, 0) * b4(i, 0) + a9(i, 0) * b3(i, 0);
    coefsND(i, 23) = a6(i, 0) * b9(i, 0) + a7(i, 0) * b8(i, 0)
                     + a8(i, 0) * b7(i, 0) + a9(i, 0) * b6(i, 0);
    coefsND(i, 24) = a8(i, 0) * b9(i, 0) + a9(i, 0) * b8(i, 0);
    coefsND(i, 25)
        = a4(i, 0) * b4(i, 0) + a1(i, 0) * b10(i, 0) + a10(i, 0) * b1(i, 0);
    coefsND(i, 26) = a4(i, 0) * b7(i, 0) + a7(i, 0) * b4(i, 0)
                     + a2(i, 0) * b10(i, 0) + a10(i, 0) * b2(i, 0);
    coefsND(i, 27)
        = a7(i, 0) * b7(i, 0) + a5(i, 0) * b10(i, 0) + a10(i, 0) * b5(i, 0);
    coefsND(i, 28) = a3(i, 0) * b10(i, 0) + a4(i, 0) * b9(i, 0)
                     + a9(i, 0) * b4(i, 0) + a10(i, 0) * b3(i, 0);
    coefsND(i, 29) = a6(i, 0) * b10(i, 0) + a7(i, 0) * b9(i, 0)
                     + a9(i, 0) * b7(i, 0) + a10(i, 0) * b6(i, 0);
    coefsND(i, 30)
        = a8(i, 0) * b10(i, 0) + a9(i, 0) * b9(i, 0) + a10(i, 0) * b8(i, 0);
    coefsND(i, 31) = a4(i, 0) * b10(i, 0) + a10(i, 0) * b4(i, 0);
    coefsND(i, 32) = a7(i, 0) * b10(i, 0) + a10(i, 0) * b7(i, 0);
    coefsND(i, 33) = a9(i, 0) * b10(i, 0) + a10(i, 0) * b9(i, 0);
    coefsND(i, 34) = a10(i, 0) * b10(i, 0);
  }
}

void CoefsVer_3_1_1(Eigen::MatrixXd &Cf, const Eigen::Matrix3Xd &m1,
                    const Eigen::Matrix3Xd &m2)
{
  auto numPts{m1.cols()};
  auto numCols{numPts * (numPts - 1) / 2 - 1};

  Eigen::Matrix2Xi idxBin1(2, numCols);
  size_t counter{0};
  for (decltype(numPts) i{1}; i <= numPts - 2; ++i)
  {
    for (auto j{i + 1}; j <= numPts; ++j)
    {
      ++counter;
      idxBin1(0, counter - 1) = i;
      idxBin1(1, counter - 1) = j;
    }
  }

  Eigen::VectorXd mx1(numCols, 1);
  Eigen::VectorXd my1(numCols, 1);
  Eigen::VectorXd s1(numCols, 1);
  Eigen::VectorXd nx1(numCols, 1);
  Eigen::VectorXd ny1(numCols, 1);
  Eigen::VectorXd r1(numCols, 1);
  Eigen::VectorXd mx2(numCols, 1);
  Eigen::VectorXd my2(numCols, 1);
  Eigen::VectorXd s2(numCols, 1);
  Eigen::VectorXd nx2(numCols, 1);
  Eigen::VectorXd ny2(numCols, 1);
  Eigen::VectorXd r2(numCols, 1);
  for (decltype(numCols) i{0}; i < numCols; ++i)
  {
    auto index_1{idxBin1(0, i)};
    mx1(i, 0) = m1(0, index_1 - 1);
    my1(i, 0) = m1(1, index_1 - 1);
    s1(i, 0)  = m1(2, index_1 - 1);
    nx1(i, 0) = m2(0, index_1 - 1);
    ny1(i, 0) = m2(1, index_1 - 1);
    r1(i, 0)  = m2(2, index_1 - 1);

    auto index_2{idxBin1(1, i)};
    mx2(i, 0) = m1(0, index_2 - 1);
    my2(i, 0) = m1(1, index_2 - 1);
    s2(i, 0)  = m1(2, index_2 - 1);
    nx2(i, 0) = m2(0, index_2 - 1);
    ny2(i, 0) = m2(1, index_2 - 1);
    r2(i, 0)  = m2(2, index_2 - 1);
  }

  Eigen::MatrixXd coefsN(numCols, 10);
  coefsNum(coefsN, mx1, mx2, my1, my2, nx2, ny2, r2, s1, s2);

  Eigen::MatrixXd coefsD(numCols, 10);
  coefsDen(coefsD, mx2, my2, nx1, nx2, ny1, ny2, r1, r2, s2);

  auto numEq{numPts * (numPts - 1) * (numPts - 2) / 6};

  Eigen::Matrix2Xi idxBin2(2, numEq);
  size_t counter_bin2_1{0};
  size_t counter_bin2_2{0};
  for (auto i{numPts - 1}; i >= 2; i--)
  {
    for (auto j{1 + counter_bin2_2}; j <= i - 1 + counter_bin2_2; ++j)
    {
      for (auto k{j + 1}; k <= i + counter_bin2_2; k++)
      {
        counter_bin2_1                 = counter_bin2_1 + 1;
        idxBin2(0, counter_bin2_1 - 1) = j;
        idxBin2(1, counter_bin2_1 - 1) = k;
      }
    }
    counter_bin2_2 = i + counter_bin2_2;
  }

  auto numEqDouble{2 * numEq};

  Eigen::VectorXd a1(numEqDouble, 1);
  Eigen::VectorXd a2(numEqDouble, 1);
  Eigen::VectorXd a3(numEqDouble, 1);
  Eigen::VectorXd a4(numEqDouble, 1);
  Eigen::VectorXd a5(numEqDouble, 1);
  Eigen::VectorXd a6(numEqDouble, 1);
  Eigen::VectorXd a7(numEqDouble, 1);
  Eigen::VectorXd a8(numEqDouble, 1);
  Eigen::VectorXd a9(numEqDouble, 1);
  Eigen::VectorXd a10(numEqDouble, 1);
  Eigen::VectorXd b1(numEqDouble, 1);
  Eigen::VectorXd b2(numEqDouble, 1);
  Eigen::VectorXd b3(numEqDouble, 1);
  Eigen::VectorXd b4(numEqDouble, 1);
  Eigen::VectorXd b5(numEqDouble, 1);
  Eigen::VectorXd b6(numEqDouble, 1);
  Eigen::VectorXd b7(numEqDouble, 1);
  Eigen::VectorXd b8(numEqDouble, 1);
  Eigen::VectorXd b9(numEqDouble, 1);
  Eigen::VectorXd b10(numEqDouble, 1);

  for (decltype(numEq) i{0}; i < numEq; ++i)
  {
    auto index_1{idxBin2(0, i)};
    a1(i, 0)          = coefsN(index_1 - 1, 0);
    a1(i + numEq, 0)  = coefsD(index_1 - 1, 0);
    a2(i, 0)          = coefsN(index_1 - 1, 1);
    a2(i + numEq, 0)  = coefsD(index_1 - 1, 1);
    a3(i, 0)          = coefsN(index_1 - 1, 2);
    a3(i + numEq, 0)  = coefsD(index_1 - 1, 2);
    a4(i, 0)          = coefsN(index_1 - 1, 3);
    a4(i + numEq, 0)  = coefsD(index_1 - 1, 3);
    a5(i, 0)          = coefsN(index_1 - 1, 4);
    a5(i + numEq, 0)  = coefsD(index_1 - 1, 4);
    a6(i, 0)          = coefsN(index_1 - 1, 5);
    a6(i + numEq, 0)  = coefsD(index_1 - 1, 5);
    a7(i, 0)          = coefsN(index_1 - 1, 6);
    a7(i + numEq, 0)  = coefsD(index_1 - 1, 6);
    a8(i, 0)          = coefsN(index_1 - 1, 7);
    a8(i + numEq, 0)  = coefsD(index_1 - 1, 7);
    a9(i, 0)          = coefsN(index_1 - 1, 8);
    a9(i + numEq, 0)  = coefsD(index_1 - 1, 8);
    a10(i, 0)         = coefsN(index_1 - 1, 9);
    a10(i + numEq, 0) = coefsD(index_1 - 1, 9);

    auto index_2{idxBin2(1, i)};
    b1(i, 0)          = coefsD(index_2 - 1, 0);
    b1(i + numEq, 0)  = coefsN(index_2 - 1, 0);
    b2(i, 0)          = coefsD(index_2 - 1, 1);
    b2(i + numEq, 0)  = coefsN(index_2 - 1, 1);
    b3(i, 0)          = coefsD(index_2 - 1, 2);
    b3(i + numEq, 0)  = coefsN(index_2 - 1, 2);
    b4(i, 0)          = coefsD(index_2 - 1, 3);
    b4(i + numEq, 0)  = coefsN(index_2 - 1, 3);
    b5(i, 0)          = coefsD(index_2 - 1, 4);
    b5(i + numEq, 0)  = coefsN(index_2 - 1, 4);
    b6(i, 0)          = coefsD(index_2 - 1, 5);
    b6(i + numEq, 0)  = coefsN(index_2 - 1, 5);
    b7(i, 0)          = coefsD(index_2 - 1, 6);
    b7(i + numEq, 0)  = coefsN(index_2 - 1, 6);
    b8(i, 0)          = coefsD(index_2 - 1, 7);
    b8(i + numEq, 0)  = coefsN(index_2 - 1, 7);
    b9(i, 0)          = coefsD(index_2 - 1, 8);
    b9(i + numEq, 0)  = coefsN(index_2 - 1, 8);
    b10(i, 0)         = coefsD(index_2 - 1, 9);
    b10(i + numEq, 0) = coefsN(index_2 - 1, 9);
  }

  Eigen::MatrixXd coefsND(numEqDouble, 35);
  coefsNumDen(coefsND, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4,
              b5, b6, b7, b8, b9, b10);

  for (decltype(numEq) i{0}; i < numEq; ++i)
  {
    Cf(i, 0)  = coefsND(i, 0) - coefsND(i + numEq, 0);
    Cf(i, 1)  = coefsND(i, 1) - coefsND(i + numEq, 1);
    Cf(i, 2)  = coefsND(i, 2) - coefsND(i + numEq, 2);
    Cf(i, 3)  = coefsND(i, 3) - coefsND(i + numEq, 3);
    Cf(i, 4)  = coefsND(i, 4) - coefsND(i + numEq, 4);
    Cf(i, 5)  = coefsND(i, 5) - coefsND(i + numEq, 5);
    Cf(i, 6)  = coefsND(i, 6) - coefsND(i + numEq, 6);
    Cf(i, 7)  = coefsND(i, 7) - coefsND(i + numEq, 7);
    Cf(i, 8)  = coefsND(i, 8) - coefsND(i + numEq, 8);
    Cf(i, 9)  = coefsND(i, 9) - coefsND(i + numEq, 9);
    Cf(i, 10) = coefsND(i, 10) - coefsND(i + numEq, 10);
    Cf(i, 11) = coefsND(i, 11) - coefsND(i + numEq, 11);
    Cf(i, 12) = coefsND(i, 12) - coefsND(i + numEq, 12);
    Cf(i, 13) = coefsND(i, 13) - coefsND(i + numEq, 13);
    Cf(i, 14) = coefsND(i, 14) - coefsND(i + numEq, 14);
    Cf(i, 15) = coefsND(i, 15) - coefsND(i + numEq, 15);
    Cf(i, 16) = coefsND(i, 16) - coefsND(i + numEq, 16);
    Cf(i, 17) = coefsND(i, 17) - coefsND(i + numEq, 17);
    Cf(i, 18) = coefsND(i, 18) - coefsND(i + numEq, 18);
    Cf(i, 19) = coefsND(i, 19) - coefsND(i + numEq, 19);
    Cf(i, 20) = coefsND(i, 20) - coefsND(i + numEq, 20);
    Cf(i, 21) = coefsND(i, 21) - coefsND(i + numEq, 21);
    Cf(i, 22) = coefsND(i, 22) - coefsND(i + numEq, 22);
    Cf(i, 23) = coefsND(i, 23) - coefsND(i + numEq, 23);
    Cf(i, 24) = coefsND(i, 24) - coefsND(i + numEq, 24);
    Cf(i, 25) = coefsND(i, 25) - coefsND(i + numEq, 25);
    Cf(i, 26) = coefsND(i, 26) - coefsND(i + numEq, 26);
    Cf(i, 27) = coefsND(i, 27) - coefsND(i + numEq, 27);
    Cf(i, 28) = coefsND(i, 28) - coefsND(i + numEq, 28);
    Cf(i, 29) = coefsND(i, 29) - coefsND(i + numEq, 29);
    Cf(i, 30) = coefsND(i, 30) - coefsND(i + numEq, 30);
    Cf(i, 31) = coefsND(i, 31) - coefsND(i + numEq, 31);
    Cf(i, 32) = coefsND(i, 32) - coefsND(i + numEq, 32);
    Cf(i, 33) = coefsND(i, 33) - coefsND(i + numEq, 33);
    Cf(i, 34) = coefsND(i, 34) - coefsND(i + numEq, 34);
  }
}

void QuEst_5Pt_Ver5_2(Eigen::Matrix4Xd &Q, const Eigen::Matrix3Xd &m,
                      const Eigen::Matrix3Xd &n)
{
  auto numPts{m.cols()};

  Eigen::Matrix4Xi Idx(4, 35);
  Idx << 1, 2, 5, 11, 21, 3, 6, 12, 22, 8, 14, 24, 17, 27, 31, 4, 7, 13, 23, 9,
      15, 25, 18, 28, 32, 10, 16, 26, 19, 29, 33, 20, 30, 34, 35, 2, 5, 11, 21,
      36, 6, 12, 22, 37, 14, 24, 39, 27, 42, 46, 7, 13, 23, 38, 15, 25, 40, 28,
      43, 47, 16, 26, 41, 29, 44, 48, 30, 45, 49, 50, 3, 6, 12, 22, 37, 8, 14,
      24, 39, 17, 27, 42, 31, 46, 51, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 19,
      29, 44, 33, 48, 53, 34, 49, 54, 55, 4, 7, 13, 23, 38, 9, 15, 25, 40, 18,
      28, 43, 32, 47, 52, 10, 16, 26, 41, 19, 29, 44, 33, 48, 53, 20, 30, 45,
      34, 49, 54, 35, 50, 55, 56;

  Eigen::RowVectorXi idx_w(35);
  idx_w << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35;

  Eigen::RowVectorXi idx_w0(21);
  idx_w0 << 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
      53, 54, 55, 56;

  Eigen::MatrixX4i Idx1(20, 4);
  Idx1 << 1, 2, 3, 4, 2, 5, 6, 7, 3, 6, 8, 9, 4, 7, 9, 10, 5, 11, 12, 13, 6, 12,
      14, 15, 7, 13, 15, 16, 8, 14, 17, 18, 9, 15, 18, 19, 10, 16, 19, 20, 11,
      21, 22, 23, 12, 22, 24, 25, 13, 23, 25, 26, 14, 24, 27, 28, 15, 25, 28,
      29, 16, 26, 29, 30, 17, 27, 31, 32, 18, 28, 32, 33, 19, 29, 33, 34, 20,
      30, 34, 35;

  Eigen::MatrixX4i Idx2(15, 4);
  Idx2 << 21, 1, 2, 3, 22, 2, 4, 5, 23, 3, 5, 6, 24, 4, 7, 8, 25, 5, 8, 9, 26,
      6, 9, 10, 27, 7, 11, 12, 28, 8, 12, 13, 29, 9, 13, 14, 30, 10, 14, 15, 31,
      11, 16, 17, 32, 12, 17, 18, 33, 13, 18, 19, 34, 14, 19, 20, 35, 15, 20,
      21;

  Eigen::MatrixXd Bx{Eigen::MatrixXd::Zero(35, 35)};

  auto numEq{numPts * (numPts - 1) * (numPts - 2) / 6};

  Eigen::MatrixXd Cf(numEq, 35); // coefficient matrix such that Cf * V = 0
  CoefsVer_3_1_1(Cf, m, n);

  Eigen::MatrixXd A{Eigen::MatrixXd::Zero(
      4 * numEq, 56)}; // coefficient matrix such that A * X = 0
  for (decltype(numEq) i{0}; i < numEq; ++i)
  {
    for (size_t j{0}; j < 35; ++j)
    {
      int idx_0       = Idx(0, j);
      A(i, idx_0 - 1) = Cf(i, j);

      int idx_1               = Idx(1, j);
      A(i + numEq, idx_1 - 1) = Cf(i, j);

      int idx_2                   = Idx(2, j);
      A(i + numEq * 2, idx_2 - 1) = Cf(i, j);

      int idx_3                   = Idx(3, j);
      A(i + numEq * 3, idx_3 - 1) = Cf(i, j);
    }
  }

  Eigen::MatrixXd A1(4 * numEq,
                     35); // split A into A1 and A2, A1 contains term w
  for (size_t i{0}; i < 40; ++i)
  {
    for (size_t j{0}; j < 35; ++j)
    {
      A1(i, j) = A(i, j);
    }
  }

  Eigen::MatrixXd A2(4 * numEq, 21); // A2 doesn't contains term w
  for (size_t i{0}; i < 40; ++i)
  {
    for (size_t j{0}; j < 21; ++j)
    {
      A2(i, j) = A(i, j + 35);
    }
  }

  Eigen::MatrixXd Bbar = -A2.completeOrthogonalDecomposition().solve(A1);

  for (size_t i{0}; i < 20; ++i)
  {
    Bx(Idx1(i, 0) - 1, Idx1(i, 1) - 1) = 1;
  }
  for (size_t i{0}; i < 15; ++i)
  {
    for (size_t j{0}; j < 35; ++j)
    {
      Bx(20 + i, j) = Bbar(i, j);
    }
  }

  // start armadillo
  // ------------------------------------------------------------
  arma::Mat<double> Bx_arma(35, 35);
  for (size_t i{0}; i < 35; ++i)
  {
    for (size_t j{0}; j < 35; ++j)
    {
      Bx_arma(i, j) = Bx(i, j);
    }
  }

  arma::cx_vec eigval;
  arma::cx_mat Ve_arma;
  eig_gen(eigval, Ve_arma, Bx_arma, "balance");

  arma::mat V_arma;
  V_arma = real(Ve_arma);

  // end armadillo  ------------------------------------------------------------

  Eigen::MatrixXd V(35, 35);
  for (size_t i{0}; i < 35; ++i)
  {
    for (size_t j{0}; j < 35; ++j)
    {
      V(i, j) = V_arma(i, j);
    }
  }

  // correct sign of each column, the first element is always positive
  Eigen::MatrixXd V_1(35, 35);
  for (size_t i{0}; i < 35; ++i)
  { // i represents column
    for (size_t j{0}; j < 35; ++j)
    { // j represent row

      if (V(0, i) < 0)
      {
        V_1(j, i) = V(j, i) * (-1);
      }
      else
      {
        V_1(j, i) = V(j, i) * (1);
      }
    }
  }

  // recover quaternion elements
  Eigen::RowVectorXd w(35);
  for (size_t i{0}; i < 35; ++i)
  {
    w(0, i) = sqrt(sqrt(V_1(0, i)));
  }

  Eigen::RowVectorXd w3(35);
  for (size_t i{0}; i < 35; ++i)
  {
    w3(0, i) = w(0, i) * w(0, i) * w(0, i);
  }

  Eigen::Matrix4Xd Q_0(4, 35);
  for (size_t i{0}; i < 35; ++i)
  {
    Q_0(0, i) = w(0, i);
    Q_0(1, i) = V_1(1, i) / w3(0, i);
    Q_0(2, i) = V_1(2, i) / w3(0, i);
    Q_0(3, i) = V_1(3, i) / w3(0, i);
  }

  Eigen::RowVectorXd QNrm(1, 35);
  for (size_t i{0}; i < 35; ++i)
  {
    QNrm(0, i) = sqrt(Q_0(0, i) * Q_0(0, i) + Q_0(1, i) * Q_0(1, i)
                      + Q_0(2, i) * Q_0(2, i) + Q_0(3, i) * Q_0(3, i));
  }

  // normalize each column
  for (size_t i{0}; i < 35; ++i)
  {
    Q(0, i) = Q_0(0, i) / QNrm(0, i);
    Q(1, i) = Q_0(1, i) / QNrm(0, i);
    Q(2, i) = Q_0(2, i) / QNrm(0, i);
    Q(3, i) = Q_0(3, i) / QNrm(0, i);
  }
}

void QuEst_Ver1_1(Eigen::Matrix4Xd &Q, const Eigen::Matrix3Xd &m,
                  const Eigen::Matrix3Xd &n)
{
  QuEst_5Pt_Ver5_2(Q, m, n);
}

void QuatResidue(Eigen::RowVectorXd &residu, const Eigen::Matrix3Xd &m1,
                 const Eigen::Matrix3Xd &m2, const Eigen::Matrix4Xd &qSol)
{
  auto numPts{m1.cols()};

  auto numEq{numPts * (numPts - 1) * (numPts - 2) / 6};

  Eigen::MatrixXd C0(numEq, 35);
  CoefsVer_3_1_1(C0, m1, m2); // coefficient matrix such that C * x = c

  Eigen::MatrixXd xVec(35, 35);
  for (size_t i{0}; i < 35; ++i)
  {
    xVec(0, i)  = qSol(0, i) * qSol(0, i) * qSol(0, i) * qSol(0, i);
    xVec(1, i)  = qSol(0, i) * qSol(0, i) * qSol(0, i) * qSol(1, i);
    xVec(2, i)  = qSol(0, i) * qSol(0, i) * qSol(1, i) * qSol(1, i);
    xVec(3, i)  = qSol(0, i) * qSol(1, i) * qSol(1, i) * qSol(1, i);
    xVec(4, i)  = qSol(1, i) * qSol(1, i) * qSol(1, i) * qSol(1, i);
    xVec(5, i)  = qSol(0, i) * qSol(0, i) * qSol(0, i) * qSol(2, i);
    xVec(6, i)  = qSol(0, i) * qSol(0, i) * qSol(1, i) * qSol(2, i);
    xVec(7, i)  = qSol(0, i) * qSol(1, i) * qSol(1, i) * qSol(2, i);
    xVec(8, i)  = qSol(1, i) * qSol(1, i) * qSol(1, i) * qSol(2, i);
    xVec(9, i)  = qSol(0, i) * qSol(0, i) * qSol(2, i) * qSol(2, i);
    xVec(10, i) = qSol(0, i) * qSol(1, i) * qSol(2, i) * qSol(2, i);
    xVec(11, i) = qSol(1, i) * qSol(1, i) * qSol(2, i) * qSol(2, i);
    xVec(12, i) = qSol(0, i) * qSol(2, i) * qSol(2, i) * qSol(2, i);
    xVec(13, i) = qSol(1, i) * qSol(2, i) * qSol(2, i) * qSol(2, i);
    xVec(14, i) = qSol(2, i) * qSol(2, i) * qSol(2, i) * qSol(2, i);
    xVec(15, i) = qSol(0, i) * qSol(0, i) * qSol(0, i) * qSol(3, i);
    xVec(16, i) = qSol(0, i) * qSol(0, i) * qSol(1, i) * qSol(3, i);
    xVec(17, i) = qSol(0, i) * qSol(1, i) * qSol(1, i) * qSol(3, i);
    xVec(18, i) = qSol(1, i) * qSol(1, i) * qSol(1, i) * qSol(3, i);
    xVec(19, i) = qSol(0, i) * qSol(0, i) * qSol(2, i) * qSol(3, i);
    xVec(20, i) = qSol(0, i) * qSol(1, i) * qSol(2, i) * qSol(3, i);
    xVec(21, i) = qSol(1, i) * qSol(1, i) * qSol(2, i) * qSol(3, i);
    xVec(22, i) = qSol(0, i) * qSol(2, i) * qSol(2, i) * qSol(3, i);
    xVec(23, i) = qSol(1, i) * qSol(2, i) * qSol(2, i) * qSol(3, i);
    xVec(24, i) = qSol(2, i) * qSol(2, i) * qSol(2, i) * qSol(3, i);
    xVec(25, i) = qSol(0, i) * qSol(0, i) * qSol(3, i) * qSol(3, i);
    xVec(26, i) = qSol(0, i) * qSol(1, i) * qSol(3, i) * qSol(3, i);
    xVec(27, i) = qSol(1, i) * qSol(1, i) * qSol(3, i) * qSol(3, i);
    xVec(28, i) = qSol(0, i) * qSol(2, i) * qSol(3, i) * qSol(3, i);
    xVec(29, i) = qSol(1, i) * qSol(2, i) * qSol(3, i) * qSol(3, i);
    xVec(30, i) = qSol(2, i) * qSol(2, i) * qSol(3, i) * qSol(3, i);
    xVec(31, i) = qSol(0, i) * qSol(3, i) * qSol(3, i) * qSol(3, i);
    xVec(32, i) = qSol(1, i) * qSol(3, i) * qSol(3, i) * qSol(3, i);
    xVec(33, i) = qSol(2, i) * qSol(3, i) * qSol(3, i) * qSol(3, i);
    xVec(34, i) = qSol(3, i) * qSol(3, i) * qSol(3, i) * qSol(3, i);
  }

  Eigen::MatrixXd residuMat(numEq, 35);
  residuMat = C0 * xVec;

  for (size_t i{0}; i < 35; ++i)
  {
    for (int j = 0; j < numEq; j++)
    {
      residu(0, i) = residu(0, i) + abs(residuMat(j, i));
    }
  }
}

// convert quaternion into rotation matrix ----------
void Q2R(Eigen::MatrixXd &R_Q2R, const Eigen::Matrix4Xd &Q)
{
  auto numInp_Q2R{Q.cols()};
  for (decltype(numInp_Q2R) i{0}; i < numInp_Q2R; ++i)
  {
    Eigen::Vector4d q;
    q(0, 0) = Q(0, i);
    q(1, 0) = Q(1, i);
    q(2, 0) = Q(2, i);
    q(3, 0) = Q(3, i);

    R_Q2R(0, i) = 1 - 2 * q(2, 0) * q(2, 0) - 2 * q(3, 0) * q(3, 0);
    R_Q2R(1, i) = 2 * q(1, 0) * q(2, 0) - 2 * q(3, 0) * q(0, 0);
    R_Q2R(2, i) = 2 * q(1, 0) * q(3, 0) + 2 * q(0, 0) * q(2, 0);
    R_Q2R(3, i) = 2 * q(1, 0) * q(2, 0) + 2 * q(3, 0) * q(0, 0);
    R_Q2R(4, i) = 1 - 2 * q(1, 0) * q(1, 0) - 2 * q(3, 0) * q(3, 0);
    R_Q2R(5, i) = 2 * q(2, 0) * q(3, 0) - 2 * q(1, 0) * q(0, 0);
    R_Q2R(6, i) = 2 * q(1, 0) * q(3, 0) - 2 * q(0, 0) * q(2, 0);
    R_Q2R(7, i) = 2 * q(2, 0) * q(3, 0) + 2 * q(1, 0) * q(0, 0);
    R_Q2R(8, i) = 1 - 2 * q(1, 0) * q(1, 0) - 2 * q(2, 0) * q(2, 0);
  }
}

void Q2R_3by3(Eigen::Matrix3d &R_Q2R, const Eigen::Vector4d &Q)
{

  R_Q2R(0, 0) = 1 - 2 * Q(2, 0) * Q(2, 0) - 2 * Q(3, 0) * Q(3, 0);
  R_Q2R(0, 1) = 2 * Q(1, 0) * Q(2, 0) - 2 * Q(3, 0) * Q(0, 0);
  R_Q2R(0, 2) = 2 * Q(1, 0) * Q(3, 0) + 2 * Q(0, 0) * Q(2, 0);
  R_Q2R(1, 0) = 2 * Q(1, 0) * Q(2, 0) + 2 * Q(3, 0) * Q(0, 0);
  R_Q2R(1, 1) = 1 - 2 * Q(1, 0) * Q(1, 0) - 2 * Q(3, 0) * Q(3, 0);
  R_Q2R(1, 2) = 2 * Q(2, 0) * Q(3, 0) - 2 * Q(1, 0) * Q(0, 0);
  R_Q2R(2, 0) = 2 * Q(1, 0) * Q(3, 0) - 2 * Q(0, 0) * Q(2, 0);
  R_Q2R(2, 1) = 2 * Q(2, 0) * Q(3, 0) + 2 * Q(1, 0) * Q(0, 0);
  R_Q2R(2, 2) = 1 - 2 * Q(1, 0) * Q(1, 0) - 2 * Q(2, 0) * Q(2, 0);
}

// recover translation, 3 by 35 ----------
void FindTrans(Eigen::Vector3d &T, const Eigen::Matrix3Xd &m,
               const Eigen::Matrix3Xd &n, const Eigen::Vector4d &Q)
{
  auto numCols{Q.cols()};

  Eigen::MatrixXd R{Eigen::MatrixXd::Zero(9, numCols)};
  Q2R(R, Q); // convert quaternion into rotation matrix

  auto numPts{m.cols()};
  auto numInp{R.cols()};

  for (decltype(numInp) k{0}; k < numInp; k++)
  {
    Eigen::MatrixXd C{Eigen::MatrixXd::Zero(3 * numPts, 2 * numPts + 3)};

    for (decltype(numPts) i{1}; i <= numPts; ++i)
    {
      C((i - 1) * 3, 0)     = 1;
      C((i - 1) * 3 + 1, 1) = 1;
      C((i - 1) * 3 + 2, 2) = 1;

      C((i - 1) * 3, (i - 1) * 2 + 3) = R(0, k) * m(0, i - 1)
                                        + R(1, k) * m(1, i - 1)
                                        + R(2, k) * m(2, i - 1);
      C((i - 1) * 3 + 1, (i - 1) * 2 + 3) = R(3, k) * m(0, i - 1)
                                            + R(4, k) * m(1, i - 1)
                                            + R(5, k) * m(2, i - 1);
      C((i - 1) * 3 + 2, (i - 1) * 2 + 3) = R(6, k) * m(0, i - 1)
                                            + R(7, k) * m(1, i - 1)
                                            + R(8, k) * m(2, i - 1);

      C((i - 1) * 3, (i - 1) * 2 + 4)     = -n(0, i - 1);
      C((i - 1) * 3 + 1, (i - 1) * 2 + 4) = -n(1, i - 1);
      C((i - 1) * 3 + 2, (i - 1) * 2 + 4) = -n(2, i - 1);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinV);
    Eigen::MatrixXd N = svd.matrixV();

    // adjust the sign
    size_t numPos{0};
    size_t numNeg{0};
    for (decltype(numPts) i{0}; i < 2 * numPts; ++i)
    {
      if (N(i + 3, 2 * numPts + 2) > 0)
      {
        numPos++;
      }
      if (N(i + 3, 2 * numPts + 2) < 0)
      {
        numNeg++;
      }
    }

    if (numPos < numNeg)
    {
      T(0, k) = -N(0, 2 * numPts + 2);
      T(1, k) = -N(1, 2 * numPts + 2);
      T(2, k) = -N(2, 2 * numPts + 2);
    }
    else
    {
      T(0, k) = N(0, 2 * numPts + 2);
      T(1, k) = N(1, 2 * numPts + 2);
      T(2, k) = N(2, 2 * numPts + 2);
    }
  }
}

void QuEst_fit(const Eigen::MatrixXd &allData, const VecIndex &useIndices,
               Eigen::VectorXd &test_model)
{

  auto numPts{useIndices.size()};

  // take allData out, into x1 and x2 ----------
  Eigen::Matrix3Xd x1(3, numPts);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    x1(0, i) = allData(0, useIndices[i]);
    x1(1, i) = allData(1, useIndices[i]);
    x1(2, i) = allData(2, useIndices[i]);
  }
  Eigen::Matrix3Xd x2(3, numPts);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    x2(0, i) = allData(3, useIndices[i]);
    x2(1, i) = allData(4, useIndices[i]);
    x2(2, i) = allData(5, useIndices[i]);
  }

  // take first 5 points of x1 and x2, feeding into QuEst ----------
  Eigen::Matrix3Xd x1_5pts(3, 5);
  for (size_t i{0}; i < 5; ++i)
  {
    x1_5pts(0, i) = x1(0, i);
    x1_5pts(1, i) = x1(1, i);
    x1_5pts(2, i) = x1(2, i);
  }
  Eigen::Matrix3Xd x2_5pts(3, 5);
  for (size_t i{0}; i < 5; ++i)
  {
    x2_5pts(0, i) = x2(0, i);
    x2_5pts(1, i) = x2(1, i);
    x2_5pts(2, i) = x2(2, i);
  }

  // run QuEst algorithm, get 35 candidates ----------
  Eigen::Matrix4Xd Q(4, 35);
  QuEst_Ver1_1(Q, x1_5pts, x2_5pts);

  // score function, pick the best estimated pose solution ----------
  Eigen::RowVectorXd res{Eigen::RowVectorXd::Zero(35)};
  QuatResidue(res, x1, x2, Q);

  int mIdx = 0;
  for (size_t i{1}; i < 35; ++i)
  {
    if (res(0, i) < res(0, mIdx))
    {
      mIdx = i;
    }
  }

  Eigen::Vector4d Q_select;
  Q_select(0, 0) = Q(0, mIdx);
  Q_select(1, 0) = Q(1, mIdx);
  Q_select(2, 0) = Q(2, mIdx);
  Q_select(3, 0) = Q(3, mIdx);

  Eigen::Vector3d T;
  FindTrans(T, x1_5pts, x2_5pts, Q_select);

  test_model(0) = T(0, 0);
  test_model(1) = T(1, 0);
  test_model(2) = T(2, 0);
  test_model(3) = Q_select(0, 0);
  test_model(4) = Q_select(1, 0);
  test_model(5) = Q_select(2, 0);
  test_model(6) = Q_select(3, 0);
}

bool QuEst_degenerate(const Eigen::MatrixXd &data, const VecIndex &ind)
{
  (void) data;
  (void) ind;
  return false;
}

VecIndex RandomIndexes(size_t maxIndex)
{
  VecIndex ind(maxIndex);
  for (size_t i{0}; i < maxIndex; ++i)
  {
    ind[i] = i;
  }
  auto seed{std::chrono::system_clock::now().time_since_epoch().count()};
  std::shuffle(ind.begin(), ind.end(), std::default_random_engine(seed));
  return ind;
}

void QuEst_distance(const Eigen::MatrixXd &data,
                    const Eigen::VectorXd &test_model,
                    const double distance_threshold,
                    Eigen::VectorXd &select_model,
                    std::vector<int> &select_inliers)
{
  auto numPts{data.cols()};

  // extract all feature points
  Eigen::Matrix3Xd x1(3, numPts);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    x1(0, i) = data(0, i);
    x1(1, i) = data(1, i);
    x1(2, i) = data(2, i);
  }
  Eigen::Matrix3Xd x2(3, numPts);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    x2(0, i) = data(3, i);
    x2(1, i) = data(4, i);
    x2(2, i) = data(5, i);
  }

  // rotation matrix
  Eigen::Vector4d q(4, 1);
  q(0, 0) = test_model(3, 0);
  q(1, 0) = test_model(4, 0);
  q(2, 0) = test_model(5, 0);
  q(3, 0) = test_model(6, 0);
  Eigen::Matrix3d R;
  Q2R_3by3(R, q);

  // skew matrix
  double t_norm = std::sqrt(test_model(0, 0) * test_model(0, 0)
                            + test_model(1, 0) * test_model(1, 0)
                            + test_model(2, 0) * test_model(2, 0));

  Eigen::Vector3d t(3, 1);
  t(0, 0) = test_model(0, 0) / t_norm;
  t(1, 0) = test_model(1, 0) / t_norm;
  t(2, 0) = test_model(2, 0) / t_norm;

  Eigen::Matrix3d Tx;
  Tx(0, 0) = 0;
  Tx(0, 1) = -t(2, 0);
  Tx(0, 2) = t(1, 0);
  Tx(1, 0) = t(2, 0);
  Tx(1, 1) = 0;
  Tx(1, 2) = -t(0, 0);
  Tx(2, 0) = -t(1, 0);
  Tx(2, 1) = t(0, 0);
  Tx(2, 2) = 0;

  // fundamental matrix ----------
  Eigen::Matrix3d F(3, 3);
  F = Tx * R;

  Eigen::RowVectorXd x2tFx1{Eigen::RowVectorXd::Zero(numPts)};
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    x2tFx1(0, i)
        = (x2(0, i) * F(0, 0) + x2(1, i) * F(1, 0) + x2(2, i) * F(2, 0))
              * x1(0, i)
          + (x2(0, i) * F(0, 1) + x2(1, i) * F(1, 1) + x2(2, i) * F(2, 1))
                * x1(1, i)
          + (x2(0, i) * F(0, 2) + x2(1, i) * F(1, 2) + x2(2, i) * F(2, 2))
                * x1(2, i);
  }

  // evaluate distance ----------
  Eigen::Matrix3Xd Fx1(3, numPts);
  Fx1 = F * x1;

  Eigen::Matrix3Xd Ftx2(3, numPts);
  Ftx2 = F.transpose() * x2;

  select_inliers.clear();

  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    double d = x2tFx1(0, i) * x2tFx1(0, i)
               / (Fx1(0, i) * Fx1(0, i) + Fx1(1, i) * Fx1(1, i)
                  + Ftx2(0, i) * Ftx2(0, i) + Ftx2(1, i) * Ftx2(1, i));

    if (abs(d) < distance_threshold)
    {
      select_inliers.push_back(i);
    }
  }

  select_model = test_model;
}

bool QuEst_RANSAC0(
    void (*QuEst_fit)(const Eigen::MatrixXd &, const VecIndex &,
                      Eigen::VectorXd &),
    const Eigen::MatrixXd &data, const VecIndex &ind,
    Eigen::VectorXd &test_model,
    void (*QuEst_distance)(const Eigen::MatrixXd &, const Eigen::VectorXd &,
                           const double, Eigen::VectorXd &, std::vector<int> &),
    const Eigen::MatrixXd &data_1, const Eigen::VectorXd &ind_1,
    const double distance_threshold, Eigen::VectorXd &select_model,
    std::vector<int> &select_inliers,
    bool (*QuEst_degenerate)(const Eigen::MatrixXd &, const VecIndex &),
    const Eigen::MatrixXd &data_2, const VecIndex &ind_2,
    const size_t minimumSizeSamplesToFit, Eigen::VectorXd &best_model,
    std::vector<int> &best_inliers)
{
  (void) ind;
  (void) data_1;
  (void) ind_1;

  const auto Npts{data.cols()}; // number of points in data

  const double p{0.99}; // desired probability of choosing at least one sample
                        // free from outliers

  // max number to select non-degenerate data set
  const size_t maxDataTrials{100};

  const size_t maxIter{1000}; // max number of iterations

  size_t trialcount{0};

  size_t bestscore{0};

  double N{1.0}; // dummy initialisation for number of trials

  while (N > trialcount)
  {

    bool degenerate{true};
    size_t count{1};

    while (degenerate)
    {

      // test that these points are not a degenerate configuration
      degenerate = QuEst_degenerate(data_2, ind_2);

      if (!degenerate)
      {
        // fit model to this random selection of data points
        VecIndex ind_degen = RandomIndexes(static_cast<size_t>(Npts));
        VecIndex ind_degen_select(minimumSizeSamplesToFit);
        for (size_t i{0}; i < minimumSizeSamplesToFit; ++i)
        {
          ind_degen_select[i] = i;
        }

        (*QuEst_fit)(data, ind_degen_select, test_model);
      }

      if (++count > maxDataTrials)
      {
        // safeguard against being stuck in this loop forever
        break;
      }
    }

    // once we are out here, we should have some kind of model
    // evaluate distances between points and model
    // returning the indices of elements that are inliers
    (*QuEst_distance)(data, test_model, distance_threshold, select_model,
                      select_inliers);

    // find the number of inliers to this model
    size_t ninliers{select_inliers.size()};

    if (ninliers > bestscore)
    {
      bestscore    = ninliers;
      best_inliers = select_inliers;
      best_model   = select_model;

      // update estimate of N, the number of trials,
      // to ensure we pick with probability p, a data set with no outliers
      double fracinliers{ninliers / static_cast<double>(Npts)};
      double pNoOutliers{
          1 - pow(fracinliers, static_cast<double>(minimumSizeSamplesToFit))};

      // avoid division by -Inf
      pNoOutliers
          = std::max(std::numeric_limits<double>::epsilon(), pNoOutliers);
      // avoid division by 0
      pNoOutliers
          = std::min(1.0 - std::numeric_limits<double>::epsilon(), pNoOutliers);

      // update N
      N = log(1 - p) / log(pNoOutliers);
    }

    ++trialcount;

    if (trialcount > maxIter)
    {
      break;
    }
  }

  if (best_model.rows() > 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

ResultType QuEst_RANSAC(Eigen::Matrix3Xd const &x1, Eigen::Matrix3Xd const &x2)
{
  auto numPts{x1.cols()};
  Eigen::RowVectorXd x1n(1, numPts);
  Eigen::RowVectorXd x2n(1, numPts);
  Eigen::Matrix3Xd x1_normalized(3, numPts);
  Eigen::Matrix3Xd x2_normalized(3, numPts);
  // calculate L1 norm
  x1n = x1.colwise().lpNorm<1>();
  x2n = x2.colwise().lpNorm<1>();
  // normalize the points
  x1_normalized = x1 / x1n;
  x2_normalized = x2 / x2n;
  // formulate data
  Eigen::MatrixXd data(6, numPts);
  for (decltype(numPts) i{0}; i < numPts; ++i)
  {
    data(0, i) = x1_normalized(0, i);
    data(1, i) = x1_normalized(1, i);
    data(2, i) = x1_normalized(2, i);
    data(3, i) = x2_normalized(0, i);
    data(4, i) = x2_normalized(1, i);
    data(5, i) = x2_normalized(2, i);
  }
  // minimum samples for fitting function
  const size_t minimumSizeSamplesToFit{6};
  // distance threshold between data and model
  const double distance_threshold{1e-6};
  // output model and inliers
  Eigen::VectorXd test_model(7);
  Eigen::VectorXd select_model(7);
  Eigen::VectorXd best_model(7);
  std::vector<int> select_inliers;
  std::vector<int> best_inliers;
  // random indicies in range (0, minimumSizeSamplesToFit-1)
  VecIndex ind = RandomIndexes(minimumSizeSamplesToFit);
  QuEst_RANSAC0(&QuEst_fit, data, ind, test_model, &QuEst_distance, data,
                test_model, distance_threshold, select_model, select_inliers,
                &QuEst_degenerate, data, ind, minimumSizeSamplesToFit,
                best_model, best_inliers);

  double model_trans_x_sat;
  if (best_model(0) > 5)
  {
    model_trans_x_sat = 5;
  }
  else if (best_model(0) < -5)
  {
    model_trans_x_sat = -5;
  }
  else
  {
    model_trans_x_sat = best_model(0);
  }
  double model_trans_y_sat;
  if (best_model(1) > 5)
  {
    model_trans_y_sat = 5;
  }
  else if (best_model(1) < -5)
  {
    model_trans_y_sat = -5;
  }
  else
  {
    model_trans_y_sat = best_model(1);
  }
  double model_trans_z_sat;
  if (best_model(2) > 5)
  {
    model_trans_z_sat = 5;
  }
  else if (best_model(2) < -5)
  {
    model_trans_z_sat = -5;
  }
  else
  {
    model_trans_z_sat = best_model(2);
  }
  double model_rotation_w_sat;
  if (best_model(3) > 1)
  {
    model_rotation_w_sat = 1;
  }
  else if (best_model(3) < -1)
  {
    model_rotation_w_sat = -1;
  }
  else
  {
    model_rotation_w_sat = best_model(3);
  }
  double model_rotation_x_sat;
  if (best_model(4) > 1)
  {
    model_rotation_x_sat = 1;
  }
  else if (best_model(4) < -1)
  {
    model_rotation_x_sat = -1;
  }
  else
  {
    model_rotation_x_sat = best_model(4);
  }
  double model_rotation_y_sat;
  if (best_model(5) > 1)
  {
    model_rotation_y_sat = 1;
  }
  else if (best_model(5) < -1)
  {
    model_rotation_y_sat = -1;
  }
  else
  {
    model_rotation_y_sat = best_model(5);
  }
  double model_rotation_z_sat;
  if (best_model(6) > 1)
  {
    model_rotation_z_sat = 1;
  }
  else if (best_model(6) < -1)
  {
    model_rotation_z_sat = -1;
  }
  else
  {
    model_rotation_z_sat = best_model(6);
  }

  ResultType result;
  result.translation.x = model_trans_x_sat;
  result.translation.y = model_trans_y_sat;
  result.translation.z = model_trans_z_sat;
  result.rotation.w    = model_rotation_w_sat;
  result.rotation.x    = model_rotation_x_sat;
  result.rotation.y    = model_rotation_y_sat;
  result.rotation.z    = model_rotation_z_sat;

  return result;
}

ResultType QuEst_Solve(Eigen::Matrix3Xd const &M1, Eigen::Matrix3Xd const &M2,
                       Eigen::Matrix3d const &K1, Eigen::Matrix3d const &K2)
{
  Eigen::Matrix3Xd m1{K1.inverse() * M1};
  Eigen::Matrix3Xd m2{K2.inverse() * M2};
  return QuEst_RANSAC(m1, m2);
}

Eigen::Matrix3Xd CreateMatrix(std::vector<cv::Point2f> const &pts)
{
  Eigen::Matrix3Xd M(3, pts.size());
  for (size_t i{0}; i < pts.size(); ++i)
  {
    M(0, i) = pts[i].x;
    M(1, i) = pts[i].y;
    M(2, i) = 1.0;
  }
  return M;
}

ResultType QuEst_Solve(std::vector<cv::Point2f> const &pts1,
                       std::vector<cv::Point2f> const &pts2,
                       Eigen::Matrix3d const &K1, Eigen::Matrix3d const &K2)
{
  Eigen::Matrix3Xd M1{CreateMatrix(pts1)};
  Eigen::Matrix3Xd M2{CreateMatrix(pts2)};
  return QuEst_Solve(M1, M2, K1, K2);
}

#endif /* QUEST_HPP */
