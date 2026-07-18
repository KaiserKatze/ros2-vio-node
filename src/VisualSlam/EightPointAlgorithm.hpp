#pragma once

#include <cstddef>
#include <tuple>

#include <Eigen/Dense>

struct ProjectiveGeometry
{
  static void Homo2Nonhomo(const Eigen::Matrix3Xd &homo,
                           Eigen::Matrix2Xd &nonhomo)
  {
    if (homo.rows() != 3)
    {
      throw std::runtime_error("Homo2Nonhomo: homo must be 3xN shape.");
    }
    nonhomo.resize(2, homo.cols());
    Eigen::Array<double, 1, Eigen::Dynamic> denom = homo.row(2).array();
    denom                  = (denom.abs() < 1e-12).select(1e-12, denom);
    nonhomo.row(0).array() = homo.row(0).array() / denom;
    nonhomo.row(1).array() = homo.row(1).array() / denom;
  }

  static void Nonhomo2Homo(const Eigen::Matrix2Xd &nonhomo,
                           Eigen::Matrix3Xd &homo)
  {
    if (nonhomo.rows() != 2)
    {
      throw std::runtime_error("Nonhomo2Homo: nonhomo must be 2xN shape.");
    }
    homo.resize(3, nonhomo.cols());
    homo.row(0).array() = nonhomo.row(0).array();
    homo.row(1).array() = nonhomo.row(1).array();
    homo.row(2)         = Eigen::RowVectorXd::Ones(nonhomo.cols());
  }

  struct DistortFunction // 基类
  {

    virtual ~DistortFunction() = 0;

    /**
     * @param points: 理想的（无畸变的）像素点坐标，归一化图像坐标系
     */
    virtual void Distort(Eigen::Matrix3Xd &) const {}

    virtual void DistortPoint(Eigen::Vector2d &) const {}
  };

  template <std::size_t nk = 2, std::size_t np = 2, typename E = double>
  struct DistortFunctionBrownConrady : DistortFunction
  {
    using VectorK = Eigen::Matrix<E, nk, 1>;
    using VectorP = Eigen::Matrix<E, np, 1>;
    VectorK k;
    VectorP p;

    using Scalar = E;

    virtual void Distort(Eigen::Matrix3Xd &points) const override
    {
      if (points.rows() != 3)
      {
        throw std::runtime_error(
            "DistortFunctionBrownConrady::Distort: points must be 3xN."
        );
      }
      Eigen::Matrix2Xd nonhomo;
      Homo2Nonhomo(points, nonhomo);
      for (int i = 0; i < nonhomo.cols(); ++i)
      {
        Eigen::Vector2d point = nonhomo.col(i);
        DistortPoint(point);
        nonhomo.col(i) = point;
      }
      points.row(0) = nonhomo.row(0);
      points.row(1) = nonhomo.row(1);
      points.row(2).setOnes();
    }

    virtual void DistortPoint(Eigen::Vector2d &point) const override
    {
      // 施加径向畸变
      const Scalar r2{point.squaredNorm()};
      Scalar coeff{1.0};
      const auto k_rows{this->k.rows()};
      coeff += this->k(0) * r2;
      const Scalar r4{r2 * r2};
      coeff += this->k(1) * r4;
      point *= coeff;
      // 施加切向畸变
      const auto p_rows{this->p.rows()};
      Scalar &x{point(0)};
      Scalar &y{point(1)};
      const Scalar p1{this->p(0)};
      const Scalar p2{this->p(1)};
      const Scalar delta_x{2 * p1 * x * y + p2 * (r2 + 2 * x * x)};
      const Scalar delta_y{2 * p2 * x * y * p1 * (r2 + 2 * y * y)};
      x += delta_x;
      y += delta_y;
    }
  };

  /**
 * 将模型点投影到像平面上，得到像素点的非齐次坐标
 *
 * @param modelPointsInWorldCoordinates: 模型点在世界坐标系中的齐次坐标（默认 Z 坐标为 0）
 * @param iMat: 相机内部参数矩阵
 * @param rMat: 旋转矩阵
 * @param tVec: 平移向量
 */
  static Eigen::Matrix2Xd
  Project(const Eigen::Matrix3Xd &modelPointsInWorldCoordinates,
          const Eigen::Matrix3d &iMat, const Eigen::Matrix3d &rMat,
          const Eigen::Vector3d &tVec, const DistortFunction &distortFunction)
  {
    if (modelPointsInWorldCoordinates.rows() != 3)
    {
      throw std::runtime_error(
          "Project: modelPointsInWorldCoordinates must be 3xN."
      );
    }
    Eigen::Matrix3d rtMat;
    Eigen::Matrix3Xd modelPointsInCameraCoordinates;
    Eigen::Matrix3Xd pixelPointsInImageCoordinates;
    Eigen::Matrix2Xd pixelPointsInPixelCoordinates;

    rtMat.col(0) = rMat.col(0);
    rtMat.col(1) = rMat.col(1);
    rtMat.col(2) = tVec;
    // 将模型点的齐次坐标从世界坐标系变换到相机坐标系
    modelPointsInCameraCoordinates = rtMat * modelPointsInWorldCoordinates;
    // 套用畸变模型
    distortFunction.Distort(modelPointsInCameraCoordinates);
    // 将模型点投影到像平面，得到像素点在像平面坐标系上的齐次坐标
    pixelPointsInImageCoordinates = iMat * modelPointsInCameraCoordinates;
    // 归一化得到像素点的非齐次坐标
    Homo2Nonhomo(pixelPointsInImageCoordinates, pixelPointsInPixelCoordinates);
    return pixelPointsInPixelCoordinates;
  }

  static Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d &vec)
  {
    const auto x{vec.x()};
    const auto y{vec.y()};
    const auto z{vec.z()};

    Eigen::Matrix3d vec_antisym{
        {0.0, -z, y},
        {z, 0.0, -x},
        {-y, x, 0.0},
    };
    return vec_antisym;
  }
};

template <typename value_type = float>
struct EightPointAlgorithm
{
  /**
 * @brief 各向同性逆归一化
 * @return 各向同性归一化变换矩阵
 */
  static Eigen::Matrix<value_type, 3, 3> IsotropicScalingNormalize(
      Eigen::Matrix<value_type, 3, Eigen::Dynamic> &points
  )
  {
    if (points.rows() != 3 || points.cols() == 0)
    {
      throw std::runtime_error{
          "IsotropicScalingNormalize: points must be 3xN.",
      };
    }
    static const value_type tiny = 1e-8;

    value_type centroidX = points.row(0).mean();
    value_type centroidY = points.row(1).mean();

    Eigen::Matrix<value_type, 2, Eigen::Dynamic> centeredPoints
        = points.topRows(2);
    centeredPoints.row(0).array() -= centroidX;
    centeredPoints.row(1).array() -= centroidY;
    Eigen::Vector<value_type, Eigen::Dynamic> distances
        = centeredPoints.colwise().norm();
    value_type meanDistance = distances.mean();

    value_type scale
        = (meanDistance < tiny) ? 1.0 : (std::sqrt(2.0) / meanDistance);

    Eigen::Matrix<value_type, 3, 3> sMat;
    sMat << scale, 0.0, -scale * centroidX, 0.0, scale, -scale * centroidY, 0.0,
        0.0, 1.0;

    points = sMat * points;
    return sMat;
  }

  struct TriangulationConfig
  {
    // 左目相机内参矩阵
    Eigen::Matrix<value_type, 3, 3> mat_cam_left_;
    // 右目相机内参矩阵
    Eigen::Matrix<value_type, 3, 3> mat_cam_right_;
    // 从左目相机到右目相机的旋转
    Eigen::Matrix<value_type, 3, 3> rotation_;
    // 从左目相机到游牧相机的平移
    Eigen::Vector<value_type, 3> translation_;
    // 路标点在左目像平面上的投影点的齐次坐标
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_left_;
    // 路标点在右目像平面上的投影点的齐次坐标
    Eigen::Matrix<value_type, 3, Eigen::Dynamic> pixel_right_;
    // 左目视图的各向同性归一化变换
    Eigen::Matrix<value_type, 3, 3> T_left_;
    // 右目视图的各向同性归一化变换
    Eigen::Matrix<value_type, 3, 3> T_right_;

    TriangulationConfig(
        const Eigen::Matrix<value_type, 3, 3> &mat_cam_left,
        const Eigen::Matrix<value_type, 3, 3> &mat_cam_right,
        const Eigen::Matrix<value_type, 3, 3> &rotation,
        const Eigen::Vector<value_type, 3> &translation,
        const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &pixel_left,
        const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &pixel_right,
        const Eigen::Matrix<value_type, 3, 3> &T_left,
        const Eigen::Matrix<value_type, 3, 3> &T_right
    ) :
      mat_cam_left_{mat_cam_left}, mat_cam_right_{mat_cam_right},
      rotation_{rotation}, translation_{translation}, pixel_left_{pixel_left},
      pixel_right_{pixel_right}, T_left_{T_left}, T_right_{T_right}
    {
    }

    TriangulationConfig(
        const Eigen::Matrix<value_type, 3, 3> &mat_cam_left,
        const Eigen::Matrix<value_type, 3, 3> &mat_cam_right,
        const Eigen::Matrix<value_type, 3, 3> &rotation,
        const Eigen::Vector<value_type, 3> &translation,
        const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &pixel_left,
        const Eigen::Matrix<value_type, 3, Eigen::Dynamic> &pixel_right
    ) :
      TriangulationConfig{
          mat_cam_left,
          mat_cam_right,
          rotation,
          translation,
          pixel_left,
          pixel_right,
          Eigen::Matrix<value_type, 3, 3>::Identity(),
          Eigen::Matrix<value_type, 3, 3>::Identity(),
      }
    {
      T_left_  = IsotropicScalingNormalize(pixel_left_);
      T_right_ = IsotropicScalingNormalize(pixel_right_);
    }

    Eigen::Vector<value_type, 3> GetLeftEpipole() const
    {
      return mat_cam_left_ * rotation_.transpose() * translation_;
    }

    Eigen::Vector<value_type, 3> GetRightEpipole() const
    {
      return mat_cam_right_ * translation_;
    }

    /**
   * @brief 利用两个相机内参矩阵、旋转矩阵、平移向量，计算基础矩阵
   */
    Eigen::Matrix<value_type, 3, 3> ComputeFundamentalMatrix() const
    {
      // 左目视图中极点的齐次坐标
      const Eigen::Vector<value_type, 3> epipole_left{GetLeftEpipole()};
      // 左极点齐次坐标的叉乘矩阵
      Eigen::Matrix<value_type, 3, 3> epipole_left_antisym;
      epipole_left_antisym << 0.0, -epipole_left.z(), epipole_left.y(), //
          epipole_left.z(), 0.0, -epipole_left.x(),                     //
          -epipole_left.y(), epipole_left.x(), 0.0;
      return mat_cam_right_.transpose().inverse() * rotation_
             * mat_cam_left_.transpose() * epipole_left_antisym;
    }

    Eigen::Matrix<value_type, 3, 3> ComputeEssentialMatrix(
        const Eigen::Matrix<value_type, 3, 3> &fundamental_matrix
    ) const
    {
      return mat_cam_right_.transpose() * fundamental_matrix * mat_cam_left_;
    }
  };

  /**
 * @brief 三角化
 * @return 路标点的齐次坐标
 */
  static Eigen::Matrix<value_type, 4, Eigen::Dynamic>
  Triangulate(const TriangulationConfig &config)
  {
    if (config.pixel_left_.cols() != config.pixel_right_.cols())
    {
      throw std::runtime_error(
          "Triangulate: number of pixel points in two views must match."
      );
    }

    Eigen::Matrix<value_type, 3, 4> projectMatrix_left;
    projectMatrix_left << config.mat_cam_left_,
        Eigen::Vector<value_type, 3>::Zero();

    const auto m1_left{projectMatrix_left.row(0)};
    const auto m2_left{projectMatrix_left.row(1)};
    const auto m3_left{projectMatrix_left.row(2)};

    Eigen::Matrix<value_type, 3, 4> projectMatrix_right;
    projectMatrix_right << config.rotation_, config.translation_;
    projectMatrix_right = config.mat_cam_right_ * projectMatrix_right;

    const auto m1_right{projectMatrix_right.row(0)};
    const auto m2_right{projectMatrix_right.row(1)};
    const auto m3_right{projectMatrix_right.row(2)};

    auto nCols{config.pixel_left_.cols()};
    Eigen::Matrix<value_type, 4, Eigen::Dynamic> matA(4, 4 * nCols);
    matA.setZero();
    for (decltype(nCols) i = 0; i < nCols; ++i)
    {
      const value_type u_left{config.pixel_left_(0, i)};
      const value_type v_left{config.pixel_left_(1, i)};
      const value_type u_right{config.pixel_right_(0, i)};
      const value_type v_right{config.pixel_right_(1, i)};

      matA(Eigen::all, Eigen::seqN(4 * i, 4)) //
          << u_left * m3_left - m1_left,
          v_left * m3_left - m2_left, u_right * m3_right - m1_right,
          v_right * m3_right - m2_right;
    }

    Eigen::JacobiSVD<decltype(matA)> svd{matA, Eigen::ComputeThinV};
    Eigen::Vector<value_type, Eigen::Dynamic> matA_svd_result{
        svd.matrixV().col(svd.matrixV().cols() - 1)
    };
    Eigen::Map<decltype(matA)> modelPointsInWorldCoordinates{
        matA_svd_result.data(),
        4,
        matA_svd_result.size() / 4,
    };

    // TODO 以后用 LM 方法
    // 将 modelPointsInWorldCoordinates 作为初始值
    // 通过缩小重投影误差
    // 求出路标点的最优坐标

    return modelPointsInWorldCoordinates;
  }

  /**
 * @brief 估计基础矩阵
 * @return 基础矩阵
 */
  static Eigen::Matrix<value_type, 3, 3>
  EstimateFundamentalMatrix(const TriangulationConfig &config)
  {
    auto nCols{config.pixel_left_.cols()};
    Eigen::Matrix<value_type, Eigen::Dynamic, 9> matW(nCols, 9);
    matW.setZero();
    for (decltype(nCols) i = 0; i < nCols; ++i)
    {
      const value_type u_left{config.pixel_left_(0, i)};
      const value_type v_left{config.pixel_left_(1, i)};
      const value_type u_right{config.pixel_right_(0, i)};
      const value_type v_right{config.pixel_right_(1, i)};
      matW.row(i)              //
          << u_left * u_right, //
          v_left * u_right,    //
          u_right,             //
          u_left * v_right,    //
          v_left * v_right,    //
          v_right,             //
          u_left,              //
          v_left,              //
          1;
    }

    // 进行第一次奇异值分解（给出最小二乘解）
    Eigen::JacobiSVD<decltype(matW)> svd1{matW, Eigen::ComputeFullV};
    Eigen::Vector<value_type, Eigen::Dynamic> matW_svd_result{
        svd1.matrixV().col(svd1.matrixV().cols() - 1)
    };

    // 进行第二次奇异值分解（保证基础矩阵的秩为2）
    Eigen::JacobiSVD<Eigen::Matrix<value_type, 3, 3>> svd2{
        Eigen::Map<Eigen::Matrix<value_type, 3, 3>>{matW_svd_result.data(), 3,
                                                    3},
        Eigen::ComputeFullU | Eigen::ComputeFullV,
    };
    Eigen::Vector<value_type, 3> singularValues{svd2.singularValues()};
    singularValues[2] = 0.0; // 将最小奇异值置零
    Eigen::Matrix<value_type, 3, 3> fundamental_matrix{
        svd2.matrixU() * singularValues.asDiagonal()
            * svd2.matrixV().transpose(),
    };

    // 各向同性逆归一化
    fundamental_matrix
        = config.T_right_.transpose() * fundamental_matrix * config.T_left_;

    return fundamental_matrix;
  }

  /**
 * @brief 估计单应矩阵
 * @return 单应矩阵
 */
  static Eigen::Matrix<value_type, 3, 3>
  EstimateHomography(const TriangulationConfig &&config)
  {
    auto nCols{config.pixel_left_.cols()};
    Eigen::Matrix<value_type, Eigen::Dynamic, 9> matA(2 * nCols, 9);

    matA.setZero();
    for (decltype(nCols) i = 0; i < nCols; ++i)
    {
      const value_type u_left{config.pixel_left_(0, i)};
      const value_type v_left{config.pixel_left_(1, i)};
      const value_type u_right{config.pixel_right_(0, i)};
      const value_type v_right{config.pixel_right_(1, i)};
      Eigen::RowVector<value_type, 3> zero_vector{
          Eigen::RowVector<value_type, 3>::Zero()
      };
      matA.row(2 * i + 0)                //
          << -config.pixel_left_.col(i), // -u_left, -v_left, -1.0,
          zero_vector,                   //
          u_left * u_right,              //
          v_left * u_right,              //
          u_right;                       //
      matA.row(2 * i + 1)                //
          << zero_vector,                //
          -config.pixel_left_.col(i),    // -u_left, -v_left, -1.0,
          u_left * v_right,              //
          v_left * v_right,              //
          v_right;                       //
    }

    Eigen::JacobiSVD<Eigen::Matrix<value_type, Eigen::Dynamic, 9>> svd{
        matA, Eigen::ComputeThinV
    };
    Eigen::Vector<value_type, 9> matA_svd_result{
        svd.matrixV().col(svd.matrixV().cols() - 1)
    };
    Eigen::Map<Eigen::Matrix<value_type, 3, 3>> matH{matA_svd_result.data(), 3,
                                                     3};

    // 各向同性逆归一化
    Eigen::Matrix<value_type, 3, 3> homography_matrix{
        config.T_right_.transpose() * matH * config.T_left_,
    };

    return homography_matrix;
  }

  static auto DecomposeEssentialMatrix(
      const Eigen::Matrix<value_type, 3, 3> &essential_matrix
  )
  {
    const Eigen::Matrix<value_type, 3, 3> matW{
        {0.0, -1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const Eigen::Matrix<value_type, 3, 3> matZ{
        {0.0, 1.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    };
    // 本质矩阵 E 可以分解为 T 的叉乘矩阵与 R 的乘积
    // T 的叉乘矩阵可以分解为 U * Z * U^T (相差一个常数的情况下等式成立)
    // 其中 U 是单位正交矩阵 (行列式等于1)
    // 而 Z = diag(1,1,0) * W^T (相差一个常数的情况下等式成立)
    // 故 T_antisym = U * diag(1,1,0) * W^T * U^T
    // 或           = U * diag(1,1,0) * W * U^T

    Eigen::JacobiSVD<Eigen::Matrix<value_type, 3, 3>> svd1{
        essential_matrix,
        Eigen::ComputeFullU | Eigen::ComputeFullV,
    };
    const Eigen::Matrix<value_type, 3, 3> matU{svd1.matrixU()};
    const Eigen::Matrix<value_type, 3, 3> matV{svd1.matrixV()};

    // matV^T = W * U^T * R
    // R = U * W^T * matV^T
    Eigen::Matrix<value_type, 3, 3> matR1{matU * matW.transpose()
                                          * matV.transpose()};
    matR1 *= matR1.determinant();

    Eigen::Matrix<value_type, 3, 3> matR2{matU * matW * matV.transpose()};
    matR2 *= matR2.determinant();

    Eigen::JacobiSVD<Eigen::Matrix<value_type, 3, 3>> svd2{
        matU * matZ * matU.transpose(),
        Eigen::ComputeFullV,
    };
    Eigen::Vector<value_type, 3> vecT{
        svd2.matrixV().col(svd2.matrixV().cols() - 1),
    };

    // TODO
    // 选择一对特征点，进行三角化，理论上正确的一组解 (R, T) 可以保证对应的路标点在两个相机坐标系下的 Z 坐标都是正数
    // 为了让算法更鲁邦，还可以对多对点进行三角化，选择在两个相机坐标系啊下 Z 坐标为正数的点对个数最多的一组解 (R, T)

    return std::make_tuple(matR1, matR2, vecT);
  }
};
