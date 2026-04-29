#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

namespace EuRoC
{

struct EuRoC
{
  static constexpr double fu0{458.654};
  static constexpr double fv0{457.296};
  static constexpr double cu0{367.215};
  static constexpr double cv0{248.375};
  static constexpr double k01{-0.28340811};
  static constexpr double k02{0.07395907};
  static constexpr double p01{0.00019359};
  static constexpr double p02{1.76187114e-05};
  static constexpr double fu1{457.587};
  static constexpr double fv1{456.134};
  static constexpr double cu1{379.999};
  static constexpr double cv1{255.238};
  static constexpr double k11{-0.28368365};
  static constexpr double k12{0.07451284};
  static constexpr double p11{-0.00010473};
  static constexpr double p12{-3.55590700e-05};
  static constexpr int image_width{752};
  static constexpr int image_height{480};

  // https://libeigen.gitlab.io/eigen/docs-3.1/TopicStructHavingEigenMembers.html
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Eigen::Matrix3d rectifiedCameraMatrix0;
  // Eigen::Matrix3d rectifiedCameraMatrix1;
  Eigen::Matrix4d T_C1C0;
  cv::Size imageSize{752, 480};
  cv::Mat map0x;
  cv::Mat map0y;
  cv::Mat map1x;
  cv::Mat map1y;

  EuRoC()
  {
    // 1. 初始化矩阵（确保使用 double 类型）

    // https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html
    // https://docs.opencv.org/4.x/d0/daf/group__core__eigen.html

    cv::Mat cameraMatrix0;
    cv::eigen2cv(
        Eigen::Matrix3d{
            {fu0, 0.0, cu0},
            {0.0, fv0, cv0},
            {0.0, 0.0, 1.0},
        },
        cameraMatrix0);
    cv::Mat distCoeffs0;
    cv::eigen2cv(
        Eigen::Vector4d{
            k01,
            k02,
            p01,
            p02,
        },
        distCoeffs0);

    cv::Mat cameraMatrix1;
    cv::eigen2cv(
        Eigen::Matrix3d{
            {fu1, 0.0, cu1},
            {0.0, fv1, cv1},
            {0.0, 0.0, 1.0},
        },
        cameraMatrix1);
    cv::Mat distCoeffs1;
    cv::eigen2cv(
        Eigen::Vector4d{
            k11,
            k12,
            p11,
            p12,
        },
        distCoeffs1);

    const Eigen::Matrix4d T_BC0{
        {0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975},
        {0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768},
        {-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949},
        {0.0, 0.0, 0.0, 1.0},
    };

    const Eigen::Matrix4d T_BC1{
        {0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556},
        {0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024},
        {-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038},
        {0.0, 0.0, 0.0, 1.0},
    };

    // X_B = T_BC0 * X_C0
    // X_B = T_BC1 * X_C1
    // X_C1 = T_C1C0 * X_C0 = (T_BC1.inverse() * T_BC0) * X_C0
    T_C1C0 = T_BC1.inverse() * T_BC0;

    // 2. 使用更安全的方法提取 R 和 T

    // Rotation matrix from the coordinate system of the first camera to the second camera
    cv::Mat stereoR(3, 3, CV_64FC1);
    cv::eigen2cv(
        // 提取左上角 3x3 矩阵作为旋转矩阵
        Eigen::Matrix3d{T_C1C0(Eigen::seq(0, 2), Eigen::seq(0, 2))}, stereoR);

    // Translation vector from the coordinate system of the first camera to the second camera
    cv::Mat stereoT(3, 1, CV_64FC1);
    cv::eigen2cv(
        // 提取第 4 列的前 3 行作为平移向量
        Eigen::Vector3d{T_C1C0(Eigen::seq(0, 2), Eigen::seq(3, 3))}, stereoT);

    // 3. 调用立体校正

    // Output 3x3 rectification transform (rotation matrix) for the first camera.
    // This matrix brings points given in the unrectified first camera's
    // coordinate system to points in the rectified first camera's coordinate
    // system. In more technical terms, it performs a change of basis from the
    // unrectified first camera's coordinate system to the rectified first
    // camera's coordinate system
    cv::Mat R0;
    // Output 3x3 rectification transform (rotation matrix) for the second camera.
    // This matrix brings points given in the unrectified second camera's
    // coordinate system to points in the rectified second camera's coordinate
    // system. In more technical terms, it performs a change of basis from the
    // unrectified second camera's coordinate system to the rectified second
    // camera's coordinate system
    cv::Mat R1;
    // Output 3x4 projection matrix in the new (rectified) coordinate systems for
    // the first camera, i.e. it projects points given in the rectified first
    // camera coordinate system into the rectified first camera's image
    cv::Mat P0;
    // Output 3x4 projection matrix in the new (rectified) coordinate systems for
    // the second camera, i.e. it projects points given in the rectified first
    // camera coordinate system into the rectified second camera's image
    cv::Mat P1;
    // Output 4×4 disparity-to-depth mapping matrix
    cv::Mat Q;

    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    cv::stereoRectify(cameraMatrix0, distCoeffs0, cameraMatrix1, distCoeffs1,
                      imageSize, stereoR, stereoT, R0, R1, P0, P1, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);

    // 4. 初始化映射表

    // https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga0d37b45f780b32f63ed19c21aa9fd333
    cv::initUndistortRectifyMap(cameraMatrix0, distCoeffs0, R0, P0, imageSize,
                                CV_32FC1, map0x, map0y);
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize,
                                CV_32FC1, map1x, map1y);

    // cv::cv2eigen(P0, rectifiedCameraMatrix0);
    // cv::cv2eigen(P1, rectifiedCameraMatrix1);

    printf("EuRoC setup done\n");
  }

  EuRoC(const EuRoC &)            = delete;
  EuRoC &operator=(const EuRoC &) = delete;
  EuRoC(EuRoC &&)                 = delete;
  EuRoC &operator=(EuRoC &&)      = delete;

  std::pair<cv::Mat, cv::Mat> remap(const cv::Mat &image0,
                                    const cv::Mat &image1) const
  {
    // printf("Left Image Size: %d x %d\n", image0.size().width, image0.size().height);
    // printf("Right Image Size: %d x %d\n", image1.size().width, image1.size().height);
    cv::Mat rectified0;
    cv::Mat rectified1;
    // https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
    cv::remap(image0, rectified0, map0x, map0y, cv::INTER_LINEAR);
    cv::remap(image1, rectified1, map1x, map1y, cv::INTER_LINEAR);
    return std::make_pair(rectified0, rectified1);
  }

  std::pair<cv::Mat, cv::Mat> grayscale(const cv::Mat &rectified0,
                                        const cv::Mat &rectified1) const
  {
    cv::Mat gray0;
    cv::Mat gray1;
    // https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    cv::cvtColor(rectified0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectified1, gray1, cv::COLOR_BGR2GRAY);
    return std::make_pair(gray0, gray1);
  }
};

} // namespace EuRoC

template <typename E = std::filesystem::path> struct StereoFrame
{
  double timestamp_; // in seconds
  E image_left_;
  E image_right_;
};

struct ImageDataLoader
{
private:
  struct ImageIndexEntry
  {
    std::uint64_t timestamp_; // in nanoseconds
    std::filesystem::path image_path_;
  };

  static std::string_view trim(std::string_view str)
  {
    const auto first = str.find_first_not_of(" \t\n\r\v\f");
    if (first == std::string_view::npos)
    {
      return {};
    }

    const auto last = str.find_last_not_of(" \t\n\r\v\f");
    return str.substr(first, last - first + 1);
  }

  static std::string get_item_as_string(std::stringstream &ss)
  {
    std::string item;
    std::getline(ss, item, ',');
    return std::string{trim(item)};
  }

  static std::uint64_t get_item_as_uint64(std::stringstream &ss)
  {
    std::string item;
    std::getline(ss, item, ',');
    auto sv{trim(item)};
    std::uint64_t result{0};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse uint64: " + std::string(sv)};
    }
    return result;
  }

  static std::vector<ImageIndexEntry>
  ReadIndex(const std::filesystem::path &index_path,
            const std::filesystem::path &dir_path)
  {
    std::ifstream file(index_path);
    std::string line;

    std::vector<ImageIndexEntry> data;
    data.reserve(32767);

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      auto timestamp{
          get_item_as_uint64(ss), // in nanoseconds
      };
      auto image_name{
          get_item_as_string(ss),
      };
      if (!image_name.ends_with(".png"))
      {
        throw std::runtime_error{
            "Assertion Error: file name not ends with '.png'!"};
      }
      data.emplace_back(timestamp, dir_path / image_name);
    }
    return data;
  }

  static std::vector<StereoFrame<>>
  MergeIndex(std::vector<ImageIndexEntry> &&series_cam0,
             std::vector<ImageIndexEntry> &&series_cam1)
  {
    if (series_cam0.size() != series_cam1.size())
    {
      std::stringstream ss;
      ss << "Invalid vector size! "
            "Left Camera took "
         << series_cam0.size()
         << " photos, "
            "Right Camera took "
         << series_cam1.size() << " photos.";
      throw std::invalid_argument{ss.str()};
    }
    for (auto itr0 = series_cam0.begin(), itr1 = series_cam1.begin();
         itr0 != series_cam0.end() && itr1 != series_cam1.end(); ++itr0, ++itr1)
    {
      ImageIndexEntry &image0{*itr0};
      ImageIndexEntry &image1{*itr1};
      if (image0.timestamp_ != image1.timestamp_)
      {
        std::stringstream ss;
        ss << "Invalid timestamp! "
              "Left Camera ("
           << image0.timestamp_
           << "), "
              "Right Camera ("
           << image1.timestamp_ << ").";
        throw std::runtime_error{ss.str()};
      }
    }
    std::vector<StereoFrame<>> result;
    result.reserve(series_cam0.size());
    for (auto itr0 = series_cam0.begin(), itr1 = series_cam1.begin();
         itr0 != series_cam0.end() && itr1 != series_cam1.end(); ++itr0, ++itr1)
    {
      ImageIndexEntry &image0{*itr0};
      ImageIndexEntry &image1{*itr1};
      double timestamp{1e-9 * image0.timestamp_};
      // std::cerr << "\t{timestamp=" << std::fixed << std::setprecision(6)
      //           << timestamp << ", "
      //           << "image0=" << image0.image_path_ << ", "
      //           << "image1=" << image1.image_path_ << "}\n";
      result.emplace_back(timestamp, std::move(image0.image_path_),
                          std::move(image1.image_path_));
    }
    return result;
  }

  static std::vector<StereoFrame<>> Load()
  {
    static const std::filesystem::path path_cam0_data{
        R"(/mnt/e/Documents/mav0/cam0/data.csv)",
    };
    static const std::filesystem::path path_cam1_data{
        R"(/mnt/e/Documents/mav0/cam1/data.csv)",
    };
    static const std::filesystem::path path_cam0_image_dir{
        R"(/mnt/e/Documents/mav0/cam0/data)",
    };
    static const std::filesystem::path path_cam1_image_dir{
        R"(/mnt/e/Documents/mav0/cam1/data)",
    };

    auto stereo_frames
        = MergeIndex(ReadIndex(path_cam0_data, path_cam0_image_dir),
                     ReadIndex(path_cam1_data, path_cam1_image_dir));
    std::cerr << "[INFO] 成功加载立体视觉图片索引文件, "
                 "待读取图片张数: "
              << stereo_frames.size() << ".\n";

    return stereo_frames;
  }

  /**
   * @brief 按照 BGR 格式（蓝、绿、红顺序）读取图像
   * @note 函数 cv::imread 的参数 cv::ImreadModes::IMREAD_COLOR 对应函数 cv::cvtColor 的参数 cv::COLOR_BGR2GRAY
   */
  static cv::Mat ReadImage(const std::filesystem::path &image_path)
  {
    // https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gaffb68fce322c6e52841d7d9357b9ad2d
    return cv::imread(image_path.string(), cv::ImreadModes::IMREAD_COLOR);
  }

private:
  std::vector<StereoFrame<>> stereo_frames_{Load()};
  using const_iterator = typename decltype(stereo_frames_)::const_iterator;
  const_iterator iterator{stereo_frames_.cbegin()};

public:
  explicit operator bool() const
  {
    return this->iterator != this->stereo_frames_.cend();
  }

  StereoFrame<cv::Mat> operator()()
  {
    StereoFrame<cv::Mat> result{
        iterator->timestamp_,
        ReadImage(iterator->image_left_),
        ReadImage(iterator->image_right_),
    };
    return result;
  }

  bool operator++()
  {
    if (this->iterator != this->stereo_frames_.cend())
    {
      ++this->iterator;
      return true;
    }
    return false;
  }

  bool operator--()
  {
    if (this->iterator != this->stereo_frames_.cbegin())
    {
      --this->iterator;
      return true;
    }
    return false;
  }

  size_t Rewind(size_t index = 0)
  {
    size_t size{stereo_frames_.size()};
    index    = std::min(index, size - 1);
    iterator = stereo_frames_.cbegin() + index;
    return index;
  }
};

namespace CornerDetection
{

struct AbstractDetector
{
  static constexpr size_t minCorners{10};
  static constexpr int maxCorners{100};
  // Quality level (percent of maximum)
  static constexpr double qualityLevel{0.01};
  // Min distance between corners
  static constexpr double minDistance{5.0};
  // Maximum pyramid level to construct
  static constexpr double blockSize{3.0};
  // true: Harris, false: Shi-Tomasi
  static constexpr bool useHarrisDetector{false};
  static constexpr double freeParamHarisDetector{0.04};
  static constexpr double atol_parallax{1.0};
  static constexpr double atol_coincidence{1.0};
  const cv::Size winSize{15, 15};
  static constexpr int maxLevel{2};
};

template <typename PointType = cv::Point2f>
struct TomasiDetector : public AbstractDetector
{
private:
  const cv::TermCriteria criteria_{
      (cv::TermCriteria::COUNT) | (cv::TermCriteria::EPS),
      // Maximum number of iterations
      20,
      // Minimum change per iteration
      0.3,
  };

public:
  bool FindCorners(const cv::Mat &gray_left, const cv::Mat &gray_right,
                   std::vector<PointType> &corners_left,
                   std::vector<PointType> &corners_right) const
  {
    // https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    cv::goodFeaturesToTrack(gray_left, corners_left, maxCorners, qualityLevel,
                            minDistance, cv::noArray(), blockSize,
                            useHarrisDetector, freeParamHarisDetector);

    if (corners_left.size() < minCorners)
    {
      return false;
    }

    std::vector<unsigned char> features_found;

    // https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    cv::calcOpticalFlowPyrLK(gray_left, gray_right, corners_left, corners_right,
                             features_found, cv::noArray(), winSize, maxLevel,
                             criteria_);

    // 压缩数据
    {
      std::cerr << "\t筛选前，角点个数 = " << corners_left.size() << "\n";

      // 将左目点、右目点、状态位三者打包
      auto zipped_view
          = std::views::zip(corners_left, corners_right, features_found)
            | std::views::filter(
                [atol = atol_parallax](const auto &tuple)
                {
                  const auto &[p_left, p_right, found] = tuple;
                  // 1. 必须是追踪成功的点
                  // 2. 视差过滤：保证正视差 (即左目图像中的点的横坐标必须大于右目图像中的点的横坐标)
                  // 3. 极线过滤：纵坐标之差必须小于阈值
                  return found && (p_left.x > p_right.x)
                         && (std::abs(p_left.y - p_right.y) < atol);
                });

      // 因为 view 是延迟计算的，所以必须先创建副本，绝对不能使用 std::vector::assign 方法进行赋值
      std::vector<PointType> new_corners_left;
      std::vector<PointType> new_corners_right;
      for (const auto &[point_left, point_right, found] : zipped_view)
      {
        new_corners_left.push_back(point_left);
        new_corners_right.push_back(point_right);
      }

      corners_left  = std::move(new_corners_left);
      corners_right = std::move(new_corners_right);

      std::cerr << "\t筛选后，角点个数 = " << corners_left.size() << "\n";
    }

    if (corners_right.size() < minCorners)
    {
      return false;
    }

    return true;
  }
};

} // namespace CornerDetection

namespace ProjectiveGeometry
{

void Homo2Nonhomo(const Eigen::Matrix3Xd &homo, Eigen::Matrix2Xd &nonhomo)
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

void Nonhomo2Homo(const Eigen::Matrix2Xd &nonhomo, Eigen::Matrix3Xd &homo)
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

template <size_t nk = 2, size_t np = 2, typename E = double>
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
          "DistortFunctionBrownConrady::Distort: points must be 3xN.");
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
Eigen::Matrix2Xd Project(const Eigen::Matrix3Xd &modelPointsInWorldCoordinates,
                         const Eigen::Matrix3d &iMat,
                         const Eigen::Matrix3d &rMat,
                         const Eigen::Vector3d &tVec,
                         const DistortFunction &distortFunction)
{
  if (modelPointsInWorldCoordinates.rows() != 3)
  {
    throw std::runtime_error(
        "Project: modelPointsInWorldCoordinates must be 3xN.");
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

Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d &vec)
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

} // namespace ProjectiveGeometry

namespace EightPointAlgorithm
{

/**
 * @brief 各向同性逆归一化
 * @return 各向同性归一化变换矩阵
 */
Eigen::Matrix3d IsotropicScalingNormalize(Eigen::Matrix3Xd &points)
{
  if (points.rows() != 3)
  {
    throw std::runtime_error("IsotropicScalingNormalize: points must be 3xN.");
  }
  static const double tiny = 1e-8;

  double centroidX = points.row(0).mean();
  double centroidY = points.row(1).mean();

  Eigen::Matrix2Xd centeredPoints = points.topRows(2);
  centeredPoints.row(0).array() -= centroidX;
  centeredPoints.row(1).array() -= centroidY;
  Eigen::VectorXd distances = centeredPoints.colwise().norm();
  double meanDistance       = distances.mean();

  double scale = (meanDistance < tiny) ? 1.0 : (std::sqrt(2.0) / meanDistance);

  Eigen::Matrix3d sMat;
  sMat << scale, 0.0, -scale * centroidX, 0.0, scale, -scale * centroidY, 0.0,
      0.0, 1.0;

  points = sMat * points;
  return sMat;
}

struct TriangulationConfig
{
  // 左目相机内参矩阵
  Eigen::Matrix3d mat_cam_left_;
  // 右目相机内参矩阵
  Eigen::Matrix3d mat_cam_right_;
  // 从左目相机到右目相机的旋转
  Eigen::Matrix3d rotation_;
  // 从左目相机到游牧相机的平移
  Eigen::Vector3d translation_;
  // 路标点在左目像平面上的投影点的齐次坐标
  Eigen::Matrix3Xd pixel_left_;
  // 路标点在右目像平面上的投影点的齐次坐标
  Eigen::Matrix3Xd pixel_right_;
  // 左目视图的各向同性归一化变换
  Eigen::Matrix3d T_left_;
  // 右目视图的各向同性归一化变换
  Eigen::Matrix3d T_right_;

  TriangulationConfig(const Eigen::Matrix3d &mat_cam_left,
                      const Eigen::Matrix3d &mat_cam_right,
                      const Eigen::Matrix3d &rotation,
                      const Eigen::Vector3d &translation,
                      const Eigen::Matrix3Xd &pixel_left,
                      const Eigen::Matrix3Xd &pixel_right) :
    mat_cam_left_{mat_cam_left}, mat_cam_right_{mat_cam_right},
    rotation_{rotation}, translation_{translation}, pixel_left_{pixel_left},
    pixel_right_{pixel_right}, T_left_{IsotropicScalingNormalize(pixel_left_)},
    T_right_{IsotropicScalingNormalize(pixel_right_)}
  {
  }

  Eigen::Vector3d GetLeftEpipole() const
  {
    return mat_cam_left_ * rotation_.transpose() * translation_;
  }

  Eigen::Vector3d GetRightEpipole() const
  {
    return mat_cam_right_ * translation_;
  }

  /**
   * @brief 利用两个相机内参矩阵、旋转矩阵、平移向量，计算基础矩阵
   */
  Eigen::Matrix3d ComputeFundamentalMatrix() const
  {
    // 左目视图中极点的齐次坐标
    const Eigen::Vector3d epipole_left{GetLeftEpipole()};
    // 左极点齐次坐标的叉乘矩阵
    Eigen::Matrix3d epipole_left_antisym;
    epipole_left_antisym << 0.0, -epipole_left.z(), epipole_left.y(), //
        epipole_left.z(), 0.0, -epipole_left.x(),                     //
        -epipole_left.y(), epipole_left.x(), 0.0;
    return mat_cam_right_.transpose().inverse() * rotation_
           * mat_cam_left_.transpose() * epipole_left_antisym;
  }
};

/**
 * @brief 三角化
 * @return 路标点的齐次坐标
 */
Eigen::Matrix4Xd Triangulate(const TriangulationConfig &&config)
{
  if (config.pixel_left_.cols() != config.pixel_right_.cols())
  {
    throw std::runtime_error(
        "Triangulate: number of pixel points in two views must match.");
  }

  Eigen::Matrix34d projectMatrix_left;
  projectMatrix_left << config.mat_cam_left_, Eigen::Vector3d::Zero();

  const auto m1_left{projectMatrix_left.row(0)};
  const auto m2_left{projectMatrix_left.row(1)};
  const auto m3_left{projectMatrix_left.row(2)};

  Eigen::Matrix34d projectMatrix_right;
  projectMatrix_right << config.rotation_, config.translation_;
  projectMatrix_right = config.mat_cam_right_ * projectMatrix_right;

  const auto m1_right{projectMatrix_right.row(0)};
  const auto m2_right{projectMatrix_right.row(1)};
  const auto m3_right{projectMatrix_right.row(2)};

  auto nCols{config.pixel_left_.cols()};
  Eigen::Matrix4Xd matA(4, 4 * nCols);
  matA.setZero();
  for (decltype(nCols) i = 0; i < nCols; ++i)
  {
    const double u_left{config.pixel_left_(0, i)};
    const double v_left{config.pixel_left_(1, i)};
    const double u_right{config.pixel_right_(0, i)};
    const double v_right{config.pixel_right_(1, i)};

    matA(Eigen::all, Eigen::seqN(4 * i, 4)) //
        << u_left * m3_left - m1_left,
        v_left * m3_left - m2_left, u_right * m3_right - m1_right,
        v_right * m3_right - m2_right;
  }

  Eigen::JacobiSVD<decltype(matA)> svd{matA, Eigen::ComputeThinV};
  Eigen::VectorXd matA_svd_result{svd.matrixV().col(svd.matrixV().cols() - 1)};
  Eigen::Map<Eigen::Matrix4Xd> modelPointsInWorldCoordinates{
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
Eigen::Matrix3d EstimateFundamentalMatrix(const TriangulationConfig &&config)
{
  auto nCols{config.pixel_left_.cols()};
  Eigen::MatrixX9d matW(nCols, 9);
  matW.setZero();
  for (decltype(nCols) i = 0; i < nCols; ++i)
  {
    const double u_left{config.pixel_left_(0, i)};
    const double v_left{config.pixel_left_(1, i)};
    const double u_right{config.pixel_right_(0, i)};
    const double v_right{config.pixel_right_(1, i)};
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
  Eigen::JacobiSVD<decltype(matW)> svd1{matW, Eigen::ComputeThinV};
  Eigen::VectorXd matW_svd_result{
      svd1.matrixV().col(svd1.matrixV().cols() - 1)};

  // 进行第二次奇异值分解（保证基础矩阵的秩为2）
  Eigen::JacobiSVD<Eigen::Matrix3d> svd2{
      Eigen::Map<Eigen::Matrix3d>{matW_svd_result.data(), 3, 3},
      Eigen::ComputeFullU | Eigen::ComputeFullV,
  };
  Eigen::Vector3d singularValues{svd2.singularValues()};
  singularValues[2] = 0.0; // 将最小奇异值置零
  Eigen::Matrix3d fundamental_matrix{
      svd2.matrixU() * singularValues.asDiagonal() * svd2.matrixV().transpose(),
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
Eigen::Matrix3d EstimateHomography(const TriangulationConfig &&config)
{
  auto nCols{config.pixel_left_.cols()};
  Eigen::MatrixX9d matA(2 * nCols, 9);

  matA.setZero();
  for (decltype(nCols) i = 0; i < nCols; ++i)
  {
    const double u_left{config.pixel_left_(0, i)};
    const double v_left{config.pixel_left_(1, i)};
    const double u_right{config.pixel_right_(0, i)};
    const double v_right{config.pixel_right_(1, i)};
    Eigen::RowVector3d zero_vector{Eigen::RowVector3d::Zero()};
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

  Eigen::JacobiSVD<Eigen::MatrixX9d> svd{matA, Eigen::ComputeThinV};
  Eigen::Vector9d matA_svd_result{svd.matrixV().col(svd.matrixV().cols() - 1)};
  Eigen::Map<Eigen::Matrix3d> matH{matA_svd_result.data(), 3, 3};

  // 各向同性逆归一化
  Eigen::Matrix3d homography_matrix{
      config.T_right_.transpose() * matH * config.T_left_,
  };

  return homography_matrix;
}

} // namespace EightPointAlgorithm

struct StereoSlam
{
public:
  const std::string window_name_{"VisualSlam"};
  const EuRoC::EuRoC euroc_{};

private:
  ImageDataLoader loader_{};
  CornerDetection::TomasiDetector<> detector_{};
  size_t frame_index_{};

public:
  StereoSlam()
  {
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    // cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
    //                       cv::WINDOW_FULLSCREEN);
  }

  ~StereoSlam()
  {
    cv::destroyAllWindows();
  }

  void StartVisualization()
  {
    while (true)
    {
      KeyEvent event{Visualize()};
      switch (event)
      {
      case KeyEvent::NOOP:
        break;
      case KeyEvent::EXIT:
        return;
      case KeyEvent::NEXT:
        ++loader_;
        ++frame_index_;
        break;
      case KeyEvent::PREV:
        --loader_;
        if (frame_index_ > 0)
        {
          --frame_index_;
        }
        break;
      }
    }
  }

private:
  enum class KeyEvent
  {
    EXIT,
    NEXT,
    PREV,
    NOOP
  };

  KeyEvent InterpretKeyEvent()
  {
    size_t digit{0};

    while (true)
    {
      // https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#gafa15c0501e0ddd90918f17aa071d3dd0
      const auto key{cv::waitKey(0) & 0xFF};
      if ('0' <= key && key <= '9')
      {
        digit = 10 * digit + (key - '0');
        continue;
      }

      // 处理回车键
      if (key == 13)
      {
        frame_index_ = loader_.Rewind(digit);
        return KeyEvent::NOOP;
      }
      else if (key == 's' || key == 'S')
      {
        // SaveStereoFrame(frame);
        return KeyEvent::NOOP;
      }
      else if (key == 27 || key == 'q' || key == 'Q')
      {
        return KeyEvent::EXIT;
      }
      else if (key == 'd' || key == 'D')
      {
        return KeyEvent::NEXT;
      }
      else if (key == 'a' || key == 'A')
      {
        return KeyEvent::PREV;
      }

      break;
    }

    return KeyEvent::NOOP;
  }

  static std::vector<cv::Scalar> GenerateRandomColors(int count_colors)
  {
    cv::Mat m{count_colors, 1, CV_8UC3};
    // https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0
    cv::randu(m, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    std::vector<cv::Scalar> result;
    result.reserve(count_colors);
    for (int i = 0; i < count_colors; ++i)
    {
      auto v = m.at<cv::Vec3b>(i);
      result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
  }

  const size_t count_colors_{255};
  const std::vector<cv::Scalar> colors_{GenerateRandomColors(count_colors_)};

  template <typename PointType = cv::Point2f>
  void PlotFlow(cv::Mat &flow, std::vector<PointType> const &pts0,
                std::vector<PointType> const &pts1, cv::Size offset0,
                cv::Size offset1) const
  {
    for (size_t index = 0; index < pts0.size(); ++index)
    {
      cv::Point2f pt0{pts0[index]};
      cv::Point2f pt1{pts1[index]};
      pt0.x += offset0.width;
      pt0.y += offset0.height;
      pt1.x += offset1.width;
      pt1.y += offset1.height;
      const cv::Scalar lineColor{colors_[index % count_colors_]};
      const int lineThickness{2};
      cv::line(flow, pt0, pt1, lineColor, lineThickness);
    }
  }

  KeyEvent Visualize()
  {
    if (!loader_)
    {
      return KeyEvent::EXIT;
    }

    StereoFrame<cv::Mat> frame{loader_()};

    std::cerr << "[INFO] 正在处理第 " << frame_index_ << " 张图片 ...\n";

    auto [image_left_rectified, image_right_rectified]
        = euroc_.remap(frame.image_left_, frame.image_right_);
    auto [image_left_grayscale, image_right_grayscale]
        = euroc_.grayscale(image_left_rectified, image_right_rectified);

    std::vector<cv::Point2f> corners_left, corners_right;
    detector_.FindCorners(image_left_grayscale, image_right_grayscale,
                          corners_left, corners_right);

    std::cerr << "\t检测到 " << corners_left.size() << " 个角点 ...\n";

    // cv::Size imageSize{752, 480};
    cv::Mat vis_top, vis_bottom, vis;

    // https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gaab5ceee39e0580f879df645a872c6bf7
    cv::hconcat(frame.image_left_, frame.image_right_, vis_top);

    // std::cerr << "\t上两张图片大小 = (" << vis_top.cols << " * " << vis_top.rows
    //           << ")\n"
    //           << "\t上两张图片维数 = " << vis_top.dims << "\n";

    cv::hconcat(image_left_grayscale, image_right_grayscale, vis_bottom);

    // std::cerr << "\t下两张图片大小 = (" << vis_bottom.cols << " * "
    //           << vis_bottom.rows << ")\n"
    //           << "\t下两张图片维数 = " << vis_top.dims << "\n";

    cv::cvtColor(vis_bottom, vis_bottom, cv::COLOR_GRAY2BGR);

    {
      const cv::Size flowSize{vis_bottom.size()};
      cv::Mat flow{cv::Mat::zeros(flowSize, image_left_rectified.type())};
      const cv::Size imageSize{image_left_rectified.size()};
      PlotFlow(flow, corners_left, corners_right, cv::Size{0, 0},
               cv::Size{imageSize.width, 0});
      cv::add(flow, vis_bottom, vis_bottom);
    }

    cv::vconcat(vis_top, vis_bottom, vis);

    cv::imshow(window_name_, vis);

    {
      std::stringstream ss_window_title;
      ss_window_title << "Image Frame [#" << std::setw(4) << std::setfill('0')
                      << frame_index_ << "]";
      cv::setWindowTitle(window_name_, ss_window_title.str());
    }

    return InterpretKeyEvent();
  }

  void SaveStereoFrame(const StereoFrame<cv::Mat> &frame) const
  {
    std::stringstream ss_file_name;
    ss_file_name << "frame_" << std::dec << std::setw(4) << std::setfill('0')
                 << frame_index_;
    std::string file_name{ss_file_name.str()};

    // https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    cv::imwrite(file_name + "_left.png", frame.image_left_);
    cv::imwrite(file_name + "_right.png", frame.image_right_);
  }
};

int main()
{
  StereoSlam slam{};
  slam.StartVisualization();
  return 0;
}
