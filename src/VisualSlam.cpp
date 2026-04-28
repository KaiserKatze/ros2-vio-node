#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

#include "euroc_vio/EuRoC.hpp"

struct ImageData
{
  std::uint64_t timestamp_; // in nanoseconds
  std::filesystem::path image_path_;
};

template <typename T> T get_item(std::stringstream &ss);

std::string_view trim(std::string_view str)
{
  const auto first = str.find_first_not_of(" \t\n\r\v\f");
  if (first == std::string_view::npos)
  {
    return {};
  }

  const auto last = str.find_last_not_of(" \t\n\r\v\f");
  return str.substr(first, last - first + 1);
}

template <> std::string get_item<std::string>(std::stringstream &ss)
{
  std::string item;
  std::getline(ss, item, ',');
  return std::string{trim(item)};
}

template <> std::uint64_t get_item<std::uint64_t>(std::stringstream &ss)
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

cv::Mat ReadImage(const std::filesystem::path &image_path)
{
  return cv::imread(image_path.string());
}

std::vector<ImageData> ReadIndex(const std::filesystem::path &index_path,
                                 const std::filesystem::path &dir_path)
{
  std::ifstream file(index_path);
  std::string line;

  std::vector<ImageData> data;
  data.reserve(32767);

  // 跳过表头
  std::getline(file, line);
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    auto timestamp{
        get_item<std::uint64_t>(ss), // in nanoseconds
    };
    auto image_name{
        get_item<std::string>(ss),
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

struct StereoFrame
{
  double timestamp_; // in seconds
  std::filesystem::path image_left_path_;
  std::filesystem::path image_right_path_;
};

std::vector<StereoFrame> Process(std::vector<ImageData> &&series_cam0,
                                 std::vector<ImageData> &&series_cam1)
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
    ImageData &image0{*itr0};
    ImageData &image1{*itr1};
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
  std::vector<StereoFrame> result;
  result.reserve(series_cam0.size());
  for (auto itr0 = series_cam0.begin(), itr1 = series_cam1.begin();
       itr0 != series_cam0.end() && itr1 != series_cam1.end(); ++itr0, ++itr1)
  {
    ImageData &image0{*itr0};
    ImageData &image1{*itr1};
    double timestamp{1e-9 * image0.timestamp_};
    std::cerr << "\t{timestamp=" << std::fixed << std::setprecision(6)
              << timestamp << ", "
              << "image0=" << image0.image_path_ << ", "
              << "image1=" << image1.image_path_ << "}\n";
    result.emplace_back(timestamp, std::move(image0.image_path_),
                        std::move(image1.image_path_));
  }
  return result;
}

struct AbstractDetector
{
  static constexpr size_t minCorners{5};
  static constexpr int maxCorners{100};
  static constexpr double qualityLevel{0.3};
  static constexpr double minDistance{7.0};
  static constexpr double blockSize{7.0};
  static constexpr bool useHarrisDetector{false};
  static constexpr double freeParamHarisDetector{0.04};
  static constexpr double atol_parallax{2.0};
  static constexpr double atol_coincidence{1.0};
  const cv::Size winSize{15, 15};
  static constexpr int maxLevel{2};

  virtual ~AbstractDetector() = 0;
};

struct TomasiDetector : public AbstractDetector
{
  void InitCorners(const cv::Mat &grayscale_image,
                   std::vector<cv::Point2f> &corners) const
  {
    // https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    cv::goodFeaturesToTrack(grayscale_image, corners, maxCorners, qualityLevel,
                            minDistance, cv::noArray(), blockSize,
                            useHarrisDetector, freeParamHarisDetector);
  }

  auto FilterWithStatus(std::vector<cv::Point2f> &pts,
                        std::vector<unsigned char> const &status) const
  {
    std::vector<cv::Point2f> result;
    for (size_t i{0}; i < pts.size(); ++i)
    {
      if (status[i] == 1)
      {
        result.push_back(pts[i]);
      }
    }
    pts = result;
  }
};

struct StereoSlam
{
  EuRoC euroc_{};
  std::vector<StereoFrame> stereo_frames_;

  StereoSlam()
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

    stereo_frames_ = Process(ReadIndex(path_cam0_data, path_cam0_image_dir),
                             ReadIndex(path_cam1_data, path_cam1_image_dir));
    std::cerr << "[INFO] 成功加载立体视觉图片索引文件, "
                 "待读取图片张数: "
              << stereo_frames_.size() << ".\n";

    for (size_t index{0}; StereoFrame &frame : stereo_frames_)
    {
      std::cerr << "[INFO] 正在处理第 " << ++index << " 张图片 ...\n";
      auto [image_left_rectified, image_right_rectified]
          = euroc_.remap(cv::imread(frame.image_left_path_),
                         cv::imread(frame.image_right_path_));
      auto [image_left_grayscale, image_right_grayscale]
          = euroc_.grayscale(image_left_rectified, image_right_rectified);
    }
  }
};

int main()
{
  return 0;
}
