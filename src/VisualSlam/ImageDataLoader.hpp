#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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
    static const std::filesystem::path path_home{
        std::getenv("HOME"),
    };
    static const std::filesystem::path path_mav0{
        // R"(/mnt/e/Documents/mav0),
        path_home / "EuRoC_MAV_Datasets" / "V2_01_easy" / "mav0",
    };
    static const std::filesystem::path path_cam0_data{
        path_mav0 / "cam0" / "data.csv",
    };
    static const std::filesystem::path path_cam1_data{
        path_mav0 / "cam1" / "data.csv",
    };
    static const std::filesystem::path path_cam0_image_dir{
        path_mav0 / "cam0" / "data",
    };
    static const std::filesystem::path path_cam1_image_dir{
        path_mav0 / "cam1" / "data",
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
