#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
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

#include "euroc_vio/AbstractLoader.hpp"

// 单目视觉帧
template <typename E = std::filesystem::path>
struct MonocularFrame
{
  std::int64_t timestamp_; // in nanoseconds
  E image_path_;
};

// 双目视觉帧
template <typename E = std::filesystem::path>
struct StereoFrame
{
  std::int64_t timestamp_; // in nanoseconds
  E image_left_;
  E image_right_;
};

struct ImageDataLoader
{
private:
  static std::vector<MonocularFrame<std::filesystem::path>>
  ReadIndex(const std::filesystem::path &path_mav0, std::string_view cam_name)
  {
    const std::filesystem::path path_index{path_mav0 / cam_name / "data.csv"};
    const std::filesystem::path path_dir{path_mav0 / cam_name / "data"};

    std::ifstream file{path_index};
    std::string line;

    std::vector<MonocularFrame<std::filesystem::path>> data;
    data.reserve(32767);

    // 跳过表头
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      auto timestamp{
          AbstractLoader::get_item_as_int64(ss), // in nanoseconds
      };
      auto image_name{
          AbstractLoader::get_item_as_string(ss),
      };
      if (!image_name.ends_with(".png"))
      {
        throw std::runtime_error{
            "Assertion Error: file name not ends with '.png'!"
        };
      }
      data.emplace_back(timestamp, path_dir / image_name);
    }
    return data;
  }

  static std::vector<StereoFrame<std::filesystem::path>>
  MergeIndex(std::vector<MonocularFrame<std::filesystem::path>> &&series_cam0,
             std::vector<MonocularFrame<std::filesystem::path>> &&series_cam1)
  {
    // 检查左右目视觉帧数量是否相同
    if (series_cam0.size() != series_cam1.size())
    {
      throw std::invalid_argument{std::format("Invalid vector size! "
                                              "Left Camera have {} photos, "
                                              "Right Camera have {} photos.",
                                              series_cam0.size(),
                                              series_cam1.size())};
    }
    // 检查双目视觉帧的时间戳是否一致
    for (auto itr0 = series_cam0.begin(), itr1 = series_cam1.begin();
         itr0 != series_cam0.end(); ++itr0, ++itr1)
    {
      MonocularFrame<std::filesystem::path> &image0{*itr0};
      MonocularFrame<std::filesystem::path> &image1{*itr1};
      if (image0.timestamp_ != image1.timestamp_)
      {
        throw std::runtime_error{std::format("Invalid timestamp! "
                                             "Left Camera ({}), "
                                             "Right Camera ({}).",
                                             image0.timestamp_,
                                             image1.timestamp_)};
      }
    }
    // 构造返回值
    std::vector<StereoFrame<std::filesystem::path>> result;
    result.reserve(series_cam0.size());
    for (auto itr0 = series_cam0.begin(), itr1 = series_cam1.begin();
         itr0 != series_cam0.end(); ++itr0, ++itr1)
    {
      MonocularFrame<std::filesystem::path> &image0{*itr0};
      MonocularFrame<std::filesystem::path> &image1{*itr1};
      result.emplace_back(image0.timestamp_, std::move(image0.image_path_),
                          std::move(image1.image_path_));
    }
    return result;
  }

  static std::vector<StereoFrame<std::filesystem::path>>
  Load(const std::filesystem::path &path_mav0)
  {
    auto stereo_frames{MergeIndex(ReadIndex(path_mav0, "cam0"),
                                  ReadIndex(path_mav0, "cam1"))};
    std::print(stderr,
               "[INFO] 成功加载立体视觉图片索引文件, 待读取图片张数: {}.\n",
               stereo_frames.size());
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
  std::vector<StereoFrame<std::filesystem::path>> stereo_frames_;
  using const_iterator = typename decltype(stereo_frames_)::const_iterator;
  const_iterator itr_stereo_frames_{stereo_frames_.cbegin()};

public:
  ImageDataLoader(const std::filesystem::path &path_mav0) :
    stereo_frames_{Load(path_mav0)}
  {
  }

  explicit operator bool() const
  {
    return this->itr_stereo_frames_ != this->stereo_frames_.cend();
  }

  StereoFrame<cv::Mat> operator()() const
  {
    StereoFrame<cv::Mat> result{
        this->itr_stereo_frames_->timestamp_,
        ReadImage(this->itr_stereo_frames_->image_left_),
        ReadImage(this->itr_stereo_frames_->image_right_),
    };
    return result;
  }

  bool operator++()
  {
    if (this->itr_stereo_frames_ != this->stereo_frames_.cend())
    {
      ++this->itr_stereo_frames_;
      return true;
    }
    return false;
  }

  bool operator--()
  {
    if (this->itr_stereo_frames_ != this->stereo_frames_.cbegin())
    {
      --this->itr_stereo_frames_;
      return true;
    }
    return false;
  }
};
