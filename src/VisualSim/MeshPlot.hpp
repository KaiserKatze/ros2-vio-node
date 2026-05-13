#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Room.hpp"
#include "StereoRig.hpp"

template <typename value_type> struct MeshPlot
{
  using Point2 = Eigen::Vector<value_type, 2>;
  using Point3 = Eigen::Vector<value_type, 3>;
  // 引入 Room 中定义的整数坐标别名
  using Point3i = typename Room<value_type>::Point3i;

  const std::string window_left_{"Left Camera"};
  const std::string window_right_{"Right Camera"};

  MeshPlot(bool create_named_window = true)
  {
    if (create_named_window)
    {
      cv::namedWindow(window_left_, cv::WINDOW_NORMAL);
      cv::namedWindow(window_right_, cv::WINDOW_NORMAL);
      // cv::Mat dummy_image(800, 600, CV_8UC3, 0);
      // cv::imshow(window_left_, dummy_image);
      // cv::imshow(window_right_, dummy_image);
      // cv::waitKey(1);
      // cv::moveWindow(window_left_, 0, 0);
      // cv::moveWindow(window_right_, 0, 0);
      // cv::Size window_size{800, 600};
      // cv::resizeWindow(window_left_, window_size);
      // cv::resizeWindow(window_right_, window_size);
    }
  }

  ~MeshPlot()
  {
    cv::destroyAllWindows();
  }

  // 使用索引表示三角形
  struct Triangle : public std::array<size_t, 3>
  {
    Triangle(size_t a, size_t b, size_t c)
    {
      (*this)[0] = a;
      (*this)[1] = b;
      (*this)[2] = c;
      std::sort(std::begin(*this), std::end(*this));
    }
  };

  using Edge = std::pair<size_t, size_t>;
  static Edge make_edge(size_t a, size_t b)
  {
    return {std::min(a, b), std::max(a, b)};
  }

  std::vector<Triangle> mesh_;
  std::vector<bool> mesh_colors_;

  // 构造函数：接管初始化网格与图着色的任务
  MeshPlot(const Room<value_type> &room)
  {
    // 在 XY (Z=0) 平面上切割三角形
    for (int iy = 0; iy < room.cnt_sep_width_; iy += 2)
    {
      for (int ix = 0; ix < room.cnt_sep_depth_; ix += 2)
      {
        auto pt00{room.GetIndex({ix + 0, iy + 0, 0})};
        auto pt10{room.GetIndex({ix + 1, iy + 0, 0})};
        auto pt20{room.GetIndex({ix + 2, iy + 0, 0})};
        auto pt01{room.GetIndex({ix + 0, iy + 1, 0})};
        auto pt11{room.GetIndex({ix + 1, iy + 1, 0})};
        auto pt21{room.GetIndex({ix + 2, iy + 1, 0})};
        auto pt02{room.GetIndex({ix + 0, iy + 2, 0})};
        auto pt12{room.GetIndex({ix + 1, iy + 2, 0})};
        auto pt22{room.GetIndex({ix + 2, iy + 2, 0})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }

    // 在 YZ (X=0) 平面上切割三角形
    for (int iz = 0; iz < room.cnt_sep_height_; iz += 2)
    {
      for (int iy = 0; iy < room.cnt_sep_width_; iy += 2)
      {
        auto pt00{room.GetIndex({0, iy + 0, iz + 0})};
        auto pt10{room.GetIndex({0, iy + 1, iz + 0})};
        auto pt20{room.GetIndex({0, iy + 2, iz + 0})};
        auto pt01{room.GetIndex({0, iy + 0, iz + 1})};
        auto pt11{room.GetIndex({0, iy + 1, iz + 1})};
        auto pt21{room.GetIndex({0, iy + 2, iz + 1})};
        auto pt02{room.GetIndex({0, iy + 0, iz + 2})};
        auto pt12{room.GetIndex({0, iy + 1, iz + 2})};
        auto pt22{room.GetIndex({0, iy + 2, iz + 2})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }

    // 在 ZX (Y=0) 平面上切割三角形
    for (int ix = 0; ix < room.cnt_sep_depth_; ix += 2)
    {
      for (int iz = 0; iz < room.cnt_sep_height_; iz += 2)
      {
        auto pt00{room.GetIndex({ix + 0, 0, iz + 0})};
        auto pt10{room.GetIndex({ix + 0, 0, iz + 1})};
        auto pt20{room.GetIndex({ix + 0, 0, iz + 2})};
        auto pt01{room.GetIndex({ix + 1, 0, iz + 0})};
        auto pt11{room.GetIndex({ix + 1, 0, iz + 1})};
        auto pt21{room.GetIndex({ix + 1, 0, iz + 2})};
        auto pt02{room.GetIndex({ix + 2, 0, iz + 0})};
        auto pt12{room.GetIndex({ix + 2, 0, iz + 1})};
        auto pt22{room.GetIndex({ix + 2, 0, iz + 2})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }

    // 在 XY (Z=height) 平面上切割三角形
    for (int iy = 0; iy < room.cnt_sep_width_; iy += 2)
    {
      for (int ix = 0; ix < room.cnt_sep_depth_; ix += 2)
      {
        auto pt00{room.GetIndex({ix + 0, iy + 0, room.cnt_sep_height_})};
        auto pt10{room.GetIndex({ix + 1, iy + 0, room.cnt_sep_height_})};
        auto pt20{room.GetIndex({ix + 2, iy + 0, room.cnt_sep_height_})};
        auto pt01{room.GetIndex({ix + 0, iy + 1, room.cnt_sep_height_})};
        auto pt11{room.GetIndex({ix + 1, iy + 1, room.cnt_sep_height_})};
        auto pt21{room.GetIndex({ix + 2, iy + 1, room.cnt_sep_height_})};
        auto pt02{room.GetIndex({ix + 0, iy + 2, room.cnt_sep_height_})};
        auto pt12{room.GetIndex({ix + 1, iy + 2, room.cnt_sep_height_})};
        auto pt22{room.GetIndex({ix + 2, iy + 2, room.cnt_sep_height_})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }

    // 在 YZ (X=depth) 平面上切割三角形
    for (int iz = 0; iz < room.cnt_sep_height_; iz += 2)
    {
      for (int iy = 0; iy < room.cnt_sep_width_; iy += 2)
      {
        auto pt00{room.GetIndex({room.cnt_sep_depth_, iy + 0, iz + 0})};
        auto pt10{room.GetIndex({room.cnt_sep_depth_, iy + 1, iz + 0})};
        auto pt20{room.GetIndex({room.cnt_sep_depth_, iy + 2, iz + 0})};
        auto pt01{room.GetIndex({room.cnt_sep_depth_, iy + 0, iz + 1})};
        auto pt11{room.GetIndex({room.cnt_sep_depth_, iy + 1, iz + 1})};
        auto pt21{room.GetIndex({room.cnt_sep_depth_, iy + 2, iz + 1})};
        auto pt02{room.GetIndex({room.cnt_sep_depth_, iy + 0, iz + 2})};
        auto pt12{room.GetIndex({room.cnt_sep_depth_, iy + 1, iz + 2})};
        auto pt22{room.GetIndex({room.cnt_sep_depth_, iy + 2, iz + 2})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }

    // 在 ZX (Y=width) 平面上切割三角形
    for (int ix = 0; ix < room.cnt_sep_depth_; ix += 2)
    {
      for (int iz = 0; iz < room.cnt_sep_height_; iz += 2)
      {
        auto pt00{room.GetIndex({ix + 0, room.cnt_sep_width_, iz + 0})};
        auto pt10{room.GetIndex({ix + 0, room.cnt_sep_width_, iz + 1})};
        auto pt20{room.GetIndex({ix + 0, room.cnt_sep_width_, iz + 2})};
        auto pt01{room.GetIndex({ix + 1, room.cnt_sep_width_, iz + 0})};
        auto pt11{room.GetIndex({ix + 1, room.cnt_sep_width_, iz + 1})};
        auto pt21{room.GetIndex({ix + 1, room.cnt_sep_width_, iz + 2})};
        auto pt02{room.GetIndex({ix + 2, room.cnt_sep_width_, iz + 0})};
        auto pt12{room.GetIndex({ix + 2, room.cnt_sep_width_, iz + 1})};
        auto pt22{room.GetIndex({ix + 2, room.cnt_sep_width_, iz + 2})};
        // 黑色三角形
        mesh_.emplace_back(pt00, pt01, pt11);
        mesh_.emplace_back(pt10, pt11, pt20);
        mesh_.emplace_back(pt02, pt11, pt12);
        mesh_.emplace_back(pt11, pt21, pt22);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        mesh_colors_.push_back(false);
        // 白色三角形
        mesh_.emplace_back(pt00, pt10, pt11);
        mesh_.emplace_back(pt11, pt20, pt21);
        mesh_.emplace_back(pt01, pt02, pt11);
        mesh_.emplace_back(pt11, pt12, pt22);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
        mesh_colors_.push_back(true);
      }
    }
  }

  bool Render(const cv::Mat &img_left, const cv::Mat &img_right,
              int delay = 0) const
  {
    cv::imshow(window_left_, img_left);
    cv::imshow(window_right_, img_right);
    const int key{cv::waitKey(delay) & 0xFF};
    return key == 'q';
  }

  /**
   * @brief 渲染并绘制双目相机的观测图像上的三维网格面片.
   * @param img_left 左相机渲染的目标图像矩阵.
   * @param img_right 右相机渲染的目标图像矩阵.
   * @param frame 当前帧包含的左右目可见点索引及像素坐标.
   */
  void Draw(cv::Mat &img_left, cv::Mat &img_right,
            const Frame<value_type> &frame) const
  {
    // 定义并初始化左目相机的局部可见点索引到像素坐标的映射表.
    std::map<size_t, cv::Point2f> map_left;
    // 遍历当前帧左目相机中所有可见的路标点.
    for (size_t i = 0; i < frame.indices_left_.size(); ++i)
    {
      // 提取二维点坐标并通过索引构建左目的键值对映射关系.
      map_left[frame.indices_left_[i]]
          = cv::Point2f(frame.pixels_left_[i].x(), frame.pixels_left_[i].y());
    }

    // 定义并初始化右目相机的局部可见点索引到像素坐标的映射表.
    std::map<size_t, cv::Point2f> map_right;
    // 遍历当前帧右目相机中所有可见的路标点.
    for (size_t i = 0; i < frame.indices_right_.size(); ++i)
    {
      // 提取二维点坐标并通过索引构建右目的键值对映射关系.
      map_right[frame.indices_right_[i]]
          = cv::Point2f(frame.pixels_right_[i].x(), frame.pixels_right_[i].y());
    }

    // 遍历当前网格模型中包含的所有三角形面片.
    for (size_t i = 0; i < mesh_.size(); ++i)
    {
      // 取出构成当前三角形的三个顶点的路标点全局索引.
      size_t v0{mesh_[i][0]}, v1{mesh_[i][1]}, v2{mesh_[i][2]};

      // 根据网格的颜色属性表判断当前面片应填充白色还是黑色.
      cv::Scalar fill_color
          = (mesh_colors_[i]) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

      // 检查当前三角形的三个顶点是否在左目相机的可见映射表中全部存在.
      if (map_left.count(v0) && map_left.count(v1) && map_left.count(v2))
      {
        // 声明一个 OpenCV 的点数组来存储三角形在左目图像上的三个顶点坐标.
        cv::Point pts_left[3];
        // 提取第一个顶点坐标进行四舍五入并存入该点数组中.
        pts_left[0]
            = cv::Point(std::round(map_left[v0].x), std::round(map_left[v0].y));
        // 提取第二个顶点坐标进行四舍五入并存入该点数组中.
        pts_left[1]
            = cv::Point(std::round(map_left[v1].x), std::round(map_left[v1].y));
        // 提取第三个顶点坐标进行四舍五入并存入该点数组中.
        pts_left[2]
            = cv::Point(std::round(map_left[v2].x), std::round(map_left[v2].y));
        // 调用 OpenCV 函数在左目图像上填充绘制该多边形面片.
        cv::fillConvexPoly(img_left, pts_left, 3, fill_color);
      }

      // 检查当前三角形的三个顶点是否在右目相机的可见映射表中全部存在.
      if (map_right.count(v0) && map_right.count(v1) && map_right.count(v2))
      {
        // 声明一个 OpenCV 的点数组来存储三角形在右目图像上的三个顶点坐标.
        cv::Point pts_right[3];
        // 提取第一个顶点坐标进行四舍五入并存入该点数组中.
        pts_right[0] = cv::Point(std::round(map_right[v0].x),
                                 std::round(map_right[v0].y));
        // 提取第二个顶点坐标进行四舍五入并存入该点数组中.
        pts_right[1] = cv::Point(std::round(map_right[v1].x),
                                 std::round(map_right[v1].y));
        // 提取第三个顶点坐标进行四舍五入并存入该点数组中.
        pts_right[2] = cv::Point(std::round(map_right[v2].x),
                                 std::round(map_right[v2].y));
        // 调用 OpenCV 函数在右目图像上填充绘制该多边形面片.
        cv::fillConvexPoly(img_right, pts_right, 3, fill_color);
      }
    }
  }
};
