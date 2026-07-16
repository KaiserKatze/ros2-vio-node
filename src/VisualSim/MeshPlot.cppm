module;

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

export module FastVIO:VisualSim:MeshPlot;

import std;

import FastVIO:VisualSim:Room;

namespace FastVIO::VisualSim
{

template <typename value_type>
struct MeshPlot
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
  template <typename FrameType>
  void Draw(cv::Mat &img_left, cv::Mat &img_right, const FrameType &frame) const
  {
    const std::vector<size_t> &visible_indices{std::get<0>(frame)};
    const std::vector<Point2> &img_pts_left{std::get<1>(frame)};
    const std::vector<Point2> &img_pts_right{std::get<2>(frame)};

    // 利用序号作为 key，轻量级映射
    std::map<size_t, std::pair<cv::Point2f, cv::Point2f>> visible_map;
    for (size_t i = 0; i < visible_indices.size(); ++i)
    {
      visible_map[visible_indices[i]]
          = {cv::Point2f(img_pts_left[i].x(), img_pts_left[i].y()),
             cv::Point2f(img_pts_right[i].x(), img_pts_right[i].y())};
    }

    for (size_t i = 0; i < mesh_.size(); ++i)
    {
      size_t v0{mesh_[i][0]}, v1{mesh_[i][1]}, v2{mesh_[i][2]};

      if (visible_map.count(v0) && visible_map.count(v1)
          && visible_map.count(v2))
      {
        cv::Point pts_left[3], pts_right[3];
        auto p0{visible_map[v0]}, p1{visible_map[v1]}, p2{visible_map[v2]};

        pts_left[0] = cv::Point(std::round(p0.first.x), std::round(p0.first.y));
        pts_left[1] = cv::Point(std::round(p1.first.x), std::round(p1.first.y));
        pts_left[2] = cv::Point(std::round(p2.first.x), std::round(p2.first.y));

        pts_right[0]
            = cv::Point(std::round(p0.second.x), std::round(p0.second.y));
        pts_right[1]
            = cv::Point(std::round(p1.second.x), std::round(p1.second.y));
        pts_right[2]
            = cv::Point(std::round(p2.second.x), std::round(p2.second.y));

        cv::Scalar fill_color = (mesh_colors_[i]) ? cv::Scalar(255, 255, 255)
                                                  : cv::Scalar(0, 0, 0);
        cv::fillConvexPoly(img_left, pts_left, 3, fill_color);
        cv::fillConvexPoly(img_right, pts_right, 3, fill_color);

        const cv::Point *l_ptr[1] = {pts_left};
        const cv::Point *r_ptr[1] = {pts_right};
        int npts[]                = {3};
        cv::polylines(img_left, l_ptr, npts, 1, true, cv::Scalar(0, 255, 0), 1);
        cv::polylines(img_right, r_ptr, npts, 1, true, cv::Scalar(0, 255, 0),
                      1);
      }
    }
  }
};

} // namespace FastVIO::VisualSim
