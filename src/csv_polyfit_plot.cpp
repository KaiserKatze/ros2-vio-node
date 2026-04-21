// csv_polyfit_plot.cpp
// 支持从标准输入(stdin)读取 CSV（三列：t, y1, y2），用于管道：
// ./euroc_rk4_eval ... | csv_polyfit_plot
// 也支持从文件读取：csv_polyfit_plot file.csv
//
// 自动跳过非数值行（例如表头或日志输出）
// 使用 Eigen 做多项式拟合（2~6阶），OpenCV 绘图（两个窗口）

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ======================== 数据结构 ========================
struct DataRow
{
  double t;
  double y1;
  double y2;
};

// ======================== 工具函数 ========================
bool try_parse_double(const std::string &s, double &value)
{
  try
  {
    size_t idx;
    value = std::stod(s, &idx);
    return idx > 0;
  }
  catch (...)
  {
    return false;
  }
}

// ======================== CSV 解析 ========================
// 从任意输入流读取（支持 stdin / 文件）
std::vector<DataRow> read_csv_stream(std::istream &in)
{
  std::vector<DataRow> data;
  std::string line;

  while (std::getline(in, line))
  {
    if (line.empty())
    {
      continue;
    }

    std::stringstream ss(line);
    std::string item;

    std::vector<double> values;
    while (std::getline(ss, item, ','))
    {
      double v;
      if (try_parse_double(item, v))
      {
        values.push_back(v);
      }
    }

    // 至少需要3列（t, y1, y2）
    if (values.size() < 3)
    {
      continue;
    }

    DataRow row;
    row.t  = values[0];
    row.y1 = values[1];
    row.y2 = values[2];

    data.push_back(row);
  }

  return data;
}

// ======================== 多项式拟合 ========================
Eigen::VectorXd polyfit(const std::vector<double> &x,
                        const std::vector<double> &y, int degree)
{
  const int N = static_cast<int>(x.size());
  Eigen::MatrixXd A(N, degree + 1);
  Eigen::VectorXd b(N);

  for (int i = 0; i < N; ++i)
  {
    double xi = 1.0;
    for (int j = 0; j <= degree; ++j)
    {
      A(i, j) = xi;
      xi *= x[i];
    }
    b(i) = y[i];
  }

  return A.colPivHouseholderQr().solve(b);
}

double polyval(const Eigen::VectorXd &coeffs, double x)
{
  double y  = 0.0;
  double xi = 1.0;
  for (int i = 0; i < coeffs.size(); ++i)
  {
    y += coeffs[i] * xi;
    xi *= x;
  }
  return y;
}

// ======================== 绘图 ========================
void plot_data_and_fit(const std::vector<double> &x,
                       const std::vector<double> &y,
                       const Eigen::VectorXd &coeffs,
                       const std::string &window_name)
{
  const int width  = 800;
  const int height = 600;

  cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  double xmin = *std::min_element(x.begin(), x.end());
  double xmax = *std::max_element(x.begin(), x.end());
  double ymin = *std::min_element(y.begin(), y.end());
  double ymax = *std::max_element(y.begin(), y.end());

  double xpad = 0.05 * (xmax - xmin);
  double ypad = 0.05 * (ymax - ymin);

  xmin -= xpad;
  xmax += xpad;
  ymin -= ypad;
  ymax += ypad;

  auto to_px = [&](double xv, double yv)
  {
    int px = static_cast<int>((xv - xmin) / (xmax - xmin) * width);
    int py = height - static_cast<int>((yv - ymin) / (ymax - ymin) * height);
    return cv::Point(px, py);
  };

  // 原始数据点（蓝色）
  for (size_t i = 0; i < x.size(); ++i)
  {
    cv::circle(img, to_px(x[i], y[i]), 2, cv::Scalar(255, 0, 0), -1);
  }

  // 拟合曲线（红色）
  const int samples = 1000;
  for (int i = 1; i < samples; ++i)
  {
    double t0 = xmin + (xmax - xmin) * (i - 1) / samples;
    double t1 = xmin + (xmax - xmin) * i / samples;

    double y0 = polyval(coeffs, t0);
    double y1 = polyval(coeffs, t1);

    cv::line(img, to_px(t0, y0), to_px(t1, y1), cv::Scalar(0, 0, 255), 2);
  }

  cv::imshow(window_name, img);
}

// ======================== 主函数 ========================
int main(int argc, char **argv)
{
  std::vector<DataRow> data;

  // -------- 输入来源判断 --------
  if (argc > 1)
  {
    // 从文件读取
    std::ifstream file(argv[1]);
    if (!file)
    {
      std::cerr << "无法打开文件: " << argv[1] << "\n";
      return 1;
    }
    data = read_csv_stream(file);
  }
  else
  {
    // 从 stdin 读取（支持管道）
    data = read_csv_stream(std::cin);
  }

  if (data.empty())
  {
    std::cerr << "未读取到有效数据（请检查 CSV 或管道输出）\n";
    return 1;
  }

  std::vector<double> t, y1, y2;
  t.reserve(data.size());
  y1.reserve(data.size());
  y2.reserve(data.size());

  for (const auto &row : data)
  {
    t.push_back(row.t);
    y1.push_back(row.y1);
    y2.push_back(row.y2);
  }

  // 多项式阶数（可改 2~6）
  int degree = 4;

  Eigen::VectorXd coeffs1 = polyfit(t, y1, degree);
  Eigen::VectorXd coeffs2 = polyfit(t, y2, degree);

  std::cout << "y1 coeffs: " << coeffs1.transpose() << "\n";
  std::cout << "y2 coeffs: " << coeffs2.transpose() << "\n";

  plot_data_and_fit(t, y1, coeffs1, "Y1 vs t");
  plot_data_and_fit(t, y2, coeffs2, "Y2 vs t");

  cv::waitKey(0);
  return 0;
}
