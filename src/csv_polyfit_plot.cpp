// csv_polyfit_plot.cpp
// 使用 gnuplot 进行高质量绘图（带坐标轴/网格/图例）
// 支持 stdin 管道：
//   ./euroc_rk4_eval imu.csv gt.csv | csv_polyfit_plot
// 也支持文件输入：
//   csv_polyfit_plot data.csv
//
// 功能：
//  - 自动解析 CSV（跳过非数值行）
//  - 分别对 y1, y2 做 2~6 次多项式拟合（默认 4 次）
//  - 输出带 x^k 标注的系数
//  - 使用 gnuplot 绘制两个窗口

// colcon build && ./build/euroc_vio/euroc_rk4_eval /mnt/e/Documents/mav0/imu0/data.csv /mnt/e/Documents/mav0/state_groundtruth_estimate0/data.csv | ./build/euroc_vio/csv_polyfit_plot

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ======================== 数据结构 ========================
using DataRow    = std::vector<double>;
using DataColumn = std::vector<double>;

// ======================== 工具 ========================
bool try_parse_double(std::string_view s, double &value)
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

// ======================== CSV 读取 ========================
std::vector<DataRow> read_csv_stream(std::istream &in)
{
  std::vector<DataRow> data;
  std::string line;
  // size_t line_number = 0;

  // 跳过表头
  std::getline(in, line);
  while (std::getline(in, line))
  {
    // std::cerr << "Line [" << ++line_number << "]: " << line << "\n";
    if (line.empty())
    {
      continue;
    }

    std::stringstream ss(line);
    std::string item;
    DataRow row;
    while (std::getline(ss, item, ','))
    {
      double v;
      if (try_parse_double(item, v))
      {
        row.push_back(v);
      }
    }

    data.push_back(row);
  }

  return data;
}

// ======================== polyfit ========================
/**
 * @brief 多项式拟合
 * @param x 曲线上各点的横坐标数据
 * @param y 曲线上各点的纵坐标数据
 * @param degree 拟合多项式的次数
 * @return 拟合得到的多项式系数 (按 0 次项到 degree 次项升序排列)
 */
Eigen::VectorXd polyfit(const std::vector<double> &x,
                        const std::vector<double> &y, int degree)
{
  if (x.size() == 0 || y.size() != x.size())
  {
    throw std::invalid_argument("x 和 y 的数据点数量必须相等且不为空。");
  }

  const int N = static_cast<int>(x.size());

  // N 个点最多只能唯一确定 N-1 次多项式，故取 N-1 和 degree 中的较小值
  degree = std::max(std::min(N - 1, degree), 1);

  // X 为设计矩阵 (Vandermonde 矩阵)，大小为 N 行 (degree + 1) 列
  Eigen::MatrixXd X(N, degree + 1);
  // Y 为目标向量
  Eigen::VectorXd Y(N);

  // 填充矩阵 X 和向量 Y
  for (int i = 0; i < N; ++i)
  {
    Y(i) = y[i];

    // 构造多项式的各项：x^0, x^1, x^2, ..., x^degree
    double x_pow = 1.0;
    for (int j = 0; j <= degree; ++j)
    {
      X(i, j) = x_pow;
      x_pow *= x[i]; // 连乘，避免使用计算代价较高的 pow 函数
    }
  }

  // 使用列主元 Householder QR 分解直接求解超定方程组 X * a = Y
  // 这等价于求解法方程，但避免了计算 X^T * X 带来的平方级误差放大，数值更稳定
  Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(Y);

  return coeffs;
}

double polyval(const Eigen::VectorXd &c, double x)
{
  double y = 0.0, xi = 1.0;
  for (int i = 0; i < c.size(); ++i)
  {
    y += c[i] * xi;
    xi *= x;
  }
  return y;
}

// ======================== gnuplot 绘图 ========================
void plot_with_gnuplot(const std::vector<double> &t,
                       const std::vector<double> &y,
                       const Eigen::VectorXd &coeffs, std::string_view title)
{
  FILE *gp = popen("gnuplot -persistent", "w");
  if (!gp)
  {
    std::cerr << "无法启动 gnuplot\n";
    return;
  }

  fprintf(gp, "set title '%s'\n", title.c_str());
  fprintf(gp, "set grid\n");
  fprintf(gp, "set xlabel 't'\n");
  fprintf(gp, "set ylabel 'value'\n");

  fprintf(gp, "plot '-' with points pt 7 ps 0.5 title 'data', "
              "'-' with lines lw 2 title 'polyfit'\n");

  // 原始数据
  for (size_t i = 0; i < t.size(); ++i)
  {
    fprintf(gp, "%f %f\n", t[i], y[i]);
  }
  fprintf(gp, "e\n");

  // 拟合曲线（采样）
  int samples = 1000;
  double xmin = *std::min_element(t.begin(), t.end());
  double xmax = *std::max_element(t.begin(), t.end());

  for (int i = 0; i < samples; ++i)
  {
    double x  = xmin + (xmax - xmin) * i / (samples - 1);
    double yy = polyval(coeffs, x);
    fprintf(gp, "%f %f\n", x, yy);
  }
  fprintf(gp, "e\n");

  fflush(gp);
  pclose(gp);
}

// ======================== 主函数 ========================
int main(int argc, char **argv)
{
  std::vector<DataRow> data;

  if (argc > 1)
  {
    std::ifstream file{argv[1]};
    if (!file)
    {
      std::cerr << "无法打开文件\n";
      return 1;
    }
    data = read_csv_stream(file);
  }
  else
  {
    data = read_csv_stream(std::cin);
  }

  if (data.empty())
  {
    std::cerr << "未读取到有效数据\n";
    return 1;
  }

  std::cerr << "读取有效数据行数: " << data.size() << "\n";

  auto itr                 = data.cbegin();
  const DataRow &first_row = *itr;

  size_t n_columns = first_row.size();
  if (n_columns < 2)
  {
    std::cerr << "第一行的列数 (" << n_columns << ") 小于2!\n";
    return 1;
  }

  double time_start = first_row[0];

  ++itr;
  for (; itr != data.cend(); ++itr)
  {
    if (itr->size() != n_columns)
    {
      std::cerr << "第 " << std::distance(data.cbegin(), itr)
                << " 行的列数与第一行的列数不同!\n";
      return 1;
    }
  }
  std::cerr << "有效数据列数: " << n_columns << "\n";

  if (n_columns < 2)
  {
    std::cerr << "数据列数小于2!\n";
    return 1;
  }

  std::vector<DataColumn> columns;
  columns.resize(n_columns);
  DataColumn &t = columns[0];
  for (const DataRow &row : data)
  {
    const double time_now = row[0];
    double time_delta     = time_now - time_start;
    // if (time_delta > 10) {
    //   break; // 只处理前 10 秒的数据，避免过长时间导致拟合过度复杂
    // }
    t.push_back(time_delta);
    for (size_t i = 1; i < n_columns; ++i)
    {
      columns[i].push_back(row[i]);
    }
  }

  int degree;
  degree = 2; //

  for (size_t i = 1; i < n_columns; ++i)
  {
    const DataColumn &column = columns[i];
    // 最小二乘拟合
    const auto coeffs
        = polyfit(const_cast<const DataColumn &>(t), column, degree);
    // 输出带含义的系数
    std::cerr << "==== y" << i << " polynomial ====\n";
    for (int j = 0; j < coeffs.size(); ++j)
    {
      std::cerr << "\tc[" << j << "] (x^" << j << ") = " << coeffs[j] << "\n";
    }
    // 绘图
    std::stringstream ss_title;
    ss_title << "Y" << i << " vs t (polyfit)";
    plot_with_gnuplot(t, column, coeffs, ss_title.str());
  }

  return 0;
}
