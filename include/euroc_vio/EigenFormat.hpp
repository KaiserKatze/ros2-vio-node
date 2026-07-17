module;

#include <concepts>
#include <cstddef>
#include <meta>
#include <string_view>
#include <type_traits>

import <Eigen/Dense>;

export module EigenFormat;

// import std;

// ─────────────────────────────────────────────────────────────────────────────
//  §1  Concepts — identify Eigen dense types via static reflection
// ─────────────────────────────────────────────────────────────────────────────
namespace eigen_fmt::detail
{

// True when T is any specialisation of Eigen::MatrixBase<Derived>
template <typename T>
concept EigenMatrixBase = requires {
  // MatrixBase exposes a RowsAtCompileTime enum member
  { T::RowsAtCompileTime } -> std::convertible_to<int>;
  { T::ColsAtCompileTime } -> std::convertible_to<int>;
  typename T::Scalar;
} && std::is_base_of_v<Eigen::MatrixBase<T>, T>;

// Narrow concept: plain Matrix / Vector (not expression templates)
template <typename T>
concept EigenPlainMatrix
    = EigenMatrixBase<T> && std::is_base_of_v<Eigen::PlainObjectBase<T>, T>;

// Column-vector: ColsAtCompileTime == 1
template <typename T>
concept EigenColumnVector = EigenPlainMatrix<T> && (T::ColsAtCompileTime == 1);

// Row-vector: RowsAtCompileTime == 1
template <typename T>
concept EigenRowVector = EigenPlainMatrix<T> && (T::RowsAtCompileTime == 1);

// Quaternion concept
template <typename T>
concept EigenQuaternionType = std::is_base_of_v<Eigen::QuaternionBase<T>, T>;

// ── Reflection-based concept: does T have a static member called "IsRowMajor"?
// Uses C++26 ^^-operator and std::meta::nonstatic_data_members_of
template <typename T>
concept HasRowMajorFlag = ([] consteval {
    for (std::meta::info m :
             std::meta::static_data_members_of(^^T,
                 std::meta::access_context::unchecked())) {
        if (std::meta::identifier_of(m) == "IsRowMajor"){
            return true;
}
    }
    return false;
}());

} // namespace eigen_fmt::detail

// ─────────────────────────────────────────────────────────────────────────────
//  §2  Scalar type name — obtained purely via C++26 reflection
// ─────────────────────────────────────────────────────────────────────────────
namespace eigen_fmt::detail
{

// Return a human-readable name for any arithmetic/complex scalar.
// Reflection lets us query the *canonical* type name at compile time
// without any manual if-else chains.
template <typename Scalar>
consteval std::string_view scalar_type_name()
{
  // Use display_string_of on the reflected type
  constexpr auto r = ^^Scalar;
  return std::meta::display_string_of(r); // e.g. "double", "float", …
}

// Convenience: row × col dimension string produced at compile time
template <int Rows, int Cols>
consteval std::string dim_string()
{
  if constexpr (Rows == Eigen::Dynamic && Cols == Eigen::Dynamic)
  {
    return "Xd×Xd";
  }
  else if constexpr (Rows == Eigen::Dynamic)
  {
    return std::format("Xd×{}", Cols);
  }
  else if constexpr (Cols == Eigen::Dynamic)
  {
    return std::format("{}×Xd", Rows);
  }
  else
  {
    return std::format("{}×{}", Rows, Cols);
  }
}

} // namespace eigen_fmt::detail

// ─────────────────────────────────────────────────────────────────────────────
//  §3  Matrix / Vector formatter
//
//  Format spec mini-language (subset):
//    [[fill]align][width][.precision][type]
//    type: 'r' = row layout (default for row-major / row-vectors)
//          'c' = column layout (default for col-major / col-vectors)
//          'm' = multi-line matrix with aligned columns
//          'n' = numpy-style  [[ ... ], [ ... ]]
//          'j' = JSON array   [[...],[...]]
//          't' = type header  MatrixXd(3×4){{ ... }}
// ─────────────────────────────────────────────────────────────────────────────

export namespace std
{

template <eigen_fmt::detail::EigenPlainMatrix Mat>
struct formatter<Mat>
{
private:
  char layout_ = 0; // 'r','c','m','n','j','t'
  int width_   = 0;
  int prec_    = 6; // digits after decimal point for floating-point

  // ── Reflection helper: iterate template arguments of Mat ──────────────
  // In C++26 we can query the template arguments of a specialisation
  // via std::meta::template_arguments_of.
  static consteval auto get_rows()
  {
    // Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    // template argument [1] = Rows (an int NTTP)
    constexpr auto args = std::meta::template_arguments_of(^^Mat);
    return std::meta::extract<int>(args[1]);
  }
  static consteval auto get_cols()
  {
    constexpr auto args = std::meta::template_arguments_of(^^Mat);
    return std::meta::extract<int>(args[2]);
  }
  static consteval bool is_row_major()
  {
    constexpr auto args = std::meta::template_arguments_of(^^Mat);
    // Options flags, bit 0 = RowMajor
    constexpr int opts = std::meta::extract<int>(args[3]);
    return (opts & Eigen::RowMajor) != 0;
  }

  // ── The type-name string is built fully at compile time ───────────────
  static consteval std::string type_name()
  {
    using Scalar    = typename Mat::Scalar;
    constexpr int R = get_rows(), C = get_cols();
    return std::format("Matrix<{},{}>", std::meta::display_string_of(^^Scalar),
                       eigen_fmt::detail::dim_string<R, C>());
  }

public:
  // ── parse() ─────────────────────────────────────────────────────────
  constexpr auto parse(std::format_parse_context &ctx)
  {
    auto it = ctx.begin();
    // precision
    if (it != ctx.end() && *it == '.')
    {
      ++it;
      prec_ = 0;
      while (it != ctx.end() && std::isdigit((unsigned char) *it))
      {
        prec_ = prec_ * 10 + (*it++ - '0');
      }
    }
    // type char
    if (it != ctx.end()
        && (*it == 'r' || *it == 'c' || *it == 'm' || *it == 'n' || *it == 'j'
            || *it == 't'))
    {
      layout_ = *it++;
    }
    if (it != ctx.end() && *it != '}')
    {
      throw std::format_error("invalid Eigen::Matrix format spec");
    }
    return it;
  }

  // ── format() ────────────────────────────────────────────────────────
  template <typename OutputIt>
  auto format(const Mat &m,
              std::basic_format_context<OutputIt, char> &ctx) const
  {
    auto out = ctx.out();

    // Determine effective layout
    char layout = layout_;
    if (layout == 0)
    {
      if constexpr (eigen_fmt::detail::EigenRowVector<Mat>)
      {
        layout = 'r';
      }
      else if constexpr (eigen_fmt::detail::EigenColumnVector<Mat>)
      {
        layout = 'c';
      }
      else
      {
        layout = 'm';
      }
    }

    const Eigen::Index rows = m.rows(), cols = m.cols();
    const std::string scalar_fmt
        = std::format("{{:.{}g}}", prec_); // e.g. "{:.6g}"

    auto fmt_scalar = [&](typename Mat::Scalar v)
    { return std::vformat(scalar_fmt, std::make_format_args(v)); };

    switch (layout)
    {

    // ── 'r' : row vector  [1 2 3]
    case 'r':
    {
      out = std::format_to(out, "[");
      for (Eigen::Index j = 0; j < cols; ++j)
      {
        if (j)
        {
          out = std::format_to(out, "  ");
        }
        out = std::format_to(out, "{}", fmt_scalar(m.coeff(0, j)));
      }
      out = std::format_to(out, "]");
      break;
    }

    // ── 'c' : column vector  [1; 2; 3]
    case 'c':
    {
      out = std::format_to(out, "[");
      for (Eigen::Index i = 0; i < rows; ++i)
      {
        if (i)
        {
          out = std::format_to(out, "; ");
        }
        out = std::format_to(out, "{}", fmt_scalar(m.coeff(i, 0)));
      }
      out = std::format_to(out, "]ᵀ");
      break;
    }

    // ── 'm' : multi-line  (column-aligned)
    case 'm':
    {
      // First pass: collect strings to compute column widths
      std::vector<std::vector<std::string>> cells(
          rows, std::vector<std::string>(cols)
      );
      std::vector<std::size_t> col_w(cols, 0);
      for (Eigen::Index i = 0; i < rows; ++i)
      {
        for (Eigen::Index j = 0; j < cols; ++j)
        {
          cells[i][j] = fmt_scalar(m.coeff(i, j));
          col_w[j]    = std::max(col_w[j], cells[i][j].size());
        }
      }
      // Second pass: emit aligned rows
      for (Eigen::Index i = 0; i < rows; ++i)
      {
        out = std::format_to(out, i == 0 ? "⎡" : i == rows - 1 ? "⎣" : "⎢");
        for (Eigen::Index j = 0; j < cols; ++j)
        {
          if (j)
          {
            out = std::format_to(out, "  ");
          }
          const auto &s = cells[i][j];
          // right-align within column width
          for (std::size_t k = s.size(); k < col_w[j]; ++k)
          {
            *out++ = ' ';
          }
          out = std::format_to(out, "{}", s);
        }
        out = std::format_to(out, i == 0 ? "⎤" : i == rows - 1 ? "⎦" : "⎥");
      }
      break;
    }

    // ── 'n' : NumPy-style  [[1, 2], [3, 4]]
    case 'n':
    {
      out = std::format_to(out, "[");
      for (Eigen::Index i = 0; i < rows; ++i)
      {
        if (i)
        {
          out = std::format_to(out, ",\n ");
        }
        out = std::format_to(out, "[");
        for (Eigen::Index j = 0; j < cols; ++j)
        {
          if (j)
          {
            out = std::format_to(out, ", ");
          }
          out = std::format_to(out, "{}", fmt_scalar(m.coeff(i, j)));
        }
        out = std::format_to(out, "]");
      }
      out = std::format_to(out, "]");
      break;
    }

    // ── 'j' : JSON  [[1,2],[3,4]]
    case 'j':
    {
      out = std::format_to(out, "[[");
      for (Eigen::Index i = 0; i < rows; ++i)
      {
        if (i)
        {
          out = std::format_to(out, "],[");
        }
        for (Eigen::Index j = 0; j < cols; ++j)
        {
          if (j)
          {
            out = std::format_to(out, ",");
          }
          out = std::format_to(out, "{}", fmt_scalar(m.coeff(i, j)));
        }
      }
      out = std::format_to(out, "]]");
      break;
    }

    // ── 't' : with type header  Matrix<double,3×3>{{ … }}
    case 't':
    {
      out = std::format_to(out, "{}{{", type_name());
      // Recurse with 'm' layout (reuse this formatter object)
      formatter<Mat> inner;
      inner.layout_ = 'm';
      inner.prec_   = prec_;
      out           = inner.format(m, ctx);
      out           = std::format_to(out, "}}");
      break;
    }

    default:
      break;
    }

    return out;
  }
};

} // namespace std

// ─────────────────────────────────────────────────────────────────────────────
//  §4  Quaternion formatter
//
//  Format spec:
//    'q' = compact  q(w + xi + yj + zk)          (default)
//    'v' = vector   (w, x, y, z)
//    'a' = axis-angle  axis=[x,y,z] angle=θ rad
//    'r' = rotation matrix  (3×3 matrix, delegates to Matrix formatter)
//    't' = with type header
// ─────────────────────────────────────────────────────────────────────────────

export namespace std
{

template <eigen_fmt::detail::EigenQuaternionType Q>
struct formatter<Q>
{
private:
  char mode_ = 'q';
  int prec_  = 6;

  // ── Reflect the Scalar type from the Quaternion specialisation ────────
  static consteval std::string_view scalar_name()
  {
    // Quaternion<Scalar, Options>: argument[0] = Scalar
    constexpr auto args = std::meta::template_arguments_of(^^Q);
    return std::meta::display_string_of(args[0]);
  }

public:
  constexpr auto parse(std::format_parse_context &ctx)
  {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == '.')
    {
      ++it;
      prec_ = 0;
      while (it != ctx.end() && std::isdigit((unsigned char) *it))
      {
        prec_ = prec_ * 10 + (*it++ - '0');
      }
    }
    if (it != ctx.end()
        && (*it == 'q' || *it == 'v' || *it == 'a' || *it == 'r' || *it == 't'))
    {
      mode_ = *it++;
    }
    return it;
  }

  template <typename OutputIt>
  auto format(const Q &q, std::basic_format_context<OutputIt, char> &ctx) const
  {
    auto out             = ctx.out();
    const std::string sf = std::format("{{:.{}g}}", prec_);

    auto fs
        = [&](auto v) { return std::vformat(sf, std::make_format_args(v)); };

    switch (mode_)
    {

    // q(w + xi + yj + zk)
    case 'q':
    {
      auto sign = [](auto v) { return v < 0 ? " - " : " + "; };
      out = std::format_to(out, "q({}{}{}{}{}{}{} )", fs(q.w()), sign(q.x()),
                           fs(std::abs(q.x())), "i", sign(q.y()),
                           fs(std::abs(q.y())), "j");
      out = std::format_to(out, "{}{}k", sign(q.z()), fs(std::abs(q.z())));
      break;
    }

    // (w, x, y, z)
    case 'v':
      out = std::format_to(out, "({}, {}, {}, {})", fs(q.w()), fs(q.x()),
                           fs(q.y()), fs(q.z()));
      break;

    // axis=[x,y,z]  angle=θ rad
    case 'a':
    {
      const auto aa = Eigen::AngleAxis<typename Q::Scalar>(q);
      const auto ax = aa.axis();
      out = std::format_to(out, "axis=[{}, {}, {}]  angle={} rad", fs(ax.x()),
                           fs(ax.y()), fs(ax.z()), fs(aa.angle()));
      break;
    }

    // Rotation matrix (delegates to Matrix<Scalar,3,3> formatter)
    case 'r':
    {
      const auto R = q.toRotationMatrix();
      // Delegate: just use the Matrix formatter with 'm' layout
      out = std::format_to(out, "{:m}", R);
      break;
    }

    // With type header
    case 't':
    {
      out = std::format_to(out, "Quaternion<{}>{{", scalar_name());
      formatter<Q> inner;
      inner.mode_ = 'q';
      inner.prec_ = prec_;
      out         = inner.format(q, ctx);
      out         = std::format_to(out, "}}");
      break;
    }

    default:
      break;
    }

    return out;
  }
};

} // namespace std

// ─────────────────────────────────────────────────────────────────────────────
//  §5  AngleAxis formatter
//  Uses C++26 reflection to obtain the Scalar name from the specialisation.
// ─────────────────────────────────────────────────────────────────────────────

export namespace std
{

template <typename Scalar>
  requires std::is_floating_point_v<Scalar>
struct formatter<Eigen::AngleAxis<Scalar>>
{
private:
  int prec_ = 6;

  // Reflection: get canonical type name
  static consteval std::string_view scalar_name()
  {
    return std::meta::display_string_of(^^Scalar);
  }

public:
  constexpr auto parse(std::format_parse_context &ctx)
  {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == '.')
    {
      ++it;
      prec_ = 0;
      while (it != ctx.end() && std::isdigit((unsigned char) *it))
      {
        prec_ = prec_ * 10 + (*it++ - '0');
      }
    }
    return it;
  }

  template <typename OutputIt>
  auto format(const Eigen::AngleAxis<Scalar> &aa,
              std::basic_format_context<OutputIt, char> &ctx) const
  {
    const std::string sf = std::format("{{:.{}g}}", prec_);
    auto fs
        = [&](Scalar v) { return std::vformat(sf, std::make_format_args(v)); };
    const auto &ax = aa.axis();
    return std::format_to(ctx.out(),
                          "AngleAxis<{}>{{ angle={} rad, axis=[{}, {}, {}] }}",
                          scalar_name(), fs(aa.angle()), fs(ax.x()), fs(ax.y()),
                          fs(ax.z()));
  }
};

} // namespace std

// ─────────────────────────────────────────────────────────────────────────────
//  §6  Transform (Isometry / Affine) formatter
//
//  Eigen::Transform<Scalar, Dim, Mode, Options>
//  Uses reflection to extract template arguments:
//    [0] = Scalar type
//    [1] = Dim  (int NTTP)
//    [2] = Mode (int NTTP: Isometry, Affine, AffineCompact, Projective)
// ─────────────────────────────────────────────────────────────────────────────

export namespace std
{

template <typename Scalar, int Dim, int Mode, int Options>
  requires std::is_floating_point_v<Scalar>
struct formatter<Eigen::Transform<Scalar, Dim, Mode, Options>>
{
  using T = Eigen::Transform<Scalar, Dim, Mode, Options>;

private:
  int prec_ = 6;

  static consteval std::string_view mode_name()
  {
    if constexpr (Mode == Eigen::Isometry)
    {
      return "Isometry";
    }
    else if constexpr (Mode == Eigen::Affine)
    {
      return "Affine";
    }
    else if constexpr (Mode == Eigen::AffineCompact)
    {
      return "AffineCompact";
    }
    else
    {
      return "Projective";
    }
  }

  // Reflection: build the full type display string
  static consteval std::string type_display()
  {
    return std::format("Transform<{},{}D,{}>",
                       std::meta::display_string_of(^^Scalar), Dim,
                       mode_name());
  }

public:
  constexpr auto parse(std::format_parse_context &ctx)
  {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == '.')
    {
      ++it;
      prec_ = 0;
      while (it != ctx.end() && std::isdigit((unsigned char) *it))
      {
        prec_ = prec_ * 10 + (*it++ - '0');
      }
    }
    return it;
  }

  template <typename OutputIt>
  auto format(const T &tf, std::basic_format_context<OutputIt, char> &ctx) const
  {
    // Obtain the underlying (Dim+1)×(Dim+1) matrix
    auto out = std::format_to(ctx.out(), "{}\n  matrix:\n{:m}", type_display(),
                              tf.matrix()); // delegates to Matrix formatter
    return out;
  }
};

} // namespace std

// ─────────────────────────────────────────────────────────────────────────────
//  §7  Universal reflection-driven formatter (P2996R13 §3.12 pattern)
//
//  For any Eigen type T that exposes public data members, this formatter
//  walks all non-static data members via std::meta::nonstatic_data_members_of
//  and formats each one, producing output like:
//
//      TypeName{ .member1=value1, .member2=value2 }
//
//  This is the "universal_formatter" from P2996R13 §3.12, adapted for Eigen.
// ─────────────────────────────────────────────────────────────────────────────

namespace eigen_fmt
{

struct universal_eigen_formatter
{
  constexpr auto parse(auto &ctx)
  {
    return ctx.begin();
  }

  template <typename T>
  auto format(const T &obj, auto &ctx) const
  {
    // ── Reflection: get the unqualified type name ──────────────────
    auto out = std::format_to(ctx.out(), "{}{{", std::meta::identifier_of(^^T));

    bool first = true;
    auto delim = [&]
    {
      if (!first)
      {
        out = std::format_to(out, ", ");
      }
      first = false;
    };

    // ── Walk base classes ──────────────────────────────────────────
    [:expand(std::meta::bases_of(^^T, std::meta::access_context::unchecked())):]
        >> [&]<auto base>
    {
      delim();
      out = std::format_to(
          out, "{}",
          static_cast<typename[:std::meta::type_of(base):] const &>(obj)
      );
    };

    // ── Walk non-static data members ───────────────────────────────
    [:expand(std::meta::nonstatic_data_members_of(
          ^^T, std::meta::access_context::unchecked()
      )):]
        >> [&]<auto mem>
    {
      delim();
      out = std::format_to(out, ".{}=", std::meta::identifier_of(mem));
      // Use a nested formatter for each member's type
      using MemberType = typename[:std::meta::type_of(mem):];
      std::formatter<MemberType> mf;
      std::format_parse_context pctx("");
      (void) mf.parse(pctx);
      ctx.advance_to(out);
      out = mf.format(obj.[:mem:], ctx);
    };

    *out++ = '}';
    return out;
  }
};

} // namespace eigen_fmt

// ── Opt-in for concrete Eigen types ──────────────────────────────────────────
template <>
struct std::formatter<Eigen::IOFormat> : eigen_fmt::universal_eigen_formatter
{
};
