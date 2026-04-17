#ifndef INVSQRT_HPP
#define INVSQRT_HPP

#include <cstdint>
#include <type_traits>

template <typename FloatType> struct FloatInt
{
  using type = std::conditional_t<
      sizeof(FloatType) == 1, std::int8_t,
      std::conditional_t<
          sizeof(FloatType) == 2, std::int16_t,
          std::conditional_t<
              sizeof(FloatType) == 4, std::int32_t,
              std::conditional_t<sizeof(FloatType) == 8, std::int64_t, void>>>>;
  static_assert(!std::is_same_v<type, void>, "Unsupported floating type");
};

template <typename FloatType> struct MagicNumber
{
};

template <> struct MagicNumber<double>
{
  static constexpr typename FloatInt<double>::type value{0x5fe6eb50c7b537a9ll};
};

template <> struct MagicNumber<float>
{
  static constexpr typename FloatInt<float>::type value{0x5f3759df};
};

/**
 * @brief Fast inverse square-root calculation
 * @param x Input value
 * @return Inverse square-root of x
 * @note http://en.wikipedia.org/wiki/Fast_inverse_square_root
 */
template <typename FloatType> FloatType invSqrt(FloatType x)
{
  using fint = typename FloatInt<FloatType>::type;
  union
  {
    FloatType f;
    fint i;
  } fi;
  fi.f = x;
  FloatType y{static_cast<FloatType>(0.5) * x};
  fi.i = MagicNumber<FloatType>::value - (fi.i >> 1);
  FloatType z{fi.f};
  z = z * (static_cast<FloatType>(1.5) - (y * z * z));
  return z;
}

#endif /* INVSQRT_HPP */
