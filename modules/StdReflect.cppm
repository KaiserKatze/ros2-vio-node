export module std_reflect;

import std;

export namespace std
{

// @see: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html#enum-to-string
template <typename E, bool Enumerable = std::meta::is_enumerable_type(^^E)>
  requires std::is_enum_v<E>
constexpr std::string_view enum_to_string(E value)
{
  if constexpr (Enumerable)
  {
    template for (constexpr auto e :
                  std::define_static_array(std::meta::enumerators_of(^^E)))
    {
      if (value == [:e:])
      {
        return std::meta::identifier_of(e);
      }
    }
  }

  return "<unnamed>";
}

} // namespace std
