#pragma once

#include <charconv>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

struct AbstractLoader
{
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

  static std::int64_t get_item_as_int64(std::stringstream &ss, char delim = ',')
  {
    std::string item;
    std::getline(ss, item, delim);
    auto sv{trim(item)};
    std::int64_t result{0};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse int64: " + std::string(sv)};
    }
    return result;
  }

  static float get_item_as_float(std::stringstream &ss, char delim = ',')
  {
    std::string item;
    std::getline(ss, item, delim);
    auto sv{trim(item)};
    float result{0.0f};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse float: " + std::string(sv)};
    }
    return result;
  }

  static double get_item_as_double(std::stringstream &ss, char delim = ',')
  {
    std::string item;
    std::getline(ss, item, delim);
    auto sv{trim(item)};
    double result{0.0};
    const char *first{sv.data()};
    const char *last{first + sv.size()};
    auto [ptr, ec] = std::from_chars(first, last, result);
    if (ec != std::errc())
    {
      throw std::runtime_error{"Failed to parse double: " + std::string(sv)};
    }
    return result;
  }
};
