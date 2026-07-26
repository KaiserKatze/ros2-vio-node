# CMakeLists.txt —— 构建 msckf (双目视觉 + IMU 紧耦合 MSCKF)
#
# 环境要求: CMake >= 3.25, GCC >= 14 或 Clang >= 18 (C++26 与 <print>)
# Ubuntu 依赖安装:
#   sudo apt install cmake g++ libopencv-dev libeigen3-dev libceres-dev \
#                    libyaml-cpp-dev libgoogle-glog-dev
#   Sophus 为纯头文件库, 需源码安装: git clone https://github.com/strasdat/Sophus
#
# 构建与运行:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j
#   ./msckf EuRoC_MAV_Datasets/V2_01_easy/mav0

cmake_minimum_required(VERSION 3.25)
project(msckf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)

# Sophus 旧版本只提供变量, 统一封装成导入目标
if(NOT TARGET Sophus::Sophus)
  add_library(Sophus::Sophus INTERFACE IMPORTED)
  set_target_properties(Sophus::Sophus PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Sophus_INCLUDE_DIRS}")
endif()

# Ceres 2.x 导出 Ceres::ceres, 1.x 导出 ceres
if(TARGET Ceres::ceres)
  set(CERES_TARGET Ceres::ceres)
else()
  set(CERES_TARGET ceres)
endif()

# yaml-cpp 0.8+ 导出 yaml-cpp::yaml-cpp, 旧版导出 yaml-cpp
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

add_executable(msckf msckf.cpp)

target_include_directories(msckf PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(msckf PRIVATE
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
  ${CERES_TARGET}
  ${YAML_CPP_TARGET}
  glog::glog
  Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(msckf PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3 -march=native>
  )
endif()

message(STATUS "OpenCV:   ${OpenCV_VERSION}")
message(STATUS "Eigen:    ${Eigen3_VERSION}")
message(STATUS "Ceres:    ${Ceres_VERSION} (目标: ${CERES_TARGET})")
message(STATUS "yaml-cpp 目标: ${YAML_CPP_TARGET}")
