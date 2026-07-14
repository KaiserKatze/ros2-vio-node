cmake_minimum_required(VERSION 4.3.2)

# --- 编译选项设置 ---
# @see: https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 要想使用 `import std;` 必须启用 CMake 的实验性功能 `CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`
# @see: https://github.com/Kitware/CMake/blob/v4.3.2/Help/dev/experimental.rst
# @see: https://discourse.cmake.org/t/cmake-experimental-cxx-import-std-value-history-and-state/15712
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.4.0")
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
        "f35a9ac6-8463-4d38-8eec-5d6008153e7d"
    )
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "4.3.0")
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
        "451f2fe2-a8a2-47c3-bc32-94786d8fc91b"
    )
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    if(CMAKE_VERSION VERSION_LESS "3.31.8")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
            "0e5b6991-d74f-4b3d-a41c-cf096e0b2508"
        )
    elseif(CMAKE_VERSION VERSION_LESS "4.0.0")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
            "d0edc3af-4c50-42ea-a356-e2862fe7a444"
        )
    elseif(CMAKE_VERSION VERSION_LESS "4.0.3")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
            "a9e1cf81-9932-4810-974b-6eccaf14e457"
        )
    elseif(CMAKE_VERSION VERSION_LESS "4.3.0")
        set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
            "d0edc3af-4c50-42ea-a356-e2862fe7a444"
        )
    endif()
endif()

set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

project(euroc_vio)

# 强制使用 GCC
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "GCC is required to build this project")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-Wreorder)
add_compile_options(-Wshadow)
add_compile_options(-Wuninitialized)
add_compile_options(-Wno-maybe-uninitialized)
add_compile_options(-march=native)
# C++26 开启反射机制
add_compile_options(-freflection)

# 启用 C++26 反射机制
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0")
    add_compile_options(-freflection)
endif()

# --- 查找依赖 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Armadillo REQUIRED)

# --- 辅助函数定义 ---
function(add_ros2_executable node_name source_path)
  add_executable(${node_name} ${source_path})

  target_include_directories(${node_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(${node_name}
    PRIVATE common_pch
  )

  install(TARGETS ${node_name}
    DESTINATION lib/${PROJECT_NAME}
  )
endfunction()

# ===== 共享预编译头接口库 =====
add_library(common_pch INTERFACE)
target_precompile_headers(common_pch
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/euroc_vio/pch.hpp
)
target_include_directories(common_pch
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(common_pch
  INTERFACE
  Eigen3::Eigen
  ${OpenCV_LIBRARIES}
  yaml-cpp::yaml-cpp

  rclcpp::rclcpp

  cv_bridge::cv_bridge
  message_filters::message_filters
  image_transport::image_transport

  nav_msgs::nav_msgs__rosidl_typesupport_cpp
  nav_msgs::nav_msgs__rosidl_generator_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
  geometry_msgs::geometry_msgs__rosidl_generator_cpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  sensor_msgs::sensor_msgs__rosidl_generator_cpp
  std_msgs::std_msgs__rosidl_typesupport_cpp
  std_msgs::std_msgs__rosidl_generator_cpp
)

# ===== 目标： SLAM 仿真数据集生成器 =====
add_ros2_executable(VisualSim src/VisualSim/VisualSim.cpp)

# ===== 目标： EuRoC 纯视觉双目里程计 =====
add_ros2_executable(StereoSlam src/VisualSlam/VisualSlam.cpp)

# ===== 目标： 融合数据与轨迹估计 =====
add_ros2_executable(VisualInertial src/VisualSim/VisualInertial.cpp)

# ===== 目标： 加载生成的 CSV 或 GroundTruth 并发布为 ROS2 话题 =====
add_ros2_executable(SimpleDataLoader src/DataLoader/DataLoader.cpp)

# --- 资源安装 ---
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS scripts/ESKF_Optimizer/opt.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES scripts/ESKF_Optimizer/config.yaml
  DESTINATION lib/${PROJECT_NAME}
)

# --- 单元测试 ---
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
