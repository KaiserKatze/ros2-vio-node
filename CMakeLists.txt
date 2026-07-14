cmake_minimum_required(VERSION 4.3.2)

# 强制使用 16.0 以上版本的 GCC 编译器
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "GCC (≥ 16.0) is required to build this project")
endif()
if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0")
  message(FATAL_ERROR "GCC (≥ 16.0) is required to build this project")
endif()

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

# --- 模块文件分类 ---
set(CORE_MODULE_FILES
  modules/AbstractLoader.cppm
  modules/CircularBuffer.cppm
  modules/DatumFast.cppm
  modules/DatumImu.cppm
  modules/DatumTruth.cppm
  modules/ErrorStateKalmanFilter.cppm
  modules/Integrator.cppm
  modules/Interpolation.cppm
  modules/SensorState.cppm
  modules/SensorYaml.cppm
  modules/StdReflect.cppm
  modules/StereoObservation.cppm
  modules/ZUPT.cppm
)

set(VISUAL_SIM_MODULE_FILES
  src/VisualSim/Camera.cppm
  src/VisualSim/MeshPlot.cppm
  src/VisualSim/Path.cppm
  src/VisualSim/Room.cppm
  src/VisualSim/StereoRig.cppm
)

set(VISUAL_SLAM_MODULE_FILES
  src/VisualSlam/EuRoC.cppm
  src/VisualSlam/CornerDetection.cppm
  src/VisualSlam/FastDetector.cppm
  src/VisualSlam/ImageDataLoader.cppm
)

# --- 辅助函数定义 ---
function(add_ros2_executable node_name source_path)
  add_executable(${node_name} ${source_path})

  target_include_directories(${node_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # 链接 共享依赖 以及 核心公共模块
  target_link_libraries(${node_name}
    PRIVATE common_deps euroc_vio_modules
  )

  install(TARGETS ${node_name}
    DESTINATION lib/${PROJECT_NAME}
  )
endfunction()

# ===== 共享依赖与包含路径接口库 =====
add_library(common_deps INTERFACE)
target_include_directories(common_deps
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(common_deps
  INTERFACE
  Eigen3::Eigen
  Sophus::Sophus
  Boost::boost
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

# ===== 核心模块库 =====
# 将所有公用模块打包到一个静态库，防止被各个 Node 目标反复编译浪费时间
add_library(euroc_vio_modules STATIC)
target_link_libraries(euroc_vio_modules PUBLIC common_deps)
target_sources(
  euroc_vio_modules
  PUBLIC
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    FILES ${CORE_MODULE_FILES}
)

# ===== 仿真模块库 =====
add_library(euroc_vio_sim_modules STATIC)
target_link_libraries(euroc_vio_sim_modules PUBLIC common_deps)
target_sources(
  euroc_vio_sim_modules
  PUBLIC
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    FILES ${VISUAL_SIM_MODULE_FILES}
)

# ===== 目标： SLAM 仿真数据集生成器 =====
add_ros2_executable(VisualSim src/VisualSim/VisualSim.cpp)
target_link_libraries(VisualSim PRIVATE euroc_vio_sim_modules)

# ===== 目标： EuRoC 纯视觉双目里程计 =====
add_ros2_executable(StereoSlam src/VisualSlam/VisualSlam.cpp)
target_sources(
  StereoSlam
  PRIVATE
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    FILES ${VISUAL_SLAM_MODULE_FILES}
)

# ===== 目标： 融合数据与轨迹估计 =====
add_ros2_executable(VisualInertial src/VisualSim/VisualInertial.cpp)
target_link_libraries(VisualInertial PRIVATE euroc_vio_sim_modules)

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
