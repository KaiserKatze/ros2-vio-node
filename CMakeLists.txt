cmake_minimum_required(VERSION 4.3.2)
project(euroc_vio)

# --- 编译选项设置 ---
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 强制使用 GCC
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "GCC is required to build this project")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-Wshadow)
add_compile_options(-Wuninitialized)
add_compile_options(-Wno-maybe-uninitialized)
add_compile_options(-march=native)

if(MSVC)
  add_compile_options(/utf-8)
endif()

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

  install(TARGETS ${node_name}
    DESTINATION lib/${PROJECT_NAME}
  )
endfunction()

# ===== 目标： SLAM 仿真数据集生成器 =====
add_ros2_executable(VisualSim src/VisualSim/VisualSim.cpp)
ament_target_dependencies(VisualSim
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  cv_bridge
)
target_link_libraries(VisualSim
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

# ===== 目标： 融合数据与轨迹估计 =====
add_ros2_executable(VisualInertial src/VisualSim/VisualInertial.cpp)
ament_target_dependencies(VisualInertial
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
)
target_link_libraries(VisualInertial
  Eigen3::Eigen
  ${OpenCV_LIBS}
  yaml-cpp::yaml-cpp
)
target_compile_options(VisualInertial PRIVATE -std=c++26 -freflection)

# ===== 目标： 加载生成的 CSV 或 GroundTruth 并发布为 ROS2 话题 =====
add_ros2_executable(SimpleDataLoader src/DataLoader/DataLoader.cpp)
ament_target_dependencies(SimpleDataLoader
  rclcpp
  nav_msgs
  geometry_msgs
)
target_link_libraries(SimpleDataLoader
  Eigen3::Eigen
)

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
