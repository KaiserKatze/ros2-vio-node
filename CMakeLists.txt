cmake_minimum_required(VERSION 3.28.3)
project(euroc_vio)

# --- 编译选项设置 ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
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
add_compile_options(-O3)
add_compile_options(-march=native)

if(MSVC)
  add_compile_options(/utf-8)
endif()

# 自动检测 C++ 标准（支持草案和正式标准）
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++2c" HAS_CXX26_DRAFT)
check_cxx_compiler_flag("-std=c++26" HAS_CXX26_FINAL)
check_cxx_compiler_flag("-std=c++2b" HAS_CXX23_DRAFT)
check_cxx_compiler_flag("-std=c++23" HAS_CXX23_FINAL)

if(HAS_CXX26_FINAL)
  message(STATUS "Using C++26 (final)")
  add_compile_options(-std=c++26)
  set(DETECTED_CXX_STD 26)
elseif(HAS_CXX26_DRAFT)
  message(STATUS "Using C++26 (draft: c++2c)")
  add_compile_options(-std=c++2c)
  set(DETECTED_CXX_STD 26)
elseif(HAS_CXX23_FINAL)
  message(STATUS "Using C++23 (final)")
  add_compile_options(-std=c++23)
  set(DETECTED_CXX_STD 23)
elseif(HAS_CXX23_DRAFT)
  message(STATUS "Using C++23 (draft: c++2b)")
  add_compile_options(-std=c++2b)
  set(DETECTED_CXX_STD 23)
else()
  message(WARNING "Falling back to C++20")
  add_compile_options(-std=c++20)
  set(DETECTED_CXX_STD 20)
endif()

if (DETECTED_CXX_STD LESS 26)
  message(WARNING "Requested C++26, but falling back to C++${DETECTED_CXX_STD}")
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
find_package(Armadillo REQUIRED)

# # --- ROS2 通信接口生成 ---
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(AngularDisplacementAndUnitTranslation
#   "msg/AngularDisplacementAndUnitTranslation.msg"
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
# )
# ament_export_dependencies(rosidl_default_runtime)

# --- 辅助函数定义 ---
function(add_ros2_executable node_name source_path)
  add_executable(${node_name} ${source_path})

  target_include_directories(${node_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # # 链接类型支持库 (Jazzy 推荐做法)
  # rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
  # target_link_libraries(${node_name} ${cpp_typesupport_target})

  # # 建立显式依赖关系，确保先生成 msg 头文件再编译节点代码
  # # 使用 ${PROJECT_NAME}__cpp 是最通用的做法
  # add_dependencies(${node_name} ${PROJECT_NAME}__cpp)

  install(TARGETS ${node_name}
    DESTINATION lib/${PROJECT_NAME}
  )
endfunction()

# # --- 编译 IMU 结点：预处理惯性测量单元数据 ---
# add_ros2_executable(euroc_imu src/imu.cpp)
# ament_target_dependencies(euroc_imu
#   rclcpp
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   nav_msgs
# )
# target_link_libraries(euroc_imu
#   Eigen3::Eigen
#   Boost::boost
#   ${OpenCV_LIBS}
# )

# add_ros2_executable(imu_sim src/sim.cpp)
# ament_target_dependencies(imu_sim
#   rclcpp
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   nav_msgs
# )
# target_link_libraries(imu_sim
#   Boost::boost
#   ${OpenCV_LIBS}
# )

# --- 编译 VIO 结点：融合视觉、惯性测量单元数据 ---
# add_ros2_executable(euroc_vio src/vio.cpp)
# ament_target_dependencies(euroc_vio
#   rclcpp
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   message_filters
#   cv_bridge
#   image_transport
#   OpenCV
# )
# target_link_libraries(euroc_vio
#   Eigen3::Eigen
#   Boost::boost
#   armadillo
#   ${OpenCV_LIBS}
# )

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

# # ===== 目标： 误差分析 CSV 生产 =====
# add_executable(euroc_rk4_eval src/euroc_rk4_eval.cpp)
# target_include_directories(euroc_rk4_eval PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>
# )
# install(TARGETS euroc_rk4_eval
#   DESTINATION debug/euroc_rk4_eval
# )
# target_link_libraries(euroc_rk4_eval
#   Eigen3::Eigen
#   Boost::boost
#   ${OpenCV_LIBS}
# )

# # ===== 目标： CSV 拟合与绘图 =====
# add_executable(csv_polyfit_plot src/csv_polyfit_plot.cpp)

# target_include_directories(csv_polyfit_plot PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>
# )
# install(TARGETS csv_polyfit_plot
#   DESTINATION lib/${PROJECT_NAME}
# )
# target_link_libraries(csv_polyfit_plot
#   Eigen3::Eigen
#   ${OpenCV_LIBS}
# )

# # ===== 目标： 纯视觉 SLAM =====
# add_ros2_executable(StereoSlam src/VisualSlam/VisualSlam.cpp)
# ament_target_dependencies(StereoSlam
#   rclcpp
#   geometry_msgs
#   nav_msgs
# )
# target_link_libraries(StereoSlam
#   Eigen3::Eigen
#   ${OpenCV_LIBS}
# )

# ===== 目标： 纯视觉 SLAM 仿真 =====
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

# ===== 目标： 纯视觉仿真数据的朝向估计 =====
add_ros2_executable(EstimationLoader src/VisualSim/EstimationLoader.cpp)
ament_target_dependencies(EstimationLoader
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  cv_bridge
)
target_link_libraries(EstimationLoader
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

# # ===== 目标： 加载 EuRoC 数据集 (V2_01_easy) 的真值 =====
# add_ros2_executable(EuRoC_DataLoader src/EuRoC/DataLoader.cpp)
# ament_target_dependencies(EuRoC_DataLoader
#   rclcpp
#   geometry_msgs
#   nav_msgs
#   sensor_msgs
#   cv_bridge
#   # visualization_msgs
# )
# target_link_libraries(EuRoC_DataLoader
#   Eigen3::Eigen
#   ${OpenCV_LIBS}
# )

# # ===== 目标： 加载任意数据集 =====
# add_ros2_executable(SimpleDataLoader src/DataLoader/DataLoader.cpp)
# ament_target_dependencies(SimpleDataLoader
#   rclcpp
#   nav_msgs
# )
# target_link_libraries(SimpleDataLoader
#   Eigen3::Eigen
#   ${OpenCV_LIBS}
# )

# ===== 目标： 融合数据 =====
add_ros2_executable(FuseKalman src/VisualSim/FuseKalman.cpp)
ament_target_dependencies(FuseKalman
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
)
target_link_libraries(FuseKalman
  Eigen3::Eigen
  ${OpenCV_LIBS}
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
