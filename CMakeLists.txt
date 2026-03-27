cmake_minimum_required(VERSION 3.8)
project(euroc_vio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# 强制使用 GCC
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "GCC is required to build this project")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-Werror)
add_compile_options(-O2)

# 自动检测 C++ 标准（支持草案和正式标准）
include(CheckCXXCompilerFlag)
set(CMAKE_REQUIRED_FLAGS "")

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

if (CMAKE_CXX_STANDARD LESS 26)
  message(WARNING "Requested C++26, but falling back to C++${DETECTED_CXX_STD}")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(Armadillo REQUIRED)

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

add_executable(euroc_vio src/main.cpp)

target_include_directories(euroc_vio PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)

  if(DETECTED_CXX_STD GREATER_EQUAL 26)
    target_compile_features(euroc_vio PUBLIC cxx_std_26)
  elseif(DETECTED_CXX_STD GREATER_EQUAL 23)
    target_compile_features(euroc_vio PUBLIC cxx_std_23)
  else()
    target_compile_features(euroc_vio PUBLIC cxx_std_20)
  endif()

else()

  message(WARNING "Old GCC detected, falling back to C++20")
  target_compile_features(euroc_vio PUBLIC cxx_std_20)

endif()

ament_target_dependencies(euroc_vio
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_filters
  cv_bridge
  image_transport
  OpenCV
)
install(TARGETS
  euroc_vio
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)
target_link_libraries(euroc_vio
  Eigen3::Eigen
  Boost::boost
  armadillo
  ${OpenCV_LIBS}
)

ament_package()
