FROM ubuntu:noble

# @see: https://learn.microsoft.com/zh-cn/vcpkg/get_started/get-started?pivots=shell-powershell
# docker run -it --name vio ubuntu:noble /bin/bash

USER root

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential git awk \
        wget curl \
        gnupg lsb-release

#==================================================================================================================
# GCC
#
# @see: https://gcc.gnu.org/install/prerequisites.html
# @see: https://github.com/KaiserKatze/Blog/wiki/0x0301-C-&-CPP
#==================================================================================================================

RUN apt-get install -y \
        libmpfr-dev libgmp3-dev libmpc-dev flex bison

WORKDIR /app
RUN mkdir gcc && cd gcc && git init && git remote add origin https://gcc.gnu.org/git/gcc.git && \
    export GCC_LATEST_TAG=$(git ls-remote --tags 2>/dev/null | awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^gcc-[0-9]+\.[0-9]+\.[0-9]+$/ {print tag}' | sort -V | tail -1) && \
    git pull -r --depth=1 origin $GCC_LATEST_TAG && \
    ./contrib/download_prerequisites && \
    mkdir build && cd build && \
    ../configure --prefix=/usr/local/gcc-16 \
                 --enable-languages=c,c++ \
                 --disable-multilib \
                 --disable-bootstrap && \
    make -j$(nproc) && make install-strip && \
    update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-16/bin/gcc 160 \
                        --slave /usr/bin/g++ g++ /usr/local/gcc-16/bin/g++ \
                        --slave /usr/bin/gcov gcov /usr/local/gcc-16/bin/gcov && \
    echo "/usr/local/gcc-16/lib64" | tee /etc/ld.so.conf.d/gcc-16.conf && \
    ldconfig

# 清理
WORKDIR /app
RUN rm -rf gcc

#==================================================================================================================
# GNU Make
#==================================================================================================================

WORKDIR /app
RUN mkdir gmake && cd gmake && \
    export GMAKE_VERSION=4.4.1 && \
    wget "https://mirrors.cqu.edu.cn/gnu/make/make-$GMAKE_VERSION.tar.gz" && \
    tar xf "make-$GMAKE_VERSION.tar.gz" && \
    cd "make-$GMAKE_VERSION" && \
    export GMAKE_PREFIX=/opt/gmake && \
    ./configure --prefix=/opt/gmake && make && make install && \
    export PATH=$GMAKE_PREFIX/bin:$PATH && \
    echo "export PATH=$GMAKE_PREFIX/bin:$PATH" >> /etc/profile

# 清理
WORKDIR /app
RUN rm -rf gmake

#==================================================================================================================
# CMake
#
# @see: https://cmake.org/download/
# @see: https://cmake.org/cmake/help/latest/manual/cmake.1.html
#==================================================================================================================

WORKDIR /app
RUN mkdir cmake && cd cmake && \
    export CMAKE_VERSION=4.3.3 && \
    wget "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" && \
    tar xf "cmake-$CMAKE_VERSION.tar.gz" && \
    cd "cmake-$CMAKE_VERSION" && \
    export CMAKE_PREFIX=/usr/local/cmake-$CMAKE_VERSION && \
    ./bootstrap --prefix=$CMAKE_PREFIX && \
    make -j$(nproc) && \
    make install && \
    export PATH=$CMAKE_PREFIX/bin:$PATH && \
    echo "export PATH=$CMAKE_PREFIX/bin:$PATH" >> /etc/profile

# 清理
WORKDIR /app
RUN rm -rf cmake

#==================================================================================================================
# Eigen (Vectors & Matrices)
#
# @see: https://libeigen.gitlab.io/eigen/docs-5.0/GettingStarted.html
#==================================================================================================================

WORKDIR /app
RUN mkdir eigen && cd eigen && git init && git remote add origin https://gitlab.com/libeigen/eigen.git && \
    export EIGEN_LATEST_TAG=$(git ls-remote --tags 2>/dev/null | awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ { print tag } ' | sort -V | tail -1) && \
    git pull -r --depth=1 origin $EIGEN_LATEST_TAG && \
    mkdir build && cd build && \
    cmake .. && \
    make install

# 清理
WORKDIR /app
RUN rm -rf eigen

#==================================================================================================================
# GKlib
#==================================================================================================================

WORKDIR /app
RUN git clone --depth 1 -- https://github.com/KarypisLab/GKlib.git gklib && \
    cd gklib && \
    make config cc=gcc prefix=/usr/local && \
    make && make install

# 清理
WORKDIR /app
RUN rm -rf gklib

#==================================================================================================================
# METIS
#
# @depends GKlib
#==================================================================================================================

WORKDIR /app
RUN git clone --depth 1 -- https://github.com/KarypisLab/METIS.git && \
    cd METIS && \
    make config shared=1 cc=gcc shared=1 prefix=/usr/local gklib_path=/usr/local && \
    make install

# 清理
WORKDIR /app
RUN rm -rf metis

#==================================================================================================================
# Ceres Solver (Optimazition)
#
# @depends Eigen
# @see: http://ceres-solver.org/installation.html
#==================================================================================================================

WORKDIR /app
RUN mkdir ceres-solver && cd ceres-solver && git init && git remote add origin https://github.com/ceres-solver/ceres-solver.git && \
    export CERES_SOLVER_LATEST_TAG=$(git ls-remote --tags 2>/dev/null | awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ { print tag } ' | sort -V | tail -1) && \
    git fetch --depth=1 origin tag $CERES_SOLVER_LATEST_TAG && \
    git checkout $CERES_SOLVER_LATEST_TAG && \
    apt-get install -y \
      libgoogle-glog-dev libgflags-dev \
      libatlas-base-dev \
      libsuitesparse-dev && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -D BUILD_TESTING=OFF -D BUILD_BENCHMARKS=OFF -D BUILD_EXAMPLES=OFF .. && \
    make -j2 && make install

# 清理
WORKDIR /app
RUN rm -rf ceres-solver

#==================================================================================================================
# Sophus (Lie Groups)
#
# @depends ceres-solver
# @depends METIS
# @see: https://github.com/strasdat/Sophus/blob/main/scripts/install_ubuntu_deps_incl_ceres.sh
# @see: https://github.com/strasdat/Sophus/blob/main/.github/workflows/main.yml
#==================================================================================================================

WORKDIR /app
RUN mkdir sophus && cd sophus && git init && git remote add origin https://github.com/strasdat/Sophus.git && \
    export SOPHUS_LATEST_TAG=$(git ls-remote --tags 2>/dev/null | awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ { print tag } ' | sort -V | tail -1) && \
    git pull -r --depth=1 origin $SOPHUS_LATEST_TAG && \
    mkdir build && cd build && \
    cmake -D BUILD_SOPHUS_TESTS=OFF .. && \
    make -j2 && make install

# 清理
WORKDIR /app
RUN rm -rf sophus

#==================================================================================================================
# OpenCV (Computer Vision, Image Processing)
#
# @see: https://docs.opencv.org/4.x/d0/d3d/tutorial_general_install.html
# @note: 固定选用 OpenCV 4.x 版本，因为 ROS2 cv_bridge 尚不支持 OpenCV 5
#==================================================================================================================

WORKDIR /app
RUN mkdir opencv_contrib && cd opencv_contrib && git init && git remote add origin https://github.com/opencv/opencv_contrib.git && \
    export OPENCV_CONTRIB_LATEST_TAG=$(git ls-remote --tags 2>/dev/null |\
      awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^4\.[0-9]+\.[0-9]+$/ { print tag } ' |\
      sort -V | tail -1) && \
    git fetch --depth=1 origin tag $OPENCV_CONTRIB_LATEST_TAG && \
    git checkout $OPENCV_CONTRIB_LATEST_TAG

WORKDIR /app
RUN mkdir opencv && cd opencv && git init && git remote add origin https://github.com/opencv/opencv.git && \
    export OPENCV_LATEST_TAG=$(git ls-remote --tags 2>/dev/null |\
      awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^4\.[0-9]+\.[0-9]+$/ { print tag } ' |\
      sort -V | tail -1) && \
    git fetch --depth=1 origin tag $OPENCV_LATEST_TAG && \
    git checkout $OPENCV_LATEST_TAG && \
    rm -rf build && mkdir build && cd build && \
    cmake -DBUILD_JAVA=OFF \
          -DBUILD_OBJC=OFF \
          -DBUILD_KOTLIN_EXTENSIONS=OFF \
          -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -DBUILD_opencv_legacy=OFF \
          -DBUILD_opencv_cudabgsegm=OFF \
          -DBUILD_opencv_tracking=ON \
          -DBUILD_opencv_cannops=OFF \
          -DBUILD_opencv_intensity_transform=OFF \
          -DBUILD_opencv_superres=OFF \
          -DBUILD_opencv_xfeatures2d=OFF \
          -DBUILD_opencv_saliency=OFF \
          -DBUILD_opencv_bioinspired=OFF \
          -DBUILD_opencv_cudafilters=OFF \
          -DBUILD_opencv_cudawarping=OFF \
          -DBUILD_opencv_cudev=OFF \
          -DBUILD_opencv_fuzzy=OFF \
          -DBUILD_opencv_datasets=OFF \
          -DBUILD_opencv_face=OFF \
          -DBUILD_opencv_xobjdetect=OFF \
          -DBUILD_opencv_freetype=OFF \
          -DBUILD_opencv_ximgproc=OFF \
          -DBUILD_opencv_stereo=OFF \
          -DBUILD_opencv_cudacodec=OFF \
          -DBUILD_opencv_rgbd=OFF \
          -DBUILD_opencv_matlab=OFF \
          -DBUILD_opencv_dpm=OFF \
          -DBUILD_opencv_cudaoptflow=OFF \
          -DBUILD_opencv_cudaimgproc=OFF \
          -DBUILD_opencv_phase_unwrapping=OFF \
          -DBUILD_opencv_dnn_superres=OFF \
          -DBUILD_opencv_quality=OFF \
          -DBUILD_opencv_optflow=OFF \
          -DBUILD_opencv_alphamat=OFF \
          -DBUILD_opencv_sfm=OFF \
          -DBUILD_opencv_aruco=OFF \
          -DBUILD_opencv_cvv=OFF \
          -DBUILD_opencv_bgsegm=OFF \
          -DBUILD_opencv_videostab=OFF \
          -DBUILD_opencv_xphoto=OFF \
          -DBUILD_opencv_ccalib=OFF \
          -DBUILD_opencv_dnn_objdetect=OFF \
          -DBUILD_opencv_structured_light=OFF \
          -DBUILD_opencv_plot=OFF \
          -DBUILD_opencv_cudaobjdetect=OFF \
          -DBUILD_opencv_cnn_3dobj=OFF \
          -DBUILD_opencv_viz=ON \
          -DBUILD_opencv_hfs=OFF \
          -DBUILD_opencv_img_hash=OFF \
          -DBUILD_opencv_surface_matching=OFF \
          -DBUILD_opencv_wechat_qrcode=OFF \
          -DBUILD_opencv_cudalegacy=OFF \
          -DBUILD_opencv_julia=OFF \
          -DBUILD_opencv_fastcv=OFF \
          -DBUILD_opencv_cudaarithm=OFF \
          -DBUILD_opencv_line_descriptor=OFF \
          -DBUILD_opencv_mcc=OFF \
          -DBUILD_opencv_dnns_easily_fooled=OFF \
          -DBUILD_opencv_text=OFF \
          -DBUILD_opencv_cudastereo=OFF \
          -DBUILD_opencv_hdf=OFF \
          -DBUILD_opencv_signal=OFF \
          -DBUILD_opencv_reg=OFF \
          -DBUILD_opencv_rapid=OFF \
          -DBUILD_opencv_ovis=OFF \
          -DBUILD_opencv_cudafeatures2d=OFF \
          -DBUILD_opencv_shape=OFF \
          -DBUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && make install

# 清理
WORKDIR /app
RUN rm -rf opencv

#==================================================================================================================
# ROS
#
# @see: https://docs.ros.org/en/jazzy/Installation.html
# @see: https://github.com/ros-perception/image_pipeline
#==================================================================================================================

WORKDIR /app
RUN apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    echo "export LANG=en_US.UTF-8" >> /etc/profile && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    export UBUNTU_CODENAME=$(lsb_release -c 2>/dev/null | awk '{print $2}' | tr '[:upper:]' '[:lower:]') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    export ROS_CODENAME=$([[ "$UBUNTU_CODENAME" == "noble" ]] && echo "jazzy" || [[ "$UBUNTU_CODENAME" == "jammy" ]] && echo "humble" || [[ "$UBUNTU_CODENAME" == "focal" ]] && echo "foxy" || echo "") && \
    echo "export ROS_CODENAME=$ROS_CODENAME" >> /etc/profile && \
    apt-get install -y \
        ros-$ROS_CODENAME-desktop \
        ros-$ROS_CODENAME-image-proc \
        ros-$ROS_CODENAME-message-filters \
        ros-$ROS_CODENAME-tf2-tools transforms3d \
        ros-$ROS_CODENAME-joint-state-publisher ros-$ROS_CODENAME-robot-state-publisher ros-$ROS_CODENAME-xacro \
        ros-$ROS_CODENAME-usb-cam && \
    source /opt/ros/$ROS_CODENAME/setup.bash && \
    echo "source /opt/ros/$ROS_CODENAME/setup.bash" >> /etc/profile

# 安装 rosdep
RUN apt-get install -y \
        python3 python3-pip python3-dev python3-venv \
        python3-numpy python3-pandas python3-matplotlib \
        python3-rosdep && \
    export ROSDEP_SOURCES_LIST_D=/etc/ros/rosdep/sources.list.d && \
    rosdep init || mkdir -p $ROSDEP_SOURCES_LIST_D && \
    wget https://raw.giteeusercontent.com/kaiserkatze/rosdep/raw/master/20-default.list && \
    mv 20-default.list $ROSDEP_SOURCES_LIST_D/20-default.list && \
    rosdep update

# 安装 colcon (用于构建)
RUN apt-get install -y python3-colcon-common-extensions

#==================================================================================================================
# YAML-CPP
#==================================================================================================================

WORKDIR /app
RUN mkdir yamlcpp && cd yamlcpp && git init && git remote add origin https://github.com/jbeder/yaml-cpp.git && \
    export YAMLCPP_LATEST_TAG=$(git ls-remote --tags 2>/dev/null | awk '{ n = split($2, parts, "/"); tag = parts[n] }; tag ~ /^yaml-cpp-[0-9]+\.[0-9]+\.[0-9]+$/ { print tag } ' | sort -V | tail -1) && \
    git pull -r --depth=1 origin $YAMLCPP_LATEST_TAG && \
    mkdir build && cd build && cmake -D YAML_BUILD_SHARED_LIBS=ON .. && \
    make && make install

# 清理
WORKDIR /app
RUN rm -rf yamlcpp

#==================================================================================================================
# 清尾工作
#==================================================================================================================

# @see: https://github.com/Auburn/FastNoise2/wiki

RUN useradd -m kk
USER kk
WORKDIR /home/kk

ENTRYPOINT [ "/bin/bash", "-c", "/etc/profile" ]
