#!/bin/bash

set -eux

echo "Home=$HOME"

rm -rf build log install

export MAKEFLAGS="-j1"
colcon build --packages-select euroc_vio \
  --parallel-workers 1 \
  --cmake-args \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D CMAKE_COMMAND=$HOME/cmake-4.3.2-linux-x86_64/bin/cmake \
  --event-handlers console_direct+
