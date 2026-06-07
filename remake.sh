#!/bin/bash

set -eux

echo "Home=$HOME"

colcon build --packages-select euroc_vio \
  --cmake-args \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D CMAKE_COMMAND=$HOME/cmake-4.3.2-linux-x86_64/bin/cmake \
  --event-handlers console_direct+

if [ $? -ne 0 ]; then
  rm -rf build log install

  colcon build --packages-select euroc_vio \
    --cmake-args \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_COMMAND=$HOME/cmake-4.3.2-linux-x86_64/bin/cmake \
    --event-handlers console_direct+
fi
