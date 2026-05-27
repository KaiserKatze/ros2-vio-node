#!/bin/bash

set -eux

echo "Home=$HOME"
rm -rf build log install

# 获取 compile_commands.json
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .

colcon build --packages-select euroc_vio \
  --cmake-args -DCMAKE_COMMAND=$HOME/cmake-4.3.2-linux-x86_64/bin/cmake \
  --event-handlers console_direct+
