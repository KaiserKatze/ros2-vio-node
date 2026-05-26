#!/bin/bash

colcon build --packages-select euroc_vio \
  --cmake-args -DCMAKE_COMMAND=/home/kk/cmake-4.3.2-linux-x86_64/bin/cmake \
  --event-handlers console_direct+
