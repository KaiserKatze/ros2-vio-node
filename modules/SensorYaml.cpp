module;

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

module FastVIO;

import FastVIO:SensorYaml;

namespace FastVIO
{

SensorYaml::SensorYaml() : transform_matrix_{Eigen::Matrix4d::Identity()} {}

} // namespace FastVIO
