module;

#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

module FastVIO;

import :Integrator;

namespace FastVIO
{

AbstractIntegrator::AbstractIntegrator() :
  gravity_world_{-gravity_world_norm_ * Eigen::Vector3d::UnitZ()}, pose_{},
  linear_velocity_{Eigen::Vector3d::Zero()}, previous_attitude{}
{
}

void VisualIntegrator::Update(const Sophus::SO3d &delta_attitude,
                              const Eigen::Vector3d &delta_position)
{
  Sophus::SO3d estimated_new_attitude{
      pose_.so3() * delta_attitude,
  };

  pose_.translation() += pose_.so3() * delta_position;
  previous_attitude = pose_.so3();
  pose_.so3()       = estimated_new_attitude;
}

} // namespace FastVIO
