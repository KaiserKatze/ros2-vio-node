// clang-format on

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/core.hpp>

#include <boost/numeric/odeint.hpp>

#define ENABLE_PUBLISH_TEXT 0

using namespace std::chrono_literals;
using MsgPose = geometry_msgs::msg::PoseStamped;
using MsgPath = nav_msgs::msg::Path;
using MsgImu  = sensor_msgs::msg::Imu;

static constexpr char DEFAULT_FRAME_ID[]{"map"};
static constexpr double RADIUS{1.0};
static constexpr double ANGULAR_VELOCITY{10.0 / 180.0 * 3.14159}; // 10 deg/s
// static constexpr double LINEAR_VELOCITY{10.0};

template <typename NodeType> class GroundTruthPublisher
{
public:
  GroundTruthPublisher(NodeType *node_ptr) : node_ptr_{node_ptr}
  {
    const rclcpp::QoS qos(10);
    publisher_path_
        = node_ptr_->template create_publisher<MsgPath>("/path_gt", qos);
    msg_pose_.header.frame_id    = DEFAULT_FRAME_ID;
    msg_path_.header.frame_id    = DEFAULT_FRAME_ID;
    msg_pose_.pose.orientation.w = 1.0;
    msg_pose_.pose.orientation.x = 0.0;
    msg_pose_.pose.orientation.y = 0.0;
    msg_pose_.pose.orientation.z = 0.0;
  }

  void TimerCallback()
  {
    const rclcpp::Time now{node_ptr_->get_clock()->now()};
    const double t_now{now.seconds()};
    const double t_delta{t_now - t_start};

    if (isFirstMessage)
    {
      isFirstMessage = false;
      t_start        = now.seconds();

      msg_pose_.pose.position.x = RADIUS;
      msg_pose_.pose.position.y = 0.0;
      msg_pose_.pose.position.z = 0.0;
    }
    else
    {
      msg_pose_.pose.position.x = RADIUS * std::cos(ANGULAR_VELOCITY * t_delta);
      msg_pose_.pose.position.y = RADIUS * std::sin(ANGULAR_VELOCITY * t_delta);
      msg_pose_.pose.position.z = 0.0;
    }

    msg_pose_.header.stamp = now;
    msg_path_.header.stamp = now;
    msg_path_.poses.push_back(msg_pose_);

    publisher_path_->publish(msg_path_);
  }

private:
  NodeType *node_ptr_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_path_;
  MsgPath msg_path_;
  MsgPose msg_pose_;
  bool isFirstMessage{true};
  double t_start{0.0};
};

template <typename NodeType> class ImuPublisher
{
public:
  ImuPublisher(NodeType *node_ptr) : node_ptr_{node_ptr}
  {
    const rclcpp::QoS qos(10);
    publisher_imu_ = node_ptr_->template create_publisher<MsgImu>("/imu1", qos);
    msg_imu_.header.frame_id    = DEFAULT_FRAME_ID;
    msg_imu_.orientation.w      = 1.0;
    msg_imu_.orientation.x      = 0.0;
    msg_imu_.orientation.y      = 0.0;
    msg_imu_.orientation.z      = 0.0;
    msg_imu_.angular_velocity.x = 0.0;
    msg_imu_.angular_velocity.y = 0.0;
    msg_imu_.angular_velocity.z = 0.0;
  }

  void TimerCallback()
  {
    const rclcpp::Time now{node_ptr_->get_clock()->now()};
    const double t_now{now.seconds()};
    double t_delta{t_now - t_start};

    if (isFirstMessage)
    {
      isFirstMessage = false;
      t_start        = now.seconds();
      t_delta        = 0.0;
    }

    // 速度是位移的导数
    // vx = - RADIUS * std::sin(ANGULAR_VELOCITY * t_delta) * ANGULAR_VELOCITY;
    // vy = RADIUS * std::cos(ANGULAR_VELOCITY * t_delta) * ANGULAR_VELOCITY;
    // 加速度是速度的导数
    msg_imu_.header.stamp = now;
    t_delta *= ANGULAR_VELOCITY;
    const double aar{-ANGULAR_VELOCITY * ANGULAR_VELOCITY * RADIUS};
    msg_imu_.linear_acceleration.x = aar * std::cos(t_delta);
    msg_imu_.linear_acceleration.y = aar * std::sin(t_delta);
    msg_imu_.linear_acceleration.z = 0.0;
    publisher_imu_->publish(msg_imu_);
  }

private:
  NodeType *node_ptr_;
  rclcpp::Publisher<MsgImu>::SharedPtr publisher_imu_;
  MsgImu msg_imu_;
  bool isFirstMessage{true};
  double t_start{0.0};
};

template <typename NodeType> class ImuPathPublisher
{
private:
  struct ImuState : public std::array<double, 10>
  {
    ImuState()
        : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0, 0.0}
    {
    }

    double &GetPositionX()
    {
      return (*this)[0];
    }

    double &GetPositionY()
    {
      return (*this)[1];
    }

    double &GetPositionZ()
    {
      return (*this)[2];
    }

    double &GetVelocityX()
    {
      return (*this)[3];
    }

    double &GetVelocityY()
    {
      return (*this)[4];
    }

    double &GetVelocityZ()
    {
      return (*this)[5];
    }

    double &GetQuaternionW()
    {
      return (*this)[6];
    }

    double &GetQuaternionX()
    {
      return (*this)[7];
    }

    double &GetQuaternionY()
    {
      return (*this)[8];
    }

    double &GetQuaternionZ()
    {
      return (*this)[9];
    }

    double GetPositionX() const
    {
      return (*this)[0];
    }

    double GetPositionY() const
    {
      return (*this)[1];
    }

    double GetPositionZ() const
    {
      return (*this)[2];
    }

    cv::Vec3d GetPosition() const
    {
      return cv::Vec3d(GetPositionX(), GetPositionY(), GetPositionZ());
    }

    double GetVelocityX() const
    {
      return (*this)[3];
    }

    double GetVelocityY() const
    {
      return (*this)[4];
    }

    double GetVelocityZ() const
    {
      return (*this)[5];
    }

    cv::Vec3d GetVelocity() const
    {
      return cv::Vec3d(GetVelocityX(), GetVelocityY(), GetVelocityZ());
    }

    double GetQuaternionW() const
    {
      return (*this)[6];
    }

    double GetQuaternionX() const
    {
      return (*this)[7];
    }

    double GetQuaternionY() const
    {
      return (*this)[8];
    }

    double GetQuaternionZ() const
    {
      return (*this)[9];
    }

    void SetPosition(double px, double py, double pz)
    {
      (*this)[0] = px;
      (*this)[1] = py;
      (*this)[2] = pz;
    }

    void SetPosition(const cv::Vec3d &pos)
    {
      (*this)[0] = pos[0];
      (*this)[1] = pos[1];
      (*this)[2] = pos[2];
    }

    void SetVelocity(double vx, double vy, double vz)
    {
      (*this)[3] = vx;
      (*this)[4] = vy;
      (*this)[5] = vz;
    }

    void SetVelocity(const cv::Vec3d &vel)
    {
      (*this)[3] = vel[0];
      (*this)[4] = vel[1];
      (*this)[5] = vel[2];
    }

    void SetQuaternion(double qw, double qx, double qy, double qz)
    {
      (*this)[6] = qw;
      (*this)[7] = qx;
      (*this)[8] = qy;
      (*this)[9] = qz;
    }

    void SetQuaternion(const cv::Vec4f &quat)
    {
      (*this)[6] = quat[0];
      (*this)[7] = quat[1];
      (*this)[8] = quat[2];
      (*this)[9] = quat[3];
    }

    void NormalizeQuaternion()
    {
      double &qw{GetQuaternionW()};
      double &qx{GetQuaternionX()};
      double &qy{GetQuaternionY()};
      double &qz{GetQuaternionZ()};
      const double q_norm{std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz)};
      qw /= q_norm;
      qx /= q_norm;
      qy /= q_norm;
      qz /= q_norm;
    }

    cv::Vec3d
    TransformAcceleration(const cv::Vec3d &a /* 机体坐标系中的加速度 */) const
    {
      // 提取当前姿态四元数
      const double qw{GetQuaternionW()};
      const double qx{GetQuaternionX()};
      const double qy{GetQuaternionY()};
      const double qz{GetQuaternionZ()};
      // 四元数转旋转矩阵 R (用于将机体加速度转换到世界坐标系)
      const double R00{1.0 - 2.0 * qy * qy - 2.0 * qz * qz};
      const double R01{2.0 * (qx * qy - qw * qz)};
      const double R02{2.0 * (qx * qz + qw * qy)};
      const double R10{2.0 * (qx * qy + qw * qz)};
      const double R11{1.0 - 2.0 * qx * qx - 2.0 * qz * qz};
      const double R12{2.0 * (qy * qz - qw * qx)};
      const double R20{2.0 * (qx * qz - qw * qy)};
      const double R21{2.0 * (qy * qz + qw * qx)};
      const double R22{1.0 - 2.0 * qx * qx - 2.0 * qy * qy};
      // 输出：世界坐标系中的加速度
      return {R00 * a[0] + R01 * a[1] + R02 * a[2],
              R10 * a[0] + R11 * a[1] + R12 * a[2],
              R20 * a[0] + R21 * a[1] + R22 * a[2]};
    }
  };

  struct ImuDerivative : public std::array<double, 10>
  {
    ImuDerivative()
        : std::array<double, 10>{0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0}
    {
    }

    void SetVelocity(const cv::Vec3d &velocity)
    {
      (*this)[0] = velocity[0];
      (*this)[1] = velocity[1];
      (*this)[2] = velocity[2];
    }

    void SetAcceleration(const cv::Vec3d &acceleration)
    {
      (*this)[3] = acceleration[0];
      (*this)[4] = acceleration[1];
      (*this)[5] = acceleration[2];
    }

    void SetQuaternionDerivative(const cv::Vec4d &quat_derivative)
    {
      (*this)[6] = quat_derivative[0];
      (*this)[7] = quat_derivative[1];
      (*this)[8] = quat_derivative[2];
      (*this)[9] = quat_derivative[3];
    }
  };

  struct ImuKinematicsODE
  {
    cv::Vec3d a; // 经过滤波后的加速度 (机体坐标系)
    cv::Vec3d w; // 经过滤波后的角速度 (机体坐标系)

    ImuKinematicsODE(const cv::Vec3d &accel, const cv::Vec3d &gyro)
        : a{accel}, w{gyro}
    {
    }

    void operator()(const ImuState &x, ImuDerivative &dxdt,
                    const double /* t */) const
    {
      // 提取当前姿态四元数
      const double qw{x.GetQuaternionW()};
      const double qx{x.GetQuaternionX()};
      const double qy{x.GetQuaternionY()};
      const double qz{x.GetQuaternionZ()};

      // 位置导数 = 速度
      dxdt.SetVelocity(x.GetVelocity());

      // 速度导数 = 加速度
      const cv::Vec3d accInWorldFrame{x.TransformAcceleration(a)};
      dxdt.SetAcceleration(accInWorldFrame);

      // 姿态导数
      const cv::Vec4d attDerivative{
          0.5 * (-qx * w[0] - qy * w[1] - qz * w[2]),
          0.5 * (qw * w[0] + qy * w[2] - qz * w[1]),
          0.5 * (qw * w[1] - qx * w[2] + qz * w[0]),
          0.5 * (qw * w[2] + qx * w[1] - qy * w[0]),
      };
      dxdt.SetQuaternionDerivative(attDerivative);
    }
  };

public:
  ImuPathPublisher(NodeType *node_ptr) : node_ptr_{node_ptr}
  {
    using std::placeholders::_1;
    const rclcpp::QoS qos(10);
    subscriber_imu_ = node_ptr_->template create_subscription<MsgImu>(
        "/imu1", qos,
        std::bind(&ImuPathPublisher::SubscriberCallback, this, _1));
    publisher_path_
        = node_ptr_->template create_publisher<MsgPath>("/path_imu", qos);
    msg_pose_.header.frame_id = DEFAULT_FRAME_ID;
    msg_path_.header.frame_id = DEFAULT_FRAME_ID;
    this->state_[0] = RADIUS;
    this->state_[4] = RADIUS * ANGULAR_VELOCITY;
  }

  void SubscriberCallback(const MsgImu::ConstSharedPtr &msg)
  {
    const rclcpp::Time now{msg->header.stamp};
    const double gx{msg->angular_velocity.x};
    const double gy{msg->angular_velocity.y};
    const double gz{msg->angular_velocity.z};
    const double ax{msg->linear_acceleration.x};
    const double ay{msg->linear_acceleration.y};
    const double az{msg->linear_acceleration.z};

    const ImuKinematicsODE ode{cv::Vec3d{ax, ay, az}, cv::Vec3d{gx, gy, gz}};
    rk4_.do_step(ode, state_, 0.0, 0.1);
    state_.NormalizeQuaternion();

    msg_pose_.header.stamp       = now;
    msg_pose_.pose.position.x    = state_.GetPositionX();
    msg_pose_.pose.position.y    = state_.GetPositionY();
    msg_pose_.pose.position.z    = state_.GetPositionZ();
    msg_pose_.pose.orientation.w = state_.GetQuaternionW();
    msg_pose_.pose.orientation.x = state_.GetQuaternionX();
    msg_pose_.pose.orientation.y = state_.GetQuaternionY();
    msg_pose_.pose.orientation.z = state_.GetQuaternionZ();
    msg_path_.header.stamp       = now;
    msg_path_.poses.push_back(msg_pose_);

    publisher_path_->publish(msg_path_);
  }

private:
  NodeType *node_ptr_;
  rclcpp::Subscription<MsgImu>::SharedPtr subscriber_imu_;
  rclcpp::Publisher<MsgPath>::SharedPtr publisher_path_;
  MsgPath msg_path_;
  MsgPose msg_pose_;
  ImuState state_;
  boost::numeric::odeint::runge_kutta4<ImuState, double, ImuDerivative> rk4_;
};

class ImuSimNode : public rclcpp::Node,
                   public GroundTruthPublisher<ImuSimNode>,
                   public ImuPublisher<ImuSimNode>,
                   public ImuPathPublisher<ImuSimNode>
{
public:
  ImuSimNode()
      : Node("minimal_publisher"), GroundTruthPublisher(this),
        ImuPublisher(this), ImuPathPublisher(this)
  {
    const rclcpp::QoS qos(10);
#if ENABLE_PUBLISH_TEXT
    publisher_text_
        = this->create_publisher<std_msgs::msg::String>("/sim_text", qos);
#endif

    auto timer_callback = [&]() -> void
    {
#if ENABLE_PUBLISH_TEXT
      std_msgs::msg::String message_text;
      message_text.data = "Hello, world! " + std::to_string(count_++);
      RCLCPP_INFO(get_logger(), "Publishing: '%s'", message_text.data.c_str());
      publisher_text_->publish(message_text);
#endif
      GroundTruthPublisher::TimerCallback();
      ImuPublisher::TimerCallback();
    };

    timer_ = this->create_wall_timer(100ms, timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
#if ENABLE_PUBLISH_TEXT
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_text_;
  size_t count_{0};
#endif
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuSimNode>());
  rclcpp::shutdown();
  return 0;
}
