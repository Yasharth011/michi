#pragma once

#include "common.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

#include <cobs.h>
#include <gz/msgs.hh>
#include <gz/transport/Node.hh>
#include <Eigen/Geometry>

using asio::ip::tcp;
using Eigen::Vector4d;
using Eigen::Vector3d;

enum class GazeboErrc
{
  Success = 0,
  FailedWrite = 1, // System failure
  FailedRead,
  TransmitTimeout = 10, // Timeouts
  ReceiveTimeout,
};
struct GazeboErrCategory : std::error_category
{
  const char* name() const noexcept override
  {
    return "GazeboProxyCommunication";
  }
  std::string message(int ev) const override
  {
    switch (static_cast<GazeboErrc>(ev)) {
      case GazeboErrc::FailedWrite:
        return "could not write, asio error";
      case GazeboErrc::FailedRead:
        return "could not read, asio error";
      case GazeboErrc::ReceiveTimeout:
        return "did not get response, timed out";
      case GazeboErrc::TransmitTimeout:
        return "could not send message, timed out";
      default:
        return "(unrecognized error)";
    }
  }
};
const GazeboErrCategory gz_errc_category;
std::error_code
make_error_code(GazeboErrc e)
{
  return { static_cast<int>(e), gz_errc_category };
}
namespace std {
template<>
struct is_error_code_enum<GazeboErrc> : true_type
{};
}
struct GazeboState {
  Vector3d m_odometry_position;
  Vector4d m_odometry_velocity_heading;
  Vector3d m_imu_linear_acceleration;
  Vector3d m_imu_angular_velocity;
  Vector3d m_magnetic_field;
  gz::msgs::PointCloudPacked m_packed_pointcloud;
};
struct GazeboCommands {
  gz::msgs::Twist m_cmd_vel;
  gz::msgs::Vector3d m_cmd_vel_linear;
  gz::msgs::Vector3d m_cmd_vel_angular;

  ~GazeboCommands() {
    auto l = m_cmd_vel.release_linear();
    auto a = m_cmd_vel.release_angular();
  }
};

class GazeboInterface {
  GazeboState m_gz_state;
  GazeboCommands m_gz_cmd;
  gz::transport::Node m_gz_node;
  gz::transport::Node::Publisher m_cmd_vel_pub;
  std::vector<std::string> m_subscribed_topics;

  inline size_t cobs_buffer_size(size_t input_size,
                                 bool with_trailing_zero = false)
  {
    size_t output_size = input_size + input_size / 254 + 1;
    if (with_trailing_zero)
      output_size++;
    return output_size;
  }
  auto update_base_imu(const gz::msgs::IMU &imu) -> void
  {
    m_gz_state.m_imu_linear_acceleration << imu.linear_acceleration().x(), imu.linear_acceleration().y(), imu.linear_acceleration().z();
    m_gz_state.m_imu_angular_velocity << imu.angular_velocity().x(), imu.angular_velocity().y(), imu.angular_velocity().z();
    spdlog::trace("Got imu: {}, {}", m_gz_state.m_imu_linear_acceleration,m_gz_state.m_imu_angular_velocity);
  }
  auto update_base_imu(std::string_view msg_view) -> void
  {
    gz::msgs::IMU imu;
    if (not imu.ParseFromString(msg_view)) {
      spdlog::error("Tag len: {}. Couldn't parse gz::msgs::IMU from received proxy message.", msg_view.size());
      return;
    }
    m_gz_state.m_imu_linear_acceleration << imu.linear_acceleration().x(), imu.linear_acceleration().y(), imu.linear_acceleration().z();
    m_gz_state.m_imu_angular_velocity << imu.angular_velocity().x(), imu.angular_velocity().y(), imu.angular_velocity().z();
    spdlog::debug("Got imu: {}, {}", m_gz_state.m_imu_linear_acceleration,m_gz_state.m_imu_angular_velocity);
  }
  auto update_magnetic_field(const gz::msgs::Magnetometer& mag) -> void {
    m_gz_state.m_magnetic_field << mag.field_tesla().x(), mag.field_tesla().y(),
      mag.field_tesla().z();
  }
  auto update_odometry(const gz::msgs::Odometry& odom) -> void {
    m_gz_state.m_odometry_position << odom.pose().position().x(), odom.pose().position().y(), odom.pose().position().z();

    double w = odom.pose().orientation().w(), z = odom.pose().orientation().z();
    double odom_heading = 2*acos(z);
    m_gz_state.m_odometry_velocity_heading
      << odom.twist().linear().x() * cos(odom_heading),
      -odom.twist().linear().x() * sin(odom_heading),
      odom.twist().angular().z(), odom_heading;

    spdlog::trace("Got odom: {}", m_gz_state.m_odometry_position);
  }
  auto update_odometry(std::string_view msg_view) -> void {
    gz::msgs::Odometry odom;
    if (not odom.ParseFromString(msg_view)) {
      spdlog::error("Tag len: {}. Couldn't parse gz::msgs::Odometry from received proxy message.", msg_view.size());
      return;
    }
    m_gz_state.m_odometry_position << odom.pose().position().x(), odom.pose().position().y(), odom.pose().position().z();
    spdlog::debug("Got odom: {}", m_gz_state.m_odometry_position);
  }
  auto update_pointcloud(const gz::msgs::PointCloudPacked& packed_pointcloud) -> void {
    m_gz_state.m_packed_pointcloud = packed_pointcloud;
    spdlog::trace("Got pointcloud with size {}", m_gz_state.m_packed_pointcloud.data().size());
  }
  auto update_pointcloud(std::string_view msg_view) -> void {
    gz::msgs::PointCloudPacked packed_pointcloud;
    if (not packed_pointcloud.ParseFromString(msg_view)) {
      spdlog::error("Tag len: {}. Couldn't parse gz::msgs::PointCloudPacked from received proxy message.", msg_view.size());
      return;
    }
    m_gz_state.m_packed_pointcloud = packed_pointcloud;
    spdlog::debug("Got pointcloud with size {}", m_gz_state.m_packed_pointcloud.data().size());
  }
  auto handle_message(std::string& cobs_buffer, int cobs_len) -> void {
    std::vector<char> buffer(cobs_len-1);
    int valid_len = 0;
    if (auto result = cobs_decode(buffer.data(), buffer.size(), cobs_buffer.data(), cobs_len-1); result.status != COBS_DECODE_OK) {
      spdlog::error("Couldn't decode COBS {}", int(result.status));
      return;
    } else {
      valid_len = result.out_len;
      spdlog::trace("COBS output: {}", valid_len);
      cobs_buffer.erase(0, cobs_len);
    }
    auto tag_location = std::find(buffer.begin(), buffer.begin()+valid_len, 0u);
    if (tag_location == buffer.begin()+valid_len) {
      spdlog::error("Couldn't find topic tag in message");
      return;
    }
    const int tag_size = std::distance(buffer.begin(), tag_location)+1;
    auto msg_view =
      std::string_view(reinterpret_cast<char*>(buffer.data() + tag_size),
                       valid_len - tag_size);
    std::string msg_topic(buffer.begin(), tag_location);
    if (msg_topic == "/model/rover/odometry") {
        spdlog::debug("Got rover odometry message");
        update_odometry(msg_view);
    }
    else if (msg_topic == "/world/default/model/rover/link/base_link/sensor/imu_sensor/imu") {
      spdlog::debug("Got base-link IMU message");
      update_base_imu(msg_view);
    }
    else if (msg_topic == "/depth_camera/points") {
      spdlog::info("Got depth camera pointcloud message");
      update_pointcloud(msg_view);
    }
    else {
      spdlog::info("Got message from unknown topic: {}", msg_topic);
    }
  }
  public:
    GazeboInterface()
      : m_subscribed_topics{ "/clock", "/stats" }
    {
    }
    auto loop() -> asio::awaitable<void> {
      m_gz_node.Subscribe(
        "/world/default/model/rover/link/base_link/sensor/imu_sensor/imu",
        &GazeboInterface::update_base_imu, this);
      m_gz_node.Subscribe(
        "/model/rover/odometry", &GazeboInterface::update_odometry, this);
      m_gz_node.Subscribe(
        "/depth_camera/points", &GazeboInterface::update_pointcloud, this);
      m_gz_node.Subscribe(
        "/world/default/model/rover/link/base_link/sensor/magnet/magnetometer",
        &GazeboInterface::update_magnetic_field,
        this);

      m_cmd_vel_pub = m_gz_node.Advertise<gz::msgs::Twist>("/model/rover/cmd_vel");
      co_return;
    }
    auto imu_linear_acceleration() -> Vector3d const {
      return m_gz_state.m_imu_linear_acceleration;
    }
    auto imu_angular_velocity() -> Vector3d const {
      return m_gz_state.m_imu_angular_velocity;
    }
    auto odometry_position() -> Vector3d const {
      return m_gz_state.m_odometry_position;
    }
    auto odometry_velocity_heading() -> Vector4d const {
      return m_gz_state.m_odometry_velocity_heading;
    }
    auto magnetic_field() -> Vector3d const {
      return m_gz_state.m_magnetic_field;
    }
    auto depth_camera_pointcloud() -> gz::msgs::PointCloudPacked const {
      return m_gz_state.m_packed_pointcloud;
    }
    auto set_target_velocity(Vector3d linear, Vector3d angular) -> void {
      auto l = m_gz_cmd.m_cmd_vel.release_linear();
      auto a = m_gz_cmd.m_cmd_vel.release_angular();

      m_gz_cmd.m_cmd_vel_linear.set_x(linear(0));
      m_gz_cmd.m_cmd_vel_linear.set_y(linear(1));
      m_gz_cmd.m_cmd_vel_linear.set_z(linear(2));
      m_gz_cmd.m_cmd_vel_angular.set_x(angular(0));
      m_gz_cmd.m_cmd_vel_angular.set_y(angular(1));
      m_gz_cmd.m_cmd_vel_angular.set_z(angular(2));

      m_gz_cmd.m_cmd_vel.set_allocated_linear(&m_gz_cmd.m_cmd_vel_linear);
      m_gz_cmd.m_cmd_vel.set_allocated_angular(&m_gz_cmd.m_cmd_vel_angular);

      m_cmd_vel_pub.Publish(m_gz_cmd.m_cmd_vel);
    }
};
