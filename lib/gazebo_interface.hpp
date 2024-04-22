#pragma once

#include "common.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

#include <gz/msgs.hh>
#include <Eigen/Geometry>

using asio::ip::tcp;
using Eigen::Vector4f;
using Eigen::Vector3f;

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
    return "AutopilotCommunication";
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
  Vector3f m_odometry_position;
  Vector3f m_imu_linear_acceleration;
  Vector3f m_imu_angular_velocity;
};

class GazeboInterface {
  tcp::socket m_proxy;
  GazeboState m_gz_state;
  std::vector<std::string> m_subscribed_topics;

  auto update_base_imu(std::string_view msg_view) -> void {
    gz::msgs::IMU imu;
    if (not imu.ParseFromString(msg_view)) {
      spdlog::error("Tag len: {}. Couldn't parse gz::msgs::IMU from received proxy message.", msg_view.size());
      return;
    }
    m_gz_state.m_imu_linear_acceleration << imu.linear_acceleration().x(), imu.linear_acceleration().y(), imu.linear_acceleration().z();
    m_gz_state.m_imu_angular_velocity << imu.angular_velocity().x(), imu.angular_velocity().y(), imu.angular_velocity().z();
    spdlog::debug("Got imu: {}, {}", m_gz_state.m_imu_linear_acceleration,m_gz_state.m_imu_angular_velocity);
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
  auto handle_message(std::vector<char>& buffer, int valid_len) -> void {
    auto tag_location = std::find(buffer.begin(), buffer.end(), 0u);
    if (tag_location == buffer.end()) {
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
    else {
      spdlog::info("Got message from unknown topic: {}", msg_topic);
    }
  }
  auto receive_message() -> asio::awaitable<std::error_code> {
    std::vector<char> buffer(1024);
    auto [error, len] = co_await m_proxy.async_read_some(asio::buffer(buffer), use_nothrow_awaitable);
    spdlog::trace("Read {} from proxy socket", len);
    if (error) {
      spdlog::trace("Read from m_proxy failed, asio error: {}", error.message());
      co_return error;
    }
    handle_message(buffer, len);
    co_return GazeboErrc::Success;
  }
  public:
    GazeboInterface(tcp::socket&& proxy_socket)
      : m_proxy{ std::move(proxy_socket) }
      , m_subscribed_topics{ "/clock", "/stats" }
    {
    }
    auto loop() -> asio::awaitable<void> {
      while (true) {
        // Going to have to use a channel here to buffer sends one day
        auto error = co_await receive_message();
        if (error) {
          spdlog::error("Couldn't receive_message: {}: {}",
                        error.category().name(),
                        error.message());
        }
      }
    }
    auto imu_linear_acceleration() -> Vector3f const {
      return m_gz_state.m_imu_linear_acceleration;
    }
    auto imu_angular_velocity() -> Vector3f const {
      return m_gz_state.m_imu_angular_velocity;
    }
    auto odometry_position() -> Vector3f const {
      return m_gz_state.m_odometry_position;
    }
};
