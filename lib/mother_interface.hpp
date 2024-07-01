#pragma once

#include <Eigen/Geometry>
#include "common.hpp"
#include <asio/error_code.hpp>
#include <asio/steady_timer.hpp>
#include <chrono>
#include <algorithm>
#include <asio/serial_port.hpp>
#include <asio/this_coro.hpp>
#include <asio/experimental/channel.hpp>
#include <asio/write.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <cobs.h>
#include "crc.h"

using namespace std::literals::chrono_literals;
using namespace std::chrono;
using Eigen::Vector4d;
using Eigen::Vector3d;

enum class MotherErrc
{
  Success = 0,
  NoHeartbeat = 1, // System Failure
  NoCommandAck,
  FailedWrite,
  IncompleteRead,
  FailedRead,
  CobsDecodeError,
  UnknownMessageType,
  CrcCheckFailed,
  TransmitTimeout = 10, // Timeouts
  ReceiveTimeout,
};
struct MotherErrCategory : std::error_category
{
  const char* name() const noexcept override
  {
    return "MotherCommunication";
  }
  std::string message(int ev) const override
  {
    switch (static_cast<MotherErrc>(ev)) {
      case MotherErrc::NoHeartbeat:
        return "no heartbeat received from motherboard";
      case MotherErrc::NoCommandAck:
        return "no ack received after command";
      case MotherErrc::FailedWrite:
        return "could not write, asio error";
      case MotherErrc::IncompleteRead:
        return "message missing terminator null-byte";
      case MotherErrc::CobsDecodeError:
        return "cobs decode failed";
      case MotherErrc::CrcCheckFailed:
        return "crc check failed";
      case MotherErrc::UnknownMessageType:
        return "unknown message type";
      case MotherErrc::FailedRead:
        return "could not read, asio error";
      case MotherErrc::ReceiveTimeout:
        return "did not get response, timed out";
      case MotherErrc::TransmitTimeout:
        return "could not send message, timed out";
      default:
        return "(unrecognized error)";
    }
  }
};
const MotherErrCategory Mothererrc_category;
std::error_code
make_error_code(MotherErrc e)
{
  return { static_cast<int>(e), Mothererrc_category };
}
namespace std {
template<>
struct is_error_code_enum<MotherErrc> : true_type
{};
}

// From kyvernitis.h
namespace mother {
enum MotherMsgType {
	T_MOTHER_CMD_DRIVE,
	T_MOTHER_CMD_ARM,
	T_MOTHER_CMD_LA,
	T_MOTHER_STATUS,
	T_MOTHER_ERROR,
	T_MOTHER_INFO
};
struct DiffDriveStatus
{
  float x, y, heading, linear, angular;
};
struct DiffDriveTwist
{
  float linear_x;
  float angular_z;
};
struct mother_cmd_msg
{
  struct DiffDriveTwist drive_cmd;
  float arm_joint[3];
  uint8_t adaptive_sus_cmd[4];
};
struct mother_status_msg
{
  struct DiffDriveStatus odom;
  float arm_joint_status[3];
  uint64_t timestamp;
};
struct mother_msg
{
  uint16_t type;
  struct mother_status_msg status;
  struct mother_cmd_msg cmd;
  char info[100];
  uint32_t crc;
};
constexpr size_t MOTHER_MAX_MSG_LEN = sizeof(struct mother::mother_msg) + 2;
}

struct MotherState {
  mother::DiffDriveStatus m_odometry;
  float arm_joint[3];
  uint8_t adaptive_sus_cmd[4];
};

constexpr size_t REQUESTS_QUEUE_SIZE = 25;
class MotherInterface {
  asio::serial_port m_uart;
  time_point<steady_clock> m_start;
  MotherState m_state;

  asio::experimental::channel<void(asio::error_code, mother::mother_msg)> m_requests;

  inline auto get_uptime() -> uint32_t {
    return duration_cast<milliseconds>(steady_clock::now() - m_start).count();
  }
  inline size_t cobs_buffer_size(size_t input_size,
                                 bool with_trailing_zero = false)
  {
    size_t output_size = input_size + input_size / 254 + 1;
    if (with_trailing_zero)
      output_size++;
    return output_size;
  }
  auto send_message(const mother::mother_msg& msg)
    -> asio::awaitable<std::tuple<asio::error_code, std::size_t>>
  {
    uint8_t buffer[mother::MOTHER_MAX_MSG_LEN];
    if (auto result = cobs_encode(reinterpret_cast<void*>(buffer),
                                  mother::MOTHER_MAX_MSG_LEN,
                                  reinterpret_cast<const void*>(&msg),
                                  sizeof(mother::mother_msg));
        result.status != COBS_ENCODE_OK) {
      spdlog::error("Mother: COBS encode failed: {}", int(result.status));
    }
    buffer[mother::MOTHER_MAX_MSG_LEN-1] = 0x00;
    // spdlog::info("MSG: {::#x}", buffer);
    auto res = co_await asio::async_write(
      m_uart,
      asio::buffer(buffer, mother::MOTHER_MAX_MSG_LEN),
      use_nothrow_awaitable);
    co_return res;
  }
auto receive_message() -> asio::awaitable<MotherErrc> {
  std::vector<uint8_t> buffer(mother::MOTHER_MAX_MSG_LEN);
  auto [error, len] = co_await asio::async_read(m_uart, asio::buffer(buffer, buffer.size()), use_nothrow_awaitable);
  if (error) {
    spdlog::trace("Read from m_uart failed, asio error: {}", error.message());
    co_return MotherErrc::FailedRead;
  }
  if (len != mother::MOTHER_MAX_MSG_LEN) {
    spdlog::error("Couldn't read a complete mother message. Len: {}", len);
    co_return MotherErrc::IncompleteRead;
  }
  mother::mother_msg msg;
  if (auto result = cobs_decode(
        reinterpret_cast<void*>(&msg), sizeof(msg), buffer.data(), len - 1);
      result.status != COBS_DECODE_OK) {
    spdlog::error("COBS decode failed: {}", int(result.status));
    co_return MotherErrc::CobsDecodeError;
  }

  if (msg.type > 6 or msg.type < 0)
    co_return MotherErrc::UnknownMessageType;
  if (msg.crc != crc32_ieee(reinterpret_cast<const uint8_t*>(&msg),
                            sizeof(mother::mother_msg) - sizeof(uint32_t)))
    co_return MotherErrc::CrcCheckFailed;

  // Update state
  if (msg.type == mother::MotherMsgType::T_MOTHER_INFO) {
    spdlog::warn("MOTHER: {}", msg.info);
    co_return MotherErrc::Success;
  }
  if (msg.type == mother::MotherMsgType::T_MOTHER_ERROR) {
    spdlog::error("MOTHER: {}", msg.info);
    co_return MotherErrc::Success;
  }
  m_state.m_odometry = msg.status.odom;
  const int arm_joint_status_len = 3;
  std::copy(msg.status.arm_joint_status,
            msg.status.arm_joint_status + arm_joint_status_len,
            m_state.arm_joint);
  spdlog::debug("[{}] Got mother msg: type {}, position: {:02.5f}×{:02.5f} "
               "heading {:02.2f}°, arm joints: {::02.2f}",
               msg.status.timestamp,
               msg.type,
               m_state.m_odometry.x,
               m_state.m_odometry.y,
               m_state.m_odometry.heading * 180/M_PI, m_state.arm_joint);
  co_return MotherErrc::Success;
}
public:
  MotherInterface(asio::serial_port&& sp)
    : m_uart{ std::move(sp) }
    , m_start{ steady_clock::now() }
    , m_requests(m_uart.get_executor(), REQUESTS_QUEUE_SIZE)
  {
    // m_uart.set_option(asio::serial_port_base::baud_rate(115200));
  }
  auto loop() -> asio::awaitable<void>
  {
    asio::steady_timer timer(m_uart.get_executor());

    mother::mother_msg hb = heartbeat();
    auto [error, written] = co_await send_message(hb);
    auto last_heartbeat = steady_clock::now();
    if (error) {
      spdlog::trace("Couldn't send first heartbeat, asio error: {}", error.message());
      co_return;
    }
    while (true) {
      timer.expires_after(30ms);
      auto result = co_await (m_requests.async_receive(use_nothrow_awaitable) ||
                              receive_message() || timer.async_wait(use_nothrow_awaitable));
      if (std::holds_alternative<
            std::tuple<asio::error_code, mother::mother_msg>>(result)) {
        mother::mother_msg this_msg;
        tie(error, this_msg) = std::get<std::tuple<asio::error_code, mother::mother_msg>>(result);
        if (error) {
          spdlog::trace("Couldn't send msg, asio error: {}",  error.message());   
          continue;
        }
        tie(error, written) = co_await send_message(this_msg);
        if (error) {
          spdlog::trace("Couldn't send msg, asio error: {}", error.message());   
          continue;
        }
        spdlog::debug("Sent message!");
      } else if (std::holds_alternative<MotherErrc>(result)) {
        auto error = std::get<MotherErrc>(result);
        if (error != MotherErrc::Success) {
          spdlog::error("Couldn't receive_message: {}: {}",
                        Mothererrc_category.name(),
                        Mothererrc_category.message(static_cast<int>(error)));
        }
      }
      else {
        hb = heartbeat();
        auto [hb_error, hb_written] = co_await send_message(hb);
        last_heartbeat = steady_clock::now();
        if (error) {
          spdlog::trace("Couldn't send first heartbeat, asio error: {}",
                        hb_error.message());
          co_return;
        }
      }
    }
  }

  auto heartbeat() -> mother::mother_msg {
    mother::DiffDriveTwist twist = {.linear_x = 0, .angular_z = 0};
    mother::mother_cmd_msg cmd = {.drive_cmd = twist};
    mother::mother_msg msg = { .type = 17, .cmd = cmd, .crc = 0};
    msg.crc = crc32_ieee(reinterpret_cast<const uint8_t*>(&msg), sizeof(msg) - sizeof(uint32_t));
    return msg;
  }
  auto set_target_velocity(double linear_x, double angular_z) -> asio::awaitable<void> {
    mother::DiffDriveTwist twist = {.linear_x = float(linear_x), .angular_z = float(angular_z )};
    mother::mother_cmd_msg cmd = {.drive_cmd = twist};
    mother::mother_msg msg = { .type = mother::T_MOTHER_CMD_DRIVE, .cmd = cmd, .crc = 0};
    msg.crc = crc32_ieee(reinterpret_cast<const uint8_t*>(&msg), sizeof(msg) - sizeof(uint32_t));
    spdlog::trace("CRC: {}", msg.crc);

    auto [error] = co_await m_requests.async_send(asio::error_code{}, msg, use_nothrow_awaitable);
    if (error) {
      spdlog::error("Could not send set_target_velocity, asio error: {}\n", error.message());
    }
  }
  auto odometry_position() -> Vector3d const
  {
    return Vector3d{ m_state.m_odometry.x,
                     m_state.m_odometry.y,
                     m_state.m_odometry.heading };
  }
  auto odometry_velocity_heading() -> Vector4d const
  {
    return Vector4d{ m_state.m_odometry.linear * cos(m_state.m_odometry.heading),
                     m_state.m_odometry.linear * sin(m_state.m_odometry.heading),
                     m_state.m_odometry.angular,
                     m_state.m_odometry.heading };
  }
  auto joint_position() -> Vector3d const
  {
    return Eigen::Vector3f(m_state.arm_joint).cast<double>();
  }
  auto magnetic_field() -> Vector3d const
  {
    // FIXME: Parse magnetic field and return that here
    return Eigen::Vector3f(m_state.arm_joint).cast<double>();
  }
  auto ccm() -> asio::awaitable<void> {
    asio::steady_timer timer(co_await asio::this_coro::executor);
    while (true) {
      co_await set_target_velocity(1.0, 0.0);
      timer.expires_after(40ms);
      co_await timer.async_wait(use_nothrow_awaitable);
    }
  }
};
