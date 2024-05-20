#pragma once

#include <Eigen/Geometry>
#include "common.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <asio/error_code.hpp>
#include <chrono>
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
using Eigen::Vector4f;
using Eigen::Vector3f;

enum class MotherErrc
{
  Success = 0,
  NoHeartbeat = 1, // System Failure
  NoCommandAck,
  FailedWrite,
  FailedRead,
  TransmitTimeout = 10, // Timeouts
  ReceiveTimeout,
};
struct MotherErrCategory : std::error_category
{
  const char* name() const noexcept override
  {
    return "AutopilotCommunication";
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
enum MotherMsgType
{
  T_MOTHER_CMD_DRIVE,
  T_MOTHER_CMD_ARM,
  T_MOTHER_CMD_LA,
  T_MOTHER_STATUS
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
  uint8_t arm_joint[5];
  uint8_t adaptive_sus_cmd[4];
};
struct mother_status_msg
{
  struct DiffDriveStatus odom;
  uint64_t timestamp;
};
struct mother_msg
{
  uint16_t type;
  struct mother_status_msg status;
  struct mother_cmd_msg cmd;
  uint32_t crc;
};
constexpr size_t MOTHER_MAX_MSG_LEN = sizeof(struct mother::mother_msg) + 2;
}

struct MotherState {
  mother::DiffDriveStatus m_odometry;
  uint8_t arm_joint[5];
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
    auto res = co_await asio::async_write(
      m_uart,
      asio::buffer(buffer, mother::MOTHER_MAX_MSG_LEN),
      use_nothrow_awaitable);
    co_return res;
  }
auto receive_message() -> asio::awaitable<std::error_code> {
  std::vector<uint8_t> buffer(mother::MOTHER_MAX_MSG_LEN);
  auto [error, len] = co_await m_uart.async_read_some(asio::buffer(buffer), use_nothrow_awaitable);
  if (error) {
    spdlog::trace("Read from m_uart failed, asio error: {}", error.message());
    co_return error;
  }
  if (len != mother::MOTHER_MAX_MSG_LEN) {
    spdlog::trace("Couldn't read a complete mother message");
    co_return MotherErrc::FailedRead;
  }
  mother::mother_msg msg;
  if (auto result = cobs_decode(
        reinterpret_cast<void*>(&msg), sizeof(msg), buffer.data(), len - 1);
      result.status != COBS_DECODE_OK) {
    spdlog::error("COBS decode failed: {}", int(result.status));
    co_return MotherErrc::FailedRead;
  }

  if (msg.type != mother::T_MOTHER_STATUS)
    co_return MotherErrc::FailedRead;
  if (msg.crc == crc32_ieee(reinterpret_cast<const uint8_t*>(&msg),
                            sizeof(mother::mother_msg) - sizeof(uint32_t)))
    co_return MotherErrc::FailedRead;

  m_state.m_odometry = msg.status.odom;
  spdlog::info("Got mother msg: type {}, position: {}×{}", msg.type, m_state.m_odometry.x, m_state.m_odometry.y);
  // FIXME: implement parsing other fields when implemented upstream
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

    std::error_code error;
    size_t written;
    // mavlink_message_t hb_msg = heartbeat();
    // auto [error, written] = co_await send_message(hb_msg);
    // if (error) {
    //   spdlog::trace("Couldn't send first heartbeat, asio error: {}", error.message());
    //   co_return make_unexpected(MavlinkErrc::FailedWrite);
    // }
    auto last_heartbeat = steady_clock::now();
    while (true) {
      // if (steady_clock::now() > last_heartbeat + 1s) {
      //   hb_msg = heartbeat();
      //   tie(error, written) = co_await send_message(hb_msg);
      //   if (error) {
      //     spdlog::trace("Couldn't send heartbeat, asio error: {}", error.message());   
      //     co_return make_unexpected(MavlinkErrc::FailedWrite);
      //   }
      // }
      auto result = co_await (m_requests.async_receive(use_nothrow_awaitable) || receive_message());
      if (std::holds_alternative<std::tuple<asio::error_code, mother::mother_msg>>(result)) {
        mother::mother_msg this_msg;
        tie(error, this_msg) = std::get<std::tuple<asio::error_code, mother::mother_msg>>(result);
        if (error) {
          spdlog::trace("Couldn't send msg, asio error: {}",  error.message());   
          co_return;
          // co_return make_unexpected(MotherErrc::FailedWrite);
        }
        tie(error, written) = co_await send_message(this_msg);
        if (error) {
          spdlog::trace("Couldn't send msg, asio error: {}", error.message());   
          co_return;
          // co_return make_unexpected(MotherErrc::FailedWrite);
        }
        spdlog::info("Sent message!");
      } else if (std::holds_alternative<std::error_code>(result)) {
      auto error = std::get<std::error_code>(result);
       if (error) {
          spdlog::trace("Couldn't receive_message: {}: {}", error.category().name(), error.message());
        }
      }
      // timer.expires_after(20ms);
      // co_await timer.async_wait(use_nothrow_awaitable);
    }
  }

  auto set_target_velocity(float linear_x, float angular_z) -> asio::awaitable<void> {
    mother::DiffDriveTwist twist = {.linear_x = linear_x, .angular_z = angular_z };
    mother::mother_cmd_msg cmd = {.drive_cmd = twist};
    mother::mother_msg msg = { .type = mother::T_MOTHER_CMD_DRIVE, .cmd = cmd, .crc = 0};
    msg.crc = crc32_ieee(reinterpret_cast<const uint8_t*>(&msg), sizeof(msg) - sizeof(uint32_t));
    spdlog::info("CRC: {}", msg.crc);

    auto [error] = co_await m_requests.async_send(asio::error_code{}, msg, use_nothrow_awaitable);
    if (error) {
      spdlog::error("Could not send set_target_velocity, asio error: {}\n", error.message());
    }
  }
  auto odometry_position() -> Vector3f const {
    return Vector3f{m_state.m_odometry.x, m_state.m_odometry.y, m_state.m_odometry.heading};
  }
};
