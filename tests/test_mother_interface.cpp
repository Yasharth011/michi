#include <asio/detached.hpp>
#include <exception>
#include <gtest/gtest.h>
#include <spdlog/fmt/ranges.h>
#include <thread>
#include "mother_interface.hpp"

using namespace std::literals::chrono_literals;
TEST(CobsTest, DecodeMotherMsg)
{
  uint8_t encoded_msg[] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x80,
    0x3f, 0x01, 0x01, 0x02, 0x3f, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x05, 0xed, 0x04, 0x51, 0x4e, 0x00
  };
  size_t encoded_size = 64 + 2;
  mother::mother_msg msg;
  spdlog::info("Size of mother_msg: {}", sizeof(msg));
  if (auto result = cobs_decode(reinterpret_cast<void*>(&msg),
                                sizeof(msg),
                                encoded_msg,
                                encoded_size - 1);
      result.status != COBS_DECODE_OK) {
    FAIL() << "COBS decode failed: " << int(result.status) << "\n";
  }
  uint8_t decoded_msg[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0xed, 0x04, 0x51, 0x4e };
  EXPECT_TRUE(std::equal(
    decoded_msg, decoded_msg + 64, reinterpret_cast<uint8_t*>(&msg)));
}

TEST(CobsTest, EncodeMotherMsg)
{
  mother::DiffDriveTwist twist = { .linear_x = 1.0f, .angular_z = 0.5f };
  mother::mother_cmd_msg cmd = { .drive_cmd = twist };
  mother::mother_msg msg = { .type = mother::T_MOTHER_CMD_DRIVE,
                             .cmd = cmd,
                             .crc = 1313932525 };
  uint8_t buffer[66];
  spdlog::info("Size of mother_msg: {}", sizeof(msg));
  if (auto result = cobs_encode(buffer,
                                mother::MOTHER_MAX_MSG_LEN,
                                reinterpret_cast<const void*>(&msg),
                                sizeof(mother::mother_msg));
      result.status != COBS_ENCODE_OK) {
    FAIL() << "COBS decode failed: " << int(result.status) << "\n";
  } else
    spdlog::info("Size of encoded msg: {}", result.out_len);
  spdlog::info("{::#x}", buffer);
  uint8_t encoded_msg[] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x80,
    0x3f, 0x01, 0x01, 0x02, 0x3f, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x05, 0xed, 0x04, 0x51, 0x4e, 0x00
  };
  EXPECT_TRUE(std::equal(encoded_msg, encoded_msg + 65, buffer));
}
TEST(MotherInterfaceTest, ReadWriteTest) {
  asio::io_context io_ctx(2);
asio::serial_port dev_serial(io_ctx, "/dev/ttyUSB0");
dev_serial.set_option(asio::serial_port_base::baud_rate(921600));  MotherInterface mi(std::move(dev_serial));
  asio::co_spawn(io_ctx, mi.loop(), [](std::exception_ptr p) {
    if (p) {
      try { std::rethrow_exception(p); }
      catch(const std::exception& e) {
        FAIL() << "receive_message_loop coroutine threw exception: " << e.what() << "\n";
      }
    }
  });
  asio::co_spawn(io_ctx, mi.ccm(), [](std::exception_ptr p) {
    if (p) {
      try { std::rethrow_exception(p); }
      catch(const std::exception& e) {
        FAIL() << "Set target velocity coroutine threw exception: " << e.what() << '\n';
      }
    }
  });
  io_ctx.run();
}
