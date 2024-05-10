#include "gazebo_interface.hpp"
#include <asio/detached.hpp>
#include <exception>
#include <gtest/gtest.h>
#include <thread>

using namespace std::literals::chrono_literals;
TEST(GazeboInterfaceTest, ReceiveClock) {
  asio::io_context io_ctx(2);
  GazeboInterface gi;
  gi.set_target_velocity({1,0,0}, {0,0,0});
  asio::co_spawn(io_ctx, gi.loop(), [](std::exception_ptr p) {
    if (p) {
      try {
        std::rethrow_exception(p);
      } catch (const std::exception& e) {
        FAIL() << "receive_message_loop coroutine threw exception: " << e.what()
               << "\n";
      }
    }
  });
  io_ctx.run_for(5s);
}
