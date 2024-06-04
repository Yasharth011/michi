#pragma once

#include "common.hpp"
#include "expected.hpp"
#include <asio/async_result.hpp>
#include <asio/io_context.hpp>
#include <asio/steady_timer.hpp>
#include <chrono>
#include <librealsense2/h/rs_sensor.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <librealsense2/hpp/rs_frame.hpp>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <librealsense2/hpp/rs_processing.hpp>
#include <type_traits>
#include <system_error>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

using namespace std::literals::chrono_literals;
template <typename T>
using tResult = tl::expected<T, std::error_code>;
using tl::make_unexpected;

enum class DeviceErrc {
  // 0 imples success
  NoDeviceConnected = 10, // Setup error
  LibrsError = 20, // librealsense gave back an error
};
struct DeviceErrCategory : std::error_category {
  const char* name() const noexcept override {
    return "DeviceIO";
  }
  std::string message(int ev) const override {
    switch (static_cast<DeviceErrc>(ev)) {
      case DeviceErrc::NoDeviceConnected:
      return "no device connected: cannot acquire data";
      case DeviceErrc::LibrsError:
      return "failure in librealsense";
      default:
      return "(unrecognized error)";
    }
  }
};
const DeviceErrCategory deverrc_category;
std::error_code make_error_code(DeviceErrc e) {
  return {static_cast<int>(e), deverrc_category};
}
namespace std {
  template <>
  struct is_error_code_enum<DeviceErrc> : true_type {};
}
auto setup_device(bool enable_imu = false) noexcept -> tResult<std::tuple<rs2::pipeline, float, float>> {
  try{
    rs2::pipeline pipe;
    rs2::config stream_config;
    rs2::context ctx;
    float fov[2];

    auto devices = ctx.query_devices();
    if (devices.size() == 0) return make_unexpected(DeviceErrc::NoDeviceConnected);
    stream_config.enable_stream(rs2_stream::RS2_STREAM_COLOR, 0, 640, 480, rs2_format::RS2_FORMAT_BGR8, 30); // Choose resolution here
    stream_config.enable_stream(rs2_stream::RS2_STREAM_DEPTH, 0, 640, 480, rs2_format::RS2_FORMAT_Z16, 30);
    if (enable_imu) {
      stream_config.enable_stream(rs2_stream::RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
      stream_config.enable_stream(rs2_stream::RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    }
    rs2::pipeline_profile selection = pipe.start(stream_config);
    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    spdlog::info("Depth stream {}x{}", depth_stream.width(), depth_stream.height());
    auto i = depth_stream.get_intrinsics();
    rs2_fov(&i, fov);
    fov[0] = (fov[0] * M_PI)/180.0f;
    fov[1] = (fov[1] * M_PI)/180.0f;
    return std::tie(pipe, fov[0], fov[1]);
  }
  catch (const std::exception& e) {
    spdlog::error("Exception in setup_device(): {}", e.what());
    return make_unexpected(DeviceErrc::LibrsError);
  }
}

class RealsenseDevice {
  // TODO: remove io_ctx
  auto async_update(std::chrono::microseconds wait_time)
    -> asio::awaitable<void>
  {
    asio::steady_timer timer(m_io_ctx);
    while (not pipe.poll_for_frames(&frames)) {
      timer.expires_after(wait_time);
      co_await timer.async_wait(use_nothrow_awaitable);
      spdlog::debug("Timer expired");
    }
  }

  public:
  RealsenseDevice(rs2::pipeline& pipe, asio::io_context& io_ctx) : pipe{pipe}, m_io_ctx(io_ctx) {}
  auto async_get_rgb_frame() -> asio::awaitable<rs2::frame> {
    rs2::frame rgb_frame = frames.first_or_default(RS2_STREAM_COLOR);
    do {
      co_await async_update(34ms);
      rgb_frame = frames.first_or_default(RS2_STREAM_COLOR);
    } while (not rgb_frame);
    spdlog::debug("rgb_frame size {}", rgb_frame.get_data_size());
    co_return rgb_frame;
  }
  auto async_get_depth_frame() -> asio::awaitable<rs2::depth_frame> {
    rs2::depth_frame depth = frames.get_depth_frame();
    do {
      co_await async_update(34ms);
      depth = frames.get_depth_frame();
    } while (not depth);
    // Decimation > Spatial > Temporal > Threshold
    depth = temp_filter.process(depth);
    spdlog::debug("Depth Frame# {}", depth.get_frame_number());
    co_return depth;
  }
  auto async_get_points() -> asio::awaitable<rs2::points>{
    rs2::frame depth = co_await async_get_depth_frame();
    co_return pc.calculate(depth);
  }
  auto async_get_imu() -> asio::awaitable<Eigen::Matrix<float, 3, 2>> {
    rs2::frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
    rs2::frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO);
    do {
      co_await async_update(10ms);
      if (not accel_frame) accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
      if (not gyro_frame) gyro_frame = frames.first_or_default(RS2_STREAM_GYRO);
    } while (not (accel_frame and gyro_frame));
    rs2_vector accel_data = accel_frame.as<rs2::motion_frame>().get_motion_data();
    rs2_vector gyro_data = gyro_frame.as<rs2::motion_frame>().get_motion_data();
    co_return Eigen::Matrix<float, 3, 2>{
      { accel_data.x, gyro_data.x},
      { accel_data.y, gyro_data.y},
      { accel_data.z, gyro_data.z},
    };
  }

  private:
  rs2::pipeline pipe;
  asio::io_context& m_io_ctx;
  rs2::frameset frames;

  rs2::temporal_filter temp_filter;
  rs2::pointcloud pc;
};
