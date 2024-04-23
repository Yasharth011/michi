// In Honour of the boundless Rudra
// Shashwat Ganesh, R24

#include <argparse/argparse.hpp>
#include <asio/detached.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "common.hpp"
#include "gazebo_interface.hpp"
#include "realsense_generator.hpp"
#include "ekf.hpp"

using fmt::print;
static argparse::ArgumentParser args("TubePlanner");

template<typename CameraIf, typename ImuIf, typename OdomIf>
auto
mission(std::shared_ptr<CameraIf> ci, std::shared_ptr<ImuIf> imu_if, std::shared_ptr<OdomIf> odom_if) -> asio::awaitable<void>
{
  auto this_exec = co_await asio::this_coro::executor;
  spdlog::info("IMU Linear acceleration: {}", imu_if->imu_linear_acceleration());

  Matrix<float, 2, 1> u;
  u = control_input((imu_if->imu_linear_acceleration(), imu_if->imu_angular_velocity(), odom_if->imu_odometry_position());
  Matrix<float, 4, 1> xEst;
  Matrix<float, 4, 4> PEst;
  std::tie(xEst, PEst) = run_ekf(u);

  co_return;
}
int main(int argc, char* argv[]) {
  args.add_argument("model_path").help("Path to object classification model");
  args.add_argument("--sim").default_value(true).implicit_value(true).help("Run in simulation mode");
  args.add_argument("-p", "--port").default_value(std::string("6000")).help("Simulation port to talk to gz_proxy");

  int log_verbosity = 0;
  args.add_argument("-V", "--verbose")
  .action([&](const auto &) {++log_verbosity;})
  .append()
  .default_value(false)
  .implicit_value(true)
  .nargs(0);
  
  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << '\n';
    std::cerr << args;
    return 1;
  }

  switch (log_verbosity) {
    case 0:
    spdlog::set_level(spdlog::level::info);
    break;
    case 1:
    spdlog::info("Verbosity 1: Logging debug messages");
    spdlog::set_level(spdlog::level::debug);
    break;
    default:
    spdlog::info("Verbosity 2: Logging trace messages");
    spdlog::set_level(spdlog::level::trace);
  }

  asio::io_context io_ctx;
  std::optional<GazeboInterface> wrapped_gazebo_interface;
  if (args.get<bool>("--sim")) {
    tcp::socket proxy(io_ctx);
    proxy.connect(*tcp::resolver(io_ctx).resolve("0.0.0.0", args.get<std::string>("-p"), tcp::resolver::passive));
    wrapped_gazebo_interface.emplace(GazeboInterface(std::move(proxy)));
  }
  auto gi = std::make_shared<GazeboInterface>(std::move(*wrapped_gazebo_interface));

  asio::co_spawn(io_ctx, mission<GazeboInterface, GazeboInterface, GazeboInterface>(gi, gi, gi), [](std::exception_ptr p) {
    if (p) {
      try {
        std::rethrow_exception(p);
      } catch (const std::exception& e) {
        spdlog::error("Mission coroutine threw exception: {}", e.what());
      }
    }
  });
  asio::co_spawn(io_ctx, gi->loop(), [](std::exception_ptr p) {
    if (p) {
      try {
        std::rethrow_exception(p);
      } catch (const std::exception& e) {
        spdlog::error("GazeboInterface loop coroutine threw exception: {}",
                      e.what());
      }
    }
  });
  io_ctx.run();
}
