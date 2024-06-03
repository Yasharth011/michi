// In Honour of the boundless Rudra
// Shashwat Ganesh, R24

#include <argparse/argparse.hpp>
#include <asio/detached.hpp>
#include <asio/this_coro.hpp>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <execution>
#include "common.hpp"
#include "gazebo_interface.hpp"
#include "mother_interface.hpp"
#include "realsense_generator.hpp"
#include "ekf.hpp"

using fmt::print;
using tPointcloud = pcl::PointCloud<pcl::PointXYZ>;
using namespace std::literals::chrono_literals;
static argparse::ArgumentParser args("TubePlanner");

class RealsenseImuPolicy {
  Eigen::Matrix<float, 3, 2> m_imu;
  const Eigen::Matrix<float, 3, 3> m_rot;
  int m_reads = 0;
  public:
    RealsenseImuPolicy() : m_rot{{0, 0, 1},
                                 {1, 0, 0},
                                 {0, 1, 0}} {}
      protected:
    using If = RealsenseDevice;
    auto imu_update(std::shared_ptr<If> rs_dev) -> asio::awaitable<void> {
      m_imu = co_await rs_dev->async_get_imu();
      m_imu = m_rot * m_imu;
      m_reads = 2;
    }
    auto imu_linear_acceleration(std::shared_ptr<If> rs_dev) -> asio::awaitable<Eigen::Vector3f> {
      if (not m_reads) {
        co_await imu_update(rs_dev);
      }
      m_reads--;
      co_return m_imu.col(0);
    }
    auto imu_angular_velocity(std::shared_ptr<If> rs_dev) -> asio::awaitable<Eigen::Vector3f> {
      if (not m_reads) {
        co_await imu_update(rs_dev);
      }
      m_reads--;
      co_return m_imu.col(1);
    }
};
class RealsenseDepthCamPolicy {
  pcl::VoxelGrid<pcl::PointXYZ> m_voxel_filter;
  pcl::PassThrough<pcl::PointXYZ> m_passthrough_filter;
  tPointcloud::Ptr points_to_pcl(const rs2::points& points)
  {
      tPointcloud::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

      auto sp = points.get_profile().as<rs2::video_stream_profile>();
      cloud->width = sp.width();
      cloud->height = sp.height();
      cloud->is_dense = false;
      cloud->points.resize(points.size());
      auto ptr = points.get_vertices();
      std::transform(std::execution::par_unseq,
                     points.get_vertices(),
                     points.get_vertices() + points.size(),
                     cloud->points.begin(),
                     [](auto& ptr) {
                       pcl::PointXYZ p;
                       p.x = ptr.z;
                       p.y = -ptr.x;
                       p.z = -ptr.y;
                       return p;
                     });
      return cloud;
  }
  protected:
    using If = RealsenseDevice;
    auto async_get_rgb_frame(std::shared_ptr<If> rs_dev)
      -> asio::awaitable<cv::Mat>
    {
      auto rgb_frame = co_await rs_dev->async_get_rgb_frame();
      cv::Mat image(
        cv::Size(640, 480), CV_8UC3, const_cast<void*>(rgb_frame.get_data()));
      co_return image;
    }
    auto async_get_depth_frame(std::shared_ptr<If> rs_dev)
      -> asio::awaitable<cv::Mat>
    {
      auto depth_frame = co_await rs_dev->async_get_depth_frame();
      cv::Mat image(
        cv::Size(640, 480), CV_8UC3, const_cast<void*>(depth_frame.get_data()));
      co_return image;
    }
    auto async_get_pointcloud(std::shared_ptr<If> rs_dev)
      -> asio::awaitable<tPointcloud::Ptr>
    {
      auto points = co_await rs_dev->async_get_points();
      spdlog::debug("Got points");
      auto pcl_cloud = points_to_pcl(points);

      if (pcl_cloud->size() < 50'000) co_return pcl_cloud;
      spdlog::info("Before filtering: {} points", pcl_cloud->size());

      m_passthrough_filter.setInputCloud(pcl_cloud);
      m_passthrough_filter.setFilterFieldName("x");
      m_passthrough_filter.setFilterLimits(0.0, 10.0);
      m_passthrough_filter.filter(*pcl_cloud);

      m_passthrough_filter.setInputCloud(pcl_cloud);
      m_passthrough_filter.setFilterFieldName("z");
      m_passthrough_filter.setFilterLimits(-1.0, 1.0);
      m_passthrough_filter.filter(*pcl_cloud);

      m_voxel_filter.setInputCloud(pcl_cloud);
      m_voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
      m_voxel_filter.filter(*pcl_cloud);

      spdlog::info("After filtering: {} points", pcl_cloud->size());
      co_return pcl_cloud;
    }
};
class GazeboDepthCamPolicy {
protected:
  using If = GazeboInterface;
  auto async_get_pointcloud(std::shared_ptr<If> gi)
    -> asio::awaitable<tPointcloud::Ptr>
  {
    asio::steady_timer timer(co_await asio::this_coro::executor);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

    gz::msgs::PointCloudPacked packed_msg = gi->depth_camera_pointcloud();
    while (packed_msg.point_step() == 0) {
      timer.expires_after(30ms);
      co_await timer.async_wait(use_nothrow_awaitable);
      packed_msg = gi->depth_camera_pointcloud();
    }

    spdlog::info("{}×{} pointcloud", packed_msg.height(), packed_msg.width());
    int point_step = packed_msg.point_step();
    int num_points = packed_msg.data().size() / point_step;

    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (int i = 0; i < num_points; ++i) {
      const char* point_data = packed_msg.data().c_str() + i * point_step;
      const float* point_fields = reinterpret_cast<const float*>(point_data);

      cloud->points[i].x = point_fields[0];
      cloud->points[i].y = point_fields[1];
      cloud->points[i].z = point_fields[2];
    }
    co_return cloud;
  }
};
class GazeboImuPolicy {
  protected:
    using If = GazeboInterface;
    auto imu_linear_acceleration(std::shared_ptr<If> gi) -> asio::awaitable<Eigen::Vector3f> {
      co_return gi->imu_linear_acceleration();
    }
    auto imu_angular_velocity(std::shared_ptr<If> gi) -> asio::awaitable<Eigen::Vector3f> {
      co_return gi->imu_angular_velocity();
    }
};
class BlankOdomPolicy {
  protected:
    using If = std::nullptr_t;
    auto odometry_position(std::shared_ptr<If> gi) -> Eigen::Vector3f {
      return Eigen::Vector3f(0,0,0);
    }
    auto set_target_velocity(std::shared_ptr<If> gi,
                             Eigen::Vector3f linear,
                             Eigen::Vector3f angular) -> asio::awaitable<void>
    {
      co_return;
    }
};
class GazeboOdomPolicy {
  protected:
    using If = GazeboInterface;
    Eigen::Vector3f m_last_target_velocity;
    auto odometry_position(std::shared_ptr<If> gi) -> Eigen::Vector3f {
      return gi->odometry_position();
    }
    auto odometry_velocity_heading(std::shared_ptr<If> gi) -> Eigen::Vector3f {
      return gi->odometry_velocity_heading();
    }
    auto set_target_velocity(std::shared_ptr<If> gi,
                             Eigen::Vector3f linear,
                             Eigen::Vector3f angular) -> asio::awaitable<void>
    {
      m_last_target_velocity << linear(0), 0, angular(2);
      co_return gi->set_target_velocity(linear, angular);
    }
};
class MotherOdomPolicy {
  protected:
    using If = MotherInterface;
    Eigen::Vector3f m_last_target_velocity;
    auto odometry_position(std::shared_ptr<If> mi) -> Eigen::Vector3f {
      Vector3f xy_blank = mi->odometry_position();
      return Vector3f{xy_blank(0), xy_blank(1), 0};
    }
    auto odometry_velocity_heading(std::shared_ptr<If> mi) -> Eigen::Vector3f {
      return mi->odometry_velocity_heading();
    }
    auto magnetic_field(std::shared_ptr<If> mi) -> Eigen::Vector3f {
      return mi->magnetic_field();
    }
    auto set_target_velocity(std::shared_ptr<If> mi,
                             Eigen::Vector3f linear,
                             Eigen::Vector3f angular) -> asio::awaitable<void>
    {
      m_last_target_velocity << linear(0), 0, angular(2);
      co_await mi->set_target_velocity(linear(0), angular(2));
    }
};
template<typename DepthCamPolicy, typename BaseImuPolicy, typename OdomPolicy>
class AnantaMission
  : public DepthCamPolicy
  , public BaseImuPolicy
  , public OdomPolicy
  , public std::enable_shared_from_this<AnantaMission<DepthCamPolicy, BaseImuPolicy, OdomPolicy>>
{
  using DepthCamPolicy::async_get_pointcloud;
  using BaseImuPolicy::imu_linear_acceleration;
  using BaseImuPolicy::imu_angular_velocity;
  using OdomPolicy::odometry_position;
  using OdomPolicy::set_target_velocity;
  using Mission = AnantaMission<DepthCamPolicy, BaseImuPolicy, OdomPolicy>;

  class AvoidAction {
    public:
      static constexpr float EXTENT_X_METRES = 1;
      static constexpr float EXTENT_Y_METRES = 0.5;
      static constexpr float EXTENT_Z_METRES = 0.03;

      // template <typename A, typename B, typename C>
      static int utility(AnantaMission::Mission* mission) {
        // auto min = mission->m_tree.coordToKey(mission->m_position_est(0),
        //                                       mission->m_position_est(1) -
        //                                         EXTENT_Y_METRES,
        //                                       -(EXTENT_Z_METRES + 0.2));
        auto crater_min = mission->m_tree.coordToKey(0, 0 - EXTENT_Y_METRES, -0.36);
        auto crater_max = mission->m_tree.coordToKey(
          0 + EXTENT_X_METRES, 0 + EXTENT_Y_METRES, -0.33);

        double crater_prob = 0;
        int count = 0;
        for (octomap::OcTree::leaf_bbx_iterator
               it = mission->m_tree.begin_leafs_bbx(crater_min, crater_max),
               end = mission->m_tree.end_leafs_bbx();
             it != end;
             ++it) {
          crater_prob += it->getOccupancy();
          count++;
        }
        if (count != 0) crater_prob /= count;
        else crater_prob = 1;
        spdlog::info("Crater Prob {} : iterated over: {}", 1-crater_prob, count);
        
        auto box_min = mission->m_tree.coordToKey(0.4, 0 - 0.20, -0.33);
        auto box_max = mission->m_tree.coordToKey(
          0 + 0.43, 0 + 0.2, 0);

        count = 0;
        double box_prob = 0;
        for (octomap::OcTree::leaf_bbx_iterator
               it = mission->m_tree.begin_leafs_bbx(box_min, box_max),
               end = mission->m_tree.end_leafs_bbx();
             it != end;
             ++it) {
          box_prob += it->getOccupancy();
          count++;
        }
        box_prob /= count;
        spdlog::info("Box Prob {} : iterated over: {}", box_prob, count);
        return std::max(box_prob * 100, (1 - crater_prob) * 100);
      }

    static auto execute(AnantaMission::Mission* mission) -> asio::awaitable<void> {
      // mission->set_target_velocity(mission->m_odom_if, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.57f});
      co_await mission->set_target_velocity(mission->m_odom_if, {0.0f, 0.0f, 0.0f}, {0.1f, 0.0f, 0.0f});
      asio::steady_timer timer(co_await asio::this_coro::executor);
      timer.expires_after(1s);
      co_await timer.async_wait(use_nothrow_awaitable);
      co_await mission->set_target_velocity(mission->m_odom_if, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
      co_return;
    }
  };
  auto move()
    -> asio::awaitable<void>
  {
    co_await set_target_velocity(m_odom_if, {0.1f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
  }
  auto avoid() -> asio::awaitable<void> {
    co_await set_target_velocity(m_odom_if, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.57f});
    asio::steady_timer timer(co_await asio::this_coro::executor);
    timer.expires_after(1s);
    co_await timer.async_wait(use_nothrow_awaitable);
    co_return;
  }
public:
  octomap::OcTree m_tree;
  octomap::Pointcloud m_map_cloud;
  std::shared_ptr<typename DepthCamPolicy::If> m_ci;
  std::shared_ptr<typename BaseImuPolicy::If> m_imu_if;
  std::shared_ptr<typename OdomPolicy::If> m_odom_if;
  int m_iterations;
  const float m_camera_height;
  Eigen::Matrix<float, 4, 1> m_position_est;
  AnantaMission(std::shared_ptr<typename DepthCamPolicy::If> ci,
                std::shared_ptr<typename BaseImuPolicy::If> imu_if,
                std::shared_ptr<typename OdomPolicy::If> odom_if)
    : m_tree(args.get<float>("-t"))
    , m_iterations(0)
    , m_ci(ci)
    , m_imu_if(imu_if)
    , m_odom_if(odom_if)
    , m_camera_height(args.get<float>("-c"))
    , m_position_est({0,0,0,0})
  {
  }
  auto loop() -> asio::awaitable<void>
  {
    spdlog::info("Camera height: {}m", m_camera_height);
    auto this_exec = co_await asio::this_coro::executor;
    auto localization = EKF();
    asio::steady_timer timer(this_exec);
    double front_prob = 1;

    timer.expires_after(3s);
    co_await timer.async_wait(use_nothrow_awaitable);
    while (true) {
      auto linear_accel = co_await imu_linear_acceleration(m_imu_if);
      auto angular_vel =  co_await imu_angular_velocity(m_imu_if);
      spdlog::info("IMU vel {} gyro {}",
                   linear_accel,
                   angular_vel);
      Eigen::Matrix<float, 2, 1> u =
        localization.control_input(linear_accel,
                                   angular_vel,
                                   odometry_position(m_odom_if));
      spdlog::info("Control input: {}", u);
      Eigen::Matrix<float, 4, 4> position_cov;
      std::tie(m_position_est, position_cov) = localization.run_ekf(u);
      spdlog::info("Position estimate: {}", m_position_est);

      octomap::point3d map_current_pos(
        m_position_est(0), m_position_est(1), m_camera_height);
      auto cam_cloud = co_await async_get_pointcloud(m_ci);
      m_map_cloud.reserve(cam_cloud->points.size());
      int nulls = 0;
      // std::transform(std::execution::par,
      //               cam_cloud->points.begin(),
      //               cam_cloud->points.end(),
      //               m_map_cloud.begin(),
      //               [](auto& point) {
      //                 // if (std::isinf(point.x) or std::isinf(point.y) or
      //                 //     std::isinf(point.z))
      //                 //   return;
      //                 return octomap::point3d(point.x, point.y, point.z);
      //               });
      for (const auto& point : cam_cloud->points) {
        if (std::isinf(point.x) or std::isinf(point.y) or
        std::isinf(point.z)) {
          nulls++;
          continue;
        }
        m_map_cloud.push_back(point.x, point.y, point.z);
      }
      m_tree.insertPointCloud(m_map_cloud, map_current_pos);
      spdlog::info("Got {} nulls. Inserted a cloud!", nulls);
      m_map_cloud.clear();

      auto avo_score = AvoidAction::utility(this);
      spdlog::info("Avo score: {}", avo_score);
      if (avo_score > 40) co_await AvoidAction::execute(this);

      m_iterations++;
      if (m_iterations == 1)
        spdlog::info("Wrote tree: {}", m_tree.write("m_tree.ot"));
      if constexpr (std::is_same<DepthCamPolicy, GazeboDepthCamPolicy>::value) {
        timer.expires_after(10ms);
        co_await timer.async_wait(use_nothrow_awaitable);
      }
    }

    co_return;
  }
};

int main(int argc, char* argv[]) {
  // args.add_argument("model_path").help("Path to object classification model");
  args.add_argument("--sim").default_value(false).implicit_value(true).help(
    "Run in simulation mode");
  args.add_argument("-d", "--device")
    .default_value(std::string("/dev/ttyUSB0"))
    .help("Serial port to talk to rover");
  args.add_argument("-t", "--tree")
    .default_value(0.03f)
    .help("Occupancy map resolution (in metres)").scan<'g', float>();
  args.add_argument("-c", "--camera-height")
    .default_value(0.14f)
    .help("Camera height (in metres)").scan<'g', float>();

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
  std::any mission;
  if (args.get<bool>("--sim")) {
    print("Starting in SIM mode\n\n");
    auto gi = std::make_shared<GazeboInterface>();

    mission = std::make_shared<
      AnantaMission<GazeboDepthCamPolicy, GazeboImuPolicy, GazeboOdomPolicy>>(gi, gi, gi);
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
    asio::co_spawn(
      io_ctx,
      std::any_cast<std::shared_ptr<AnantaMission<GazeboDepthCamPolicy,
                                                  GazeboImuPolicy,
                                                  GazeboOdomPolicy>>>(mission)
        ->loop(),
      [](std::exception_ptr p) {
        if (p) {
          try {
            std::rethrow_exception(p);
          } catch (const std::exception& e) {
            spdlog::error("Mission loop coroutine threw exception: {}",
                          e.what());
          }
        }
      });
  }
  else {
    print("Starting in HW mode \n\n");
    asio::serial_port dev_serial(io_ctx, args.get("-d"));
    dev_serial.set_option(asio::serial_port_base::baud_rate(115200));
    auto mi = std::make_shared<MotherInterface>(std::move(dev_serial));

    auto [rs_pipe, fovh, fovv] =
      *setup_device(true).or_else([](std::error_code e) {
        spdlog::error("Couldn't setup realsense device: {}", e.message());
      });
    auto rs_dev = std::make_shared<RealsenseDevice>(rs_pipe, io_ctx);

    mission = std::make_shared<AnantaMission<RealsenseDepthCamPolicy,
                                             RealsenseImuPolicy,
                                             MotherOdomPolicy>>(rs_dev, rs_dev, mi);
    asio::co_spawn(io_ctx, mi->loop(), [](std::exception_ptr p) {
      if (p) {
        try {
          std::rethrow_exception(p);
        } catch (const std::exception& e) {
          spdlog::error("MotherInterface loop coroutine threw exception: {}",
                        e.what());
        }
      }
    });
    asio::co_spawn(
      io_ctx,
      std::any_cast<std::shared_ptr<AnantaMission<RealsenseDepthCamPolicy,
                                                  RealsenseImuPolicy,
                                                  MotherOdomPolicy>>>(mission)
        ->loop(),
      [](std::exception_ptr p) {
        if (p) {
          try {
            std::rethrow_exception(p);
          } catch (const std::exception& e) {
            spdlog::error("Mission loop coroutine threw exception: {}",
                          e.what());
          }
        }
      });
  }
  io_ctx.run();
}
