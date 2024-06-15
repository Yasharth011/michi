// In Honour of the boundless Rudra
// Shashwat Ganesh, R24

#include "common.hpp"
#include "ekf2.hpp"
#include "gazebo_interface.hpp"
#include "mother_interface.hpp"
#include "realsense_generator.hpp"
#include <Eigen/Dense>
#include <Fusion/Fusion.h>
#include <argparse/argparse.hpp>
#include <asio/detached.hpp>
#include <chrono>
#include <cmath>
#include <execution>
#include <memory>
#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

using fmt::print;
using tPointcloud = pcl::PointCloud<pcl::PointXYZ>;
using namespace std::literals::chrono_literals;
using std::chrono::steady_clock;

static argparse::ArgumentParser args("TubePlanner");

class RealsenseImuPolicy
{
  Eigen::Matrix<float, 3, 2> m_imu;
  const Eigen::Matrix<float, 3, 3> m_rot;
  int m_reads = 0;

public:
  RealsenseImuPolicy()
  // clang-format off
    : m_rot{{ 0, 0, 1 }, 
            { 1, 0, 0 }, 
            { 0, 1, 0 } }
  {
  }
  // clang-format on

protected:
  using If = RealsenseDevice;
  auto imu_update(std::shared_ptr<If> rs_dev) -> asio::awaitable<void>
  {
    m_imu = co_await rs_dev->async_get_imu();
    m_imu = m_rot * m_imu;
    m_reads = 2;
  }
  auto imu_linear_acceleration(std::shared_ptr<If> rs_dev)
    -> asio::awaitable<Eigen::Vector3f>
  {
    if (not m_reads) {
      co_await imu_update(rs_dev);
    }
    m_reads--;
    co_return m_imu.col(0);
  }
  auto imu_angular_velocity(std::shared_ptr<If> rs_dev)
    -> asio::awaitable<Eigen::Vector3f>
  {
    if (not m_reads) {
      co_await imu_update(rs_dev);
    }
    m_reads--;
    co_return m_imu.col(1);
  }
};
class RealsenseDepthCamPolicy
{
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

    if (pcl_cloud->size() < 50'000)
      co_return pcl_cloud;
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
class GazeboDepthCamPolicy
{
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
class GazeboImuPolicy
{
protected:
  using If = GazeboInterface;
  auto imu_linear_acceleration(std::shared_ptr<If> gi)
    -> asio::awaitable<Eigen::Vector3f>
  {
    co_return gi->imu_linear_acceleration().cast<float>();
  }
  auto imu_angular_velocity(std::shared_ptr<If> gi)
    -> asio::awaitable<Eigen::Vector3f>
  {
    co_return gi->imu_angular_velocity().cast<float>();
  }
};
class BlankOdomPolicy
{
protected:
  using If = std::nullptr_t;
  auto odometry_position(std::shared_ptr<If> gi) -> Eigen::Vector3d
  {
    return Eigen::Vector3d(0, 0, 0);
  }
  auto set_target_velocity(std::shared_ptr<If> gi,
                           Eigen::Vector3d linear,
                           Eigen::Vector3d angular) -> asio::awaitable<void>
  {
    co_return;
  }
};
class GazeboOdomPolicy
{
protected:
  using If = GazeboInterface;
  Eigen::Vector3d m_last_target_velocity{ 0, 0, 0 };
  auto odometry_position(std::shared_ptr<If> gi) -> Eigen::Vector3d
  {
    return gi->odometry_position();
  }
  auto odometry_velocity_heading(std::shared_ptr<If> gi) -> Eigen::Vector4d
  {
    return gi->odometry_velocity_heading();
  }
  auto odom_magnetic_field(std::shared_ptr<If> gi) -> Eigen::Vector3d
  {
    return gi->magnetic_field();
  }
  auto set_target_velocity(std::shared_ptr<If> gi,
                           Eigen::Vector3d linear,
                           Eigen::Vector3d angular) -> asio::awaitable<void>
  {
    m_last_target_velocity << linear(0), 0, angular(2);
    co_return gi->set_target_velocity(linear, angular);
  }
};
class MotherOdomPolicy
{
protected:
  using If = MotherInterface;
  Eigen::Vector3d m_last_target_velocity{ 0, 0, 0 };
  auto odometry_position(std::shared_ptr<If> mi) -> Eigen::Vector3d
  {
    Vector3d xy_blank = mi->odometry_position();
    return Vector3d{ xy_blank(0), xy_blank(1), 0 };
  }
  auto odometry_velocity_heading(std::shared_ptr<If> mi) -> Eigen::Vector4d
  {
    return mi->odometry_velocity_heading();
  }
  auto odom_magnetic_field(std::shared_ptr<If> mi) -> Eigen::Vector3d
  {
    return mi->magnetic_field();
  }
  auto set_target_velocity(std::shared_ptr<If> mi,
                           Eigen::Vector3d linear,
                           Eigen::Vector3d angular) -> asio::awaitable<void>
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
  , public std::enable_shared_from_this<
      AnantaMission<DepthCamPolicy, BaseImuPolicy, OdomPolicy>>
{
  using BaseImuPolicy::imu_angular_velocity;
  using BaseImuPolicy::imu_linear_acceleration;
  using DepthCamPolicy::async_get_pointcloud;
  using OdomPolicy::m_last_target_velocity;
  using OdomPolicy::odom_magnetic_field;
  using OdomPolicy::odometry_position;
  using OdomPolicy::odometry_velocity_heading;
  using OdomPolicy::set_target_velocity;
  using Mission = AnantaMission<DepthCamPolicy, BaseImuPolicy, OdomPolicy>;

  struct
  {
    const FusionMatrix gyroscopeMisalignment = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                 0.0f, 0.0f, 0.0f, 1.0f };
    const FusionVector gyroscopeSensitivity = { 1.0f, 1.0f, 1.0f };
    const FusionVector gyroscopeOffset = { 0.0f, 0.0f, 0.0f };
    const FusionMatrix accelerometerMisalignment = { 1.0f, 0.0f, 0.0f,
                                                     0.0f, 1.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f };
    const FusionVector accelerometerSensitivity = { 4.0f, 1.0f, 1.0f };
    FusionVector accelerometerOffset = { 0.0f, 0.0f, 0.0f };
    const FusionMatrix softIronMatrix = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                          0.0f, 0.0f, 0.0f, 1.0f };
    const FusionVector hardIronOffset = { 0.0f, 0.0f, 0.0f };
    const unsigned int SAMPLE_RATE = 100;
    FusionAhrsSettings settings = {
      .convention = FusionConventionNwu,
      .gain = 0.5f,
      .gyroscopeRange = 1000.0f, /* actual gyroscope range in degrees/s */
      .accelerationRejection = 10.0f,
      .magneticRejection = 10.0f,
      .recoveryTriggerPeriod = 5 * SAMPLE_RATE, /* 5 seconds */
    };
    std::optional<std::chrono::time_point<std::chrono::steady_clock>> timestamp;
  } m_madgwick_params;

  class AvoidAction {
    public:
      static constexpr float EXTENT_X_METRES = 1;
      static constexpr float EXTENT_Y_METRES = 0.5;
      static constexpr float EXTENT_Z_METRES = 0.03;

      // template <typename A, typename B, typename C>
      static int utility(AnantaMission::Mission* mission) {
        return 1;
            // octomap::OcTreeKey key = mission->m_tree.coordToKey()
        //     octomap::OcTreeNode *node;
        //     double res = mission->m_tree.getResolution();
        //     int num_voxels = 1;

        //     // look up...
        //     octomap::OcTreeKey key1(key);
        //     while(true)
        //     {
        //         key1[2]++;
        //         node = mission->m_tree.search(key1);
        //         if(!node) break;
        //         if(node && !mission->m_tree.isNodeOccupied(node)) break;
        //         num_voxels++;
        //     }

        //     // look down...
        //     octomap::OcTreeKey key2(key);
        //     while(true)
        //     {
        //         key2[2]--;
        //         node = mission->m_tree.search(key2);
        //         if(!node) break;
        //         if(node && !mission->m_tree.isNodeOccupied(node)) break;
        //         num_voxels++;
        //     }

        //     auto max_superable_height_ = 0.3;
        //     return res * num_voxels - max_superable_height_;
        // // auto min = mission->m_tree.coordToKey(mission->m_position_est(0),
        //                                       mission->m_position_est(1) -
        //                                         EXTENT_Y_METRES,
        //                                       -(EXTENT_Z_METRES + 0.2));
      }

    static auto execute(AnantaMission::Mission* mission)
      -> asio::awaitable<void>
    {
      // mission->set_target_velocity(mission->m_odom_if, {0.0f, 0.0f, 0.0f},
      // {0.0f, 0.0f, 1.57f});
      co_await mission->set_target_velocity(
        mission->m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.1f, 0.0f, 0.0f });
      asio::steady_timer timer(co_await asio::this_coro::executor);
      timer.expires_after(1s);
      co_await timer.async_wait(use_nothrow_awaitable);
      co_await mission->set_target_velocity(
        mission->m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
      co_return;
    }
  };
  auto move() -> asio::awaitable<void>
  {
    auto instant_direction = Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                                      m_desired_pos(1) - m_position_heading(1) }
                       .normalized();
    auto angular_displacement = (M_PI_2 - std::atan2(instant_direction(0), instant_direction(1))) - m_position_heading(2);
    asio::steady_timer timer(co_await asio::this_coro::executor);

    if (std::fabs(angular_displacement) > 0.35f) {
      spdlog::info("Turning");
      double ang_vel = 1.0f;
      // auto ang_time =
      // std::chrono::milliseconds(int(std::abs(angular_displacement) * 1000 /
      // ang_vel));
      auto ang_time = 1s;
      if (angular_displacement < 0)
        ang_vel *= -1;
      while (true) {
        instant_direction = Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                                          m_desired_pos(1) - m_position_heading(1) }
                           .normalized();
        angular_displacement = (M_PI_2 - std::atan2(instant_direction(0), -instant_direction(1))) - m_position_heading(2);
      if (angular_displacement > 0)
        ang_vel *= -1;
        spdlog::info("Angular displacement: {:2.2f}° from WP | Currently at {::2.2f}",
                     angular_displacement * 180 / M_PI, m_position_heading);
        co_await set_target_velocity(
          m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, ang_vel });
        if (std::fabs(angular_displacement) < 0.35f) break;
        timer.expires_after(ang_time);
        co_await timer.async_wait(use_nothrow_awaitable);
      }
      spdlog::info("Stopped turning");
      auto displacement = Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                                        m_desired_pos(1) - m_position_heading(1) };
      instant_direction =
        Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                         m_desired_pos(1) - m_position_heading(1) }
          .normalized();
      while (std::sqrt(displacement.norm()) > 0.10f) {
        spdlog::info("Distance to WP: {:2.2f}m | Currently at {::2.2f}",
                     std::sqrt(displacement.norm()),
                     m_position_heading);
        co_await set_target_velocity(m_odom_if,
                                     { args.get<float>("-s"), 0.0f, 0.0f },
                                     { 0.0f, 0.0f, 0.0f });
        timer.expires_after(300ms);
        co_await timer.async_wait(use_nothrow_awaitable);
        displacement = Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                                          m_desired_pos(1) - m_position_heading(1) };
        instant_direction =
          Eigen::Vector2d{ m_desired_pos(0) - m_position_heading(0),
                           m_desired_pos(1) - m_position_heading(1) }
            .normalized();
      }
      co_await set_target_velocity(
        m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
    }
  }
  auto avoid() -> asio::awaitable<void>
  {
    co_await set_target_velocity(
      m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.57f });
    asio::steady_timer timer(co_await asio::this_coro::executor);
    timer.expires_after(1s);
    co_await timer.async_wait(use_nothrow_awaitable);
    co_return;
  }
  void fusion_update(FusionAhrs* ahrs,
                     FusionOffset* offset,
                     Vector3f lax,
                     Vector3f avl,
                     Vector3d mag)
  {
    FusionVector gyroscope = {
      float(avl(0)), float(avl(1)), float(avl(2))
    }; // actual gyroscope data in degrees/s
    FusionVector accelerometer = {
      float(lax(0)), float(lax(1)), float(lax(2))
    }; // actual accelerometer data in g
    FusionVector magnetometer = {
      float(mag(0)), float(mag(1)), float(mag(2))
    }; // actual magnetometer data in arbitrary units
    gyroscope =
      FusionCalibrationInertial(gyroscope,
                                m_madgwick_params.gyroscopeMisalignment,
                                m_madgwick_params.gyroscopeSensitivity,
                                m_madgwick_params.gyroscopeOffset);
    accelerometer =
      FusionCalibrationInertial(accelerometer,
                                m_madgwick_params.accelerometerMisalignment,
                                m_madgwick_params.accelerometerSensitivity,
                                m_madgwick_params.accelerometerOffset);
    magnetometer = FusionCalibrationMagnetic(magnetometer,
                                             m_madgwick_params.softIronMatrix,
                                             m_madgwick_params.hardIronOffset);

    // Update gyroscope offset correction algorithm
    gyroscope = FusionOffsetUpdate(offset, gyroscope);
    auto now = std::chrono::steady_clock::now();
    double diff_time = 0.0;
    if (not m_madgwick_params.timestamp)
      diff_time = 0.01;
    else
      diff_time = std::chrono::duration<double>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                      now - *m_madgwick_params.timestamp))
                    .count();
    m_madgwick_params.timestamp.emplace(now);

    FusionAhrsUpdate(ahrs, gyroscope, accelerometer, magnetometer, diff_time);
  }

public:
  octomap::OcTree m_tree;
  octomap::Pointcloud m_map_cloud;
  std::shared_ptr<typename DepthCamPolicy::If> m_ci;
  std::shared_ptr<typename BaseImuPolicy::If> m_imu_if;
  std::shared_ptr<typename OdomPolicy::If> m_odom_if;
  int m_iterations;
  const float m_camera_height;
  std::optional<Eigen::Quaterniond> m_initial_orientation;
  Eigen::Matrix<double, 6, 1> m_state;
  Eigen::Matrix<double, 3, 1> m_position_heading;
  Eigen::Matrix<double, 2, 1> m_desired_pos;
  AnantaMission(std::shared_ptr<typename DepthCamPolicy::If> ci,
                std::shared_ptr<typename BaseImuPolicy::If> imu_if,
                std::shared_ptr<typename OdomPolicy::If> odom_if)
    : m_tree(args.get<float>("-t"))
    , m_iterations(0)
    , m_ci(ci)
    , m_imu_if(imu_if)
    , m_odom_if(odom_if)
    , m_camera_height(args.get<float>("-c"))
    , m_state({ 0, 0, 0, 0, 0, 0 })
  {
  }
  auto localization() -> asio::awaitable<void>
  {
    auto this_exec = co_await asio::this_coro::executor;
    asio::steady_timer timer(this_exec);
    spdlog::info("Localization co-routine started");
    // Setup Madgwick
    if constexpr (std::is_same<BaseImuPolicy, GazeboImuPolicy>::value) {
      m_madgwick_params.settings.gyroscopeRange = 2000.0f;
    }
    FusionOffset offset;
    FusionAhrs ahrs;
    FusionOffsetInitialise(&offset, m_madgwick_params.SAMPLE_RATE);
    FusionAhrsInitialise(&ahrs);
    FusionAhrsSetSettings(&ahrs, &m_madgwick_params.settings);

    spdlog::info("Calibrating IMU...");
    auto calib_start = steady_clock::now();
    Vector3f lax_sum;
    int i = 0;
    for (auto now = steady_clock::now(); now < calib_start + 5s;
         now = steady_clock::now()) {
      i++;
      Vector3f linear_accel = co_await imu_linear_acceleration(m_imu_if);
      lax_sum += linear_accel;
      Vector3f angular_vel = co_await imu_angular_velocity(m_imu_if);
      Vector3d magnetic_field = odom_magnetic_field(m_odom_if);
    }
    FusionVector initial_earth = FusionAhrsGetEarthAcceleration(&ahrs);
    auto initial_imu_offset = lax_sum / i;
    m_madgwick_params.accelerometerOffset.array[0] = initial_imu_offset(0);
    m_madgwick_params.accelerometerOffset.array[1] = initial_imu_offset(1);
    m_madgwick_params.accelerometerOffset.array[2] = - 9.882 + initial_imu_offset(2);
    spdlog::info("IMU Offset {::2.2f}", initial_imu_offset);

    {
      auto linear_accel = co_await imu_linear_acceleration(m_imu_if);
      auto angular_vel = co_await imu_angular_velocity(m_imu_if);
      auto mag_field = odom_magnetic_field(m_odom_if);
      fusion_update(&ahrs, &offset, linear_accel, angular_vel, mag_field);
    }

    auto fusion_initial_quaternion = FusionAhrsGetQuaternion(&ahrs);
    m_initial_orientation.emplace(fusion_initial_quaternion.element.w,
                                  fusion_initial_quaternion.element.x,
                                  fusion_initial_quaternion.element.y,
                                  fusion_initial_quaternion.element.z);
    spdlog::info("Got initial orientation: {:0.2f}+{:0.2f}î+{:0.2f}ĵ+{:0.2f}̂k̂",
                 m_initial_orientation->w(),
                 m_initial_orientation->x(),
                 m_initial_orientation->y(),
                 m_initial_orientation->z());

    auto ekf = EKF2();
    std::ofstream position_file("positions.txt");
    while (true) {
      auto linear_accel = co_await imu_linear_acceleration(m_imu_if);
      auto angular_vel = co_await imu_angular_velocity(m_imu_if);
      auto mag_field = odom_magnetic_field(m_odom_if);
      auto odom_vel = odometry_velocity_heading(m_odom_if);
      fusion_update(&ahrs, &offset, linear_accel, angular_vel, mag_field);

      auto imu_quaternion = FusionAhrsGetQuaternion(&ahrs);
      Eigen::Quaterniond orientation(imu_quaternion.element.w,
                                     imu_quaternion.element.x,
                                     imu_quaternion.element.y,
                                     imu_quaternion.element.z);
      const FusionVector earth = FusionAhrsGetEarthAcceleration(&ahrs);
      auto filtered_imu_accel = Eigen::Vector3d{ earth.axis.x, earth.axis.y, earth.axis.z };

      double imu_heading = m_initial_orientation->angularDistance(orientation);
      auto ekf_control_ip =
        Eigen::Matrix<double, 6, 1>{ 0, m_last_target_velocity(0),
                                     0, m_last_target_velocity(1),
                                     0, m_last_target_velocity(2) };
      ekf.predict(ekf_control_ip);
      auto ekf_measurements = Eigen::Matrix<double, 4, 1>{
        odom_vel(0), odom_vel(1), odom_vel(3), odom_vel(2)
      };
      auto estimates = ekf.correct(ekf_measurements).first;
      auto odom_pos = odometry_position(m_odom_if);
      spdlog::debug(
        "Current position: x {:2.2f} y {:2.2f} {:2.2f}° | Odom: x {:2.2f} y "
        "{:2.2f} vel {:2.2f} {:2.2f} heading {:2.2f}° | IMU heading lax {::2.2f} {:2.2f}°",
        estimates(0),
        estimates(2),
        std::fmod(estimates(4) * 180 / M_PI, 360.0f),
        odom_pos(0),
        odom_pos(1),
        odom_vel(0),
        odom_vel(1),
        std::fmod(odom_vel(3) * 180 / M_PI, 360.0f),
        filtered_imu_accel,
        imu_heading);

      position_file << estimates(0) << " " << estimates(1) << '\n';
      m_state = estimates;
      m_position_heading = Eigen::Vector3d{ estimates(0),
                                            estimates(2),
                                            std::fmod(estimates(4), 2 * M_PI) };
      timer.expires_after(400ms);
      co_await timer.async_wait(use_nothrow_awaitable);
    }
  }
  auto loop() -> asio::awaitable<void>
  {
    spdlog::info("Camera height: {}m", m_camera_height);
    auto this_exec = co_await asio::this_coro::executor;
    asio::steady_timer timer(this_exec);
    double front_prob = 1;

    int target_index = 0;
    std::vector<Eigen::Vector2d> targets{ Eigen::Vector2d{ 1.4, -1.7 },
                                          Eigen::Vector2d{ 2.3, -2.2 },
                                          Eigen::Vector2d{ 2.9, -3.0 },
                                          Eigen::Vector2d{ 2.9, -2.1 } };
    timer.expires_after(5s);
    co_await timer.async_wait(use_nothrow_awaitable);
    while (true) {
      octomap::point3d map_current_pos(
        m_position_heading(0), m_position_heading(1), m_camera_height);
      auto cam_cloud = co_await async_get_pointcloud(m_ci);
      m_map_cloud.reserve(cam_cloud->points.size());
      int nulls = 0;
      for (const auto& point : cam_cloud->points) {
        if (std::isinf(point.x) or std::isinf(point.y) or std::isinf(point.z)) {
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
      else if (target_index < targets.size() and std::abs(
                 targets[target_index].norm() -
                 Eigen::Vector2d(m_position_heading(0), m_position_heading(1))
                   .norm()) > 0.1) {
        co_await move();
        spdlog::info("Moving to WP#{} {::2.2f}",
                     target_index,
                     targets[target_index]);
      } else if (target_index < targets.size()) {
        co_await set_target_velocity(
          m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
        spdlog::info("Reached waypoint#{} {::2.2f}",
                     target_index,
                     targets[target_index]);
        target_index++;
      }

      m_iterations++;
      if (m_iterations == 10)
        spdlog::info("Wrote tree: {}", m_tree.write("m_tree.ot"));
      // if constexpr (std::is_same<DepthCamPolicy, GazeboDepthCamPolicy>::value) {
        timer.expires_after(10ms);
        co_await timer.async_wait(use_nothrow_awaitable);
      // }
    }

    co_return;
  }
};

int
main(int argc, char* argv[])
{
  // args.add_argument("model_path").help("Path to object classification
  // model");
  args.add_argument("--sim").default_value(false).implicit_value(true).help(
    "Run in simulation mode");
  args.add_argument("-d", "--device")
    .default_value(std::string("/dev/ttyUSB0"))
    .help("Serial port to talk to rover");
  args.add_argument("-t", "--tree")
    .default_value(0.03f)
    .help("Occupancy map resolution (in metres)")
    .scan<'g', float>();
  args.add_argument("-c", "--camera-height")
    .default_value(0.14f)
    .help("Camera height (in metres)")
    .scan<'g', float>();
  args.add_argument("-f", "--control-input")
    .default_value(2.0f)
    .help("Linear control input in m/s")
    .scan<'g', float>();
  args.add_argument("-s", "--speed")
    .default_value(0.3f)
    .help("Linear speed in m/s")
    .scan<'g', float>();
  args.add_argument("-ox")
    .default_value(1.0f)
    .help("Waypoint offset in x")
    .scan<'g', float>();
  args.add_argument("-oy")
    .default_value(-0.3f)
    .help("Waypoint offset in y")
    .scan<'g', float>();

  int log_verbosity = 0;
  args.add_argument("-V", "--verbose")
    .action([&](const auto&) { ++log_verbosity; })
    .append()
    .default_value(false)
    .implicit_value(true)
    .nargs(0);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
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
      AnantaMission<GazeboDepthCamPolicy, GazeboImuPolicy, GazeboOdomPolicy>>(
      gi, gi, gi);
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
        ->localization(),
      [](std::exception_ptr p) {
        if (p) {
          try {
            std::rethrow_exception(p);
          } catch (const std::exception& e) {
            spdlog::error("Localization coroutine threw exception: {}",
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
  } else {
    print("Starting in HW mode \n\n");
    asio::serial_port dev_serial(io_ctx, args.get("-d"));
    dev_serial.set_option(asio::serial_port_base::baud_rate(115200));
    auto mi = std::make_shared<MotherInterface>(std::move(dev_serial));

    auto [rs_pipe, fovh, fovv] =
      *setup_device(true).or_else([](std::error_code e) {
        spdlog::error("Couldn't setup realsense device: {}", e.message());
      });
    auto rs_dev = std::make_shared<RealsenseDevice>(rs_pipe, io_ctx);

    mission =
      std::make_shared<AnantaMission<RealsenseDepthCamPolicy,
                                     RealsenseImuPolicy,
                                     MotherOdomPolicy>>(rs_dev, rs_dev, mi);
    asio::co_spawn(
      io_ctx,
      std::any_cast<std::shared_ptr<AnantaMission<RealsenseDepthCamPolicy,
                                                  RealsenseImuPolicy,
                                                  MotherOdomPolicy>>>(mission)
        ->localization(),
      [](std::exception_ptr p) {
        if (p) {
          try {
            std::rethrow_exception(p);
          } catch (const std::exception& e) {
            spdlog::error("Localization coroutine threw exception: {}",
                          e.what());
          }
        }
      });
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
