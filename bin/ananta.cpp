// In Honour of the boundless Rudra
// Shashwat Ganesh, R24

#include "common.hpp"
#include "ekf2.hpp"
#include "gazebo_interface.hpp"
#include "mother_interface.hpp"
#include "realsense_generator.hpp"
#include <Eigen/Dense>
#include <Fusion/Fusion.h>
#include <algorithm>
#include <argparse/argparse.hpp>
#include <asio/experimental/coro.hpp>
#include <asio/experimental/use_coro.hpp>
#include <asio/detached.hpp>
#include <chrono>
#include <cmath>
#include <execution>
#include <limits>
#include <memory>
#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <CGAL/Point_2.h>
#include <CGAL/Bbox_2.h>
#include <CGAL/Circle_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/squared_distance_2.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/Path.h>
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
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGAL::Bbox_2;
using IsoRectangle_2 = CGAL::Iso_rectangle_2<Kernel>;
using Point_2 = Kernel::Point_2;
using Circle_2 = CGAL::Circle_2<Kernel>;
namespace ob = ompl::base;
namespace og = ompl::geometric;
using asio::experimental::coro;
using asio::experimental::use_coro;

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
    -> asio::awaitable<std::pair<double, cv::Mat>>
  {
    auto rgb_frame = co_await rs_dev->async_get_rgb_frame();
    cv::Mat image(
      cv::Size(640, 480), CV_8UC3, const_cast<void*>(rgb_frame.get_data()));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    co_return std::make_pair(rgb_frame.get_timestamp() * 1e-3, image.clone());
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
class AnantaMission;

template<typename A, typename B, typename C>
class RrtMotionPlanner {
  std::vector<IsoRectangle_2> m_small_squares;
  std::vector<IsoRectangle_2> m_large_squares;
  std::vector<Circle_2> m_small_circles;
  std::vector<Circle_2> m_large_circles;
  const double m_small_square_side_metres;
  const double m_large_square_side_metres;
  const double m_small_circle_radius_metres;
  const double m_large_circle_radius_metres;
  ob::StateSpacePtr m_space;
  ob::SpaceInformationPtr m_space_info;
  ob::ProblemDefinitionPtr m_problem_def;
  int m_target_idx = 0;

  ob::OptimizationObjectivePtr path_length_objective(const ob::SpaceInformationPtr& si) {
    return ob::OptimizationObjectivePtr(new ob::PathLengthOptimizationObjective(si));
  }
  class ValidityChecker : public ob::StateValidityChecker {
    const std::vector<IsoRectangle_2>& m_squares;
    const std::vector<Circle_2>& m_circles;
    double point_to_rectangle_distance(const Point_2& p,
                                       const IsoRectangle_2& rect) const
    {
      // Extract the boundaries of the rectangle
      double x_min = rect.min().x();
      double y_min = rect.min().y();
      double x_max = rect.max().x();
      double y_max = rect.max().y();

      double x = p.x();
      double y = p.y();

      // If the point is inside the rectangle, the distance is 0
      if (x >= x_min && x <= x_max && y >= y_min && y <= y_max) {
        return 0.0;
      }

      // Calculate distances to the sides of the rectangle
      double dx = std::max({ x_min - x, 0.0, x - x_max });
      double dy = std::max({ y_min - y, 0.0, y - y_max });

      // If the point is aligned horizontally or vertically with the rectangle
      if (dx == 0) {
        return dy;
      }
      if (dy == 0) {
        return dx;
      }

      // Otherwise, the point is outside and diagonal to a corner
      return CGAL::sqrt(dx * dx + dy * dy);
    }
    double point_to_circle_distance(const Point_2& point, const Circle_2& circle) const {
      auto val = CGAL::squared_distance(point, circle.center()) - circle.squared_radius();
      return val > 0 ? val : 0;
    }

    public:
      ValidityChecker(const ob::SpaceInformationPtr& si,
                      const std::vector<IsoRectangle_2>& large_sq,
                      const std::vector<Circle_2>& large_cir)
        : ob::StateValidityChecker(si)
        , m_squares(large_sq)
        , m_circles(large_cir)
      {
      }
      bool isValid(const ob::State* state) const
      {
        return this->clearance(state) > args.get<float>("-a");
      }
      double clearance(const ob::State* state) const {
        const ob::RealVectorStateSpace::StateType* state2D =
          state->as<ob::RealVectorStateSpace::StateType>();
        Point_2 p(state2D->values[0], state2D->values[1]);

        double min_large_sq_dist = std::numeric_limits<double>::infinity();
        for (auto& sq : m_squares) {
          min_large_sq_dist =
            std::min(min_large_sq_dist, point_to_rectangle_distance(p, sq));
        }

        double min_large_cir_dist = std::numeric_limits<double>::infinity();
        for (auto& cir : m_circles) {
          min_large_cir_dist =
            std::min(min_large_cir_dist, point_to_circle_distance(p, cir));
        }

        return std::min(std::sqrt(min_large_cir_dist), min_large_sq_dist);
      }
  };
  static auto pivot_to_rect(Eigen::Vector2d pivot, Kernel::FT side_len)
    -> IsoRectangle_2
  {
    Point_2 bottomRight(pivot.x(), pivot.y());
    Kernel::FT xMin = bottomRight.x() + side_len;
    Kernel::FT yMin = bottomRight.y() + side_len;
    Point_2 topLeft(xMin, yMin);
    return IsoRectangle_2(topLeft, bottomRight);
  }
  static auto pivot_to_circle(Eigen::Vector2d pivot, Kernel::FT radius)
    -> Circle_2
  {
    Point_2 c(pivot.x(), pivot.y());
    double squaredRadius = radius * radius;
    return Circle_2(c, squaredRadius);
  }

public:
  RrtMotionPlanner(std::vector<Eigen::Vector2d> small_sq_pivots,
                   std::vector<Eigen::Vector2d> large_sq_pivots,
                   std::vector<Eigen::Vector2d> small_circle_pivots,
                   std::vector<Eigen::Vector2d> large_circle_pivots,
                   double small_sq_side_metres = 0.15,
                   double large_sq_side_metres = 0.3,
                   double small_circle_radius_metres = 0.1,
                   double large_circle_radius_metres = 0.2)
    : m_small_square_side_metres(small_sq_side_metres)
    , m_large_square_side_metres(large_sq_side_metres)
    , m_small_circle_radius_metres(small_circle_radius_metres)
    , m_large_circle_radius_metres(large_circle_radius_metres)
    , m_space(new ob::RealVectorStateSpace(2))
    , m_space_info(new ob::SpaceInformation(m_space))
  {
    std::transform(small_sq_pivots.begin(),
                   small_sq_pivots.end(),
                   m_small_squares.begin(),
                   [this](Eigen::Vector2d pivot) {
                     return pivot_to_rect(pivot, this->m_small_square_side_metres);
                   });
    std::transform(large_sq_pivots.begin(),
                   large_sq_pivots.end(),
                   std::back_inserter(m_large_squares),
                   [this](Eigen::Vector2d pivot) {
                     return pivot_to_rect(pivot, this->m_large_square_side_metres);
                   });

    std::transform(large_circle_pivots.begin(),
                   large_circle_pivots.end(),
                   std::back_inserter(m_large_circles),
                   [this](Eigen::Vector2d pivot) {
                     return pivot_to_circle(pivot, this->m_large_circle_radius_metres);
                   });

    auto space_bounds = ob::RealVectorBounds(2);
    space_bounds.setLow(0, 0);
    space_bounds.setHigh(0, 10);
    space_bounds.setLow(1, -5);
    space_bounds.setHigh(1, 0);
    m_space->as<ob::RealVectorStateSpace>()->setBounds(space_bounds);
    m_space_info->setStateValidityChecker(ob::StateValidityCheckerPtr(
      new ValidityChecker(m_space_info, m_large_squares, m_large_circles)));
    m_space_info->setup();
    m_problem_def = ob::ProblemDefinitionPtr(new ob::ProblemDefinition(m_space_info));
  }

  auto plan(Eigen::Vector2d start, Eigen::Vector2d goal)
    -> std::optional<const og::PathGeometric*>
  {
    ob::ScopedState<> start_p(m_space);
    start_p->as<ob::RealVectorStateSpace::StateType>()->values[0] = start.x();
    start_p->as<ob::RealVectorStateSpace::StateType>()->values[1] = start.y();

    ob::ScopedState<> goal_p(m_space);
    goal_p->as<ob::RealVectorStateSpace::StateType>()->values[0] = goal.x();
    goal_p->as<ob::RealVectorStateSpace::StateType>()->values[1] = goal.y();

    m_problem_def = ob::ProblemDefinitionPtr(new ob::ProblemDefinition(m_space_info));
    m_problem_def->setStartAndGoalStates(start_p, goal_p);
    m_problem_def->setOptimizationObjective(path_length_objective(m_space_info));

    auto planner = ob::PlannerPtr(new og::RRTstar(m_space_info));
    planner->setProblemDefinition(m_problem_def);
    planner->setup();
    std::optional<const og::PathGeometric*> path;
    if (planner->solve(1)) {
      auto solved_path = m_problem_def->getSolutionPath()->as<og::PathGeometric>();
      solved_path->printAsMatrix(std::cout);
      path.emplace(solved_path);
    }
    return path;
  }
  int utility(AnantaMission<A, B, C>* mission) {
    if (not mission->m_path and m_target_idx + 1 < mission->m_targets.size()) {
      return 40;
    }
    if (m_target_idx + 1 == mission->m_targets.size()) return 0;
    if (not mission->m_path) return 40;
    if (mission->m_wp_idx == mission->m_path.value()->getStateCount()) return 40;

    auto path_ptr = mission->m_path.value();
    ob::ScopedState<> pos_p(m_space);
    pos_p->as<ob::RealVectorStateSpace::StateType>()->values[0] =
      mission->m_position_heading(0);
    pos_p->as<ob::RealVectorStateSpace::StateType>()->values[1] =
      mission->m_position_heading(1);
    if (path_ptr->getClosestIndex(pos_p.get()) + 1 != path_ptr->getStateCount())
      return 20;
    auto closest_p = path_ptr->getState(path_ptr->getClosestIndex(pos_p.get()));
    Point_2 closest{
      closest_p->template as<ob::RealVectorStateSpace::StateType>()->values[0],
      closest_p->template as<ob::RealVectorStateSpace::StateType>()->values[1]};
    Point_2 pos(mission->m_position_heading(0), mission->m_position_heading(1));
    if (CGAL::squared_distance(pos, closest) <= 0.1 * 0.1) {
      m_target_idx++;
      return 40;
    }
    return 0;
  }
  auto action(AnantaMission<A,B,C>* mission) -> asio::awaitable<void> {
    auto goal = mission->m_targets[m_target_idx];
    auto now = Eigen::Vector2d{ mission->m_position_heading(0),
                                mission->m_position_heading(1) };
    spdlog::critical("Planning from {::2.2f} to target#{} {::2.2f}", now, m_target_idx, goal);
    mission->m_path = plan(now, goal);
    co_return;
  }
};
template<typename A, typename B, typename C>
class MoveAction {
  double m_kp = args.get<float>("-kp"), m_ki = args.get<float>("-ki"), m_kd = args.get<float>("-kd"), m_arrive_distance = 0.2,
         m_r = 0.0325, m_l = 0.1, m_e = 0, m_old_e = 0, m_desired_vel = args.get<float>("-s");

  auto iterate_pid(Eigen::Vector3d current, Eigen::Vector2d goal)
    -> Eigen::Vector2d
  {
    double d_x = goal.x() - current(0);
    double d_y = goal.y() - current(1);
    double g_theta = atan2(d_y, d_x);
    double alpha = g_theta - current(2);
    double e = atan2(sin(alpha), cos(alpha));
    if (std::fabs(e) < 0.08) return Eigen::Vector2d(m_desired_vel, 0);
    double e_P = e;
    double e_I = m_e + e;
    double e_D = e - m_old_e;
    double w_i = m_kp * e_P + m_ki * e_I + m_kd * e_D;
    double w = atan2(sin(w_i), cos(w_i));
    m_e = m_e + e;
    m_old_e = e;
    double v = m_desired_vel;
    spdlog::info("g: {:2.2f} c: {:2.2f} d: {:2.2f} | {:2.2f} {:2.2f}",
                 g_theta * 180 / M_PI,
                 current(2) * 180 / M_PI,
                 std::sqrt(Eigen::Vector2d(d_x, d_y).norm()),
                 e,
                 w);
    return Eigen::Vector2d(v, w);
  }

  public:
  int utility(AnantaMission<A, B, C>* mission) {
    if (not mission->m_path) return 0;
    Point_2 pos(mission->m_position_heading(0), mission->m_position_heading(1)),
      desired(mission->m_desired_pos(0), mission->m_desired_pos(1));
    if (CGAL::squared_distance(pos, desired) > 0.1*0.1) {
      return 30;
    }
    return 0;
  }
  double fixAngle(double angle) {
      return atan2(sin(angle), cos(angle));
  }
  auto action(AnantaMission<A,B,C>* mission) -> asio::awaitable<void> {
    auto current = mission->m_position_heading;
    auto goal = mission->m_desired_pos;
    auto displacement = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                        mission->m_desired_pos(1) - mission->m_position_heading(1) };
    asio::steady_timer timer(co_await asio::this_coro::executor);
    while (std::sqrt(displacement.norm()) > m_arrive_distance) {
      auto change =
        iterate_pid(mission->m_position_heading, mission->m_desired_pos);
      while(change(1) != 0) {
        co_await mission->set_target_velocity(
          mission->m_odom_if, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, change(1) });
        timer.expires_after(50ms);
        co_await timer.async_wait(use_nothrow_awaitable);
        change = iterate_pid(mission->m_position_heading, mission->m_desired_pos);
      }

      co_await mission->set_target_velocity(
        mission->m_odom_if, { m_desired_vel, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f});
      timer.expires_after(1s);
      co_await timer.async_wait(use_nothrow_awaitable);
      displacement = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                        mission->m_desired_pos(1) - mission->m_position_heading(1) };
      // spdlog::info("Dist: {:2.2f} | {::2.2f}", std::sqrt(displacement.norm()), change);
    }
  }
  auto action_old(AnantaMission<A, B, C>* mission) -> asio::awaitable<void>
  {
    auto instant_direction = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                      mission->m_desired_pos(1) - mission->m_position_heading(1) }
                       .normalized();
    auto angular_displacement = (M_PI_2 - std::atan2(instant_direction(0), -instant_direction(1))) - mission->m_position_heading(2);
    auto displacement = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                        mission->m_desired_pos(1) - mission->m_position_heading(1) };
    asio::steady_timer timer(co_await asio::this_coro::executor);

    if (std::fabs(angular_displacement) > 0.35f and std::sqrt(displacement.norm()) > 0.10f) {
        spdlog::info("Turning");
        double ang_vel = M_PI_4f;
        // auto ang_time =
        // std::chrono::milliseconds(int(std::abs(angular_displacement) * 1000 /
        // ang_vel));
        auto ang_time = 15ms;
        if (angular_displacement > 0 and ang_vel > 0)
          ang_vel *= -1;
        while (true) {
          instant_direction = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                            mission->m_desired_pos(1) - mission->m_position_heading(1) }
                             .normalized();
          angular_displacement = (M_PI_2 - std::atan2(instant_direction(0), -instant_direction(1))) - mission->m_position_heading(2);
          if (angular_displacement > 0 and ang_vel > 0)
            ang_vel *= -1;
          spdlog::info("Angular displacement: {:2.2f}° from WP | Currently at {::2.2f}",
                       angular_displacement * 180 / M_PI, mission->m_position_heading);
          co_await mission->set_target_velocity(
            mission->m_odom_if, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, ang_vel });
          if (std::fabs(angular_displacement) < 0.35f) break;
          timer.expires_after(ang_time);
          co_await timer.async_wait(use_nothrow_awaitable);
        }
      }
      spdlog::info("Stopped turning");
      displacement = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                        mission->m_desired_pos(1) - mission->m_position_heading(1) };
      instant_direction =
        Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                         mission->m_desired_pos(1) - mission->m_position_heading(1) }
          .normalized();
      if (std::sqrt(displacement.norm()) > 0.10f) {
        displacement = Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                                          mission->m_desired_pos(1) - mission->m_position_heading(1) };
        instant_direction =
          Eigen::Vector2d{ mission->m_desired_pos(0) - mission->m_position_heading(0),
                           mission->m_desired_pos(1) - mission->m_position_heading(1) }
            .normalized();
        spdlog::info("Distance to WP: {:2.2f}m | Currently at {::2.2f}",
                     std::sqrt(displacement.norm()),
                     mission->m_position_heading);
        co_await mission->set_target_velocity(mission->m_odom_if,
                                     { args.get<float>("-s"), 0.0f, 0.0f },
                                     { 0.0f, 0.0f, 0.0f });
        timer.expires_after(20ms);
        co_await timer.async_wait(use_nothrow_awaitable);
        
      }
        co_return;
        // timer.expires_after(100ms);
        // co_await timer.async_wait(use_nothrow_awaitable);
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
  public:
  using BaseImuPolicy::imu_angular_velocity;
  using BaseImuPolicy::imu_linear_acceleration;
  using DepthCamPolicy::async_get_pointcloud;
  using OdomPolicy::m_last_target_velocity;
  using OdomPolicy::odom_magnetic_field;
  using OdomPolicy::odometry_position;
  using OdomPolicy::odometry_velocity_heading;
  using OdomPolicy::set_target_velocity;
  using Mission = AnantaMission<DepthCamPolicy, BaseImuPolicy, OdomPolicy>;

  private:
  struct
  {
    const FusionMatrix gyroscopeMisalignment = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                 0.0f, 0.0f, 0.0f, 1.0f };
    const FusionVector gyroscopeSensitivity = { 1.0f, 1.0f, 1.0f };
    const FusionVector gyroscopeOffset = { 0.0f, 0.0f, 0.0f };
    const FusionMatrix accelerometerMisalignment = { 1.0f, 0.0f, 0.0f,
                                                     0.0f, 1.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f };
    const FusionVector accelerometerSensitivity = { 1.0f, 1.0f, 1.0f };
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
  Eigen::Matrix<double, 4, 1> m_state;
  Eigen::Matrix<double, 3, 1> m_position_heading;
  Eigen::Matrix<double, 2, 1> m_desired_pos;
  std::optional<const og::PathGeometric*> m_path;
  std::vector<Eigen::Vector2d> m_targets;
  MoveAction<DepthCamPolicy, BaseImuPolicy, OdomPolicy> m_move;
  time_point<steady_clock> m_timestamp;
  int m_wp_idx;
  AnantaMission(std::shared_ptr<typename DepthCamPolicy::If> ci,
                std::shared_ptr<typename BaseImuPolicy::If> imu_if,
                std::shared_ptr<typename OdomPolicy::If> odom_if)
    : m_tree(args.get<float>("-t"))
    , m_iterations(0)
    , m_ci(ci)
    , m_imu_if(imu_if)
    , m_odom_if(odom_if)
    , m_camera_height(args.get<float>("-c"))
    , m_state({ 0, 0, 0, 0 })
    , m_position_heading({1, -0.2, 0})
    // clang-format off
    , m_targets{ {args.get<float>("-wx"), args.get<float>("-wy")},
                { 1.4, -1.7 },
                { 2.3, -2.2 },
                { 2.9, -2.1 }}
    , m_timestamp{steady_clock::now()}
    // clang-format on
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

    // spdlog::info("Calibrating IMU...");
    // auto calib_start = steady_clock::now();
    // Vector3f lax_sum;
    // int i = 0;
    // for (auto now = steady_clock::now(); now < calib_start + 5s;
    //      now = steady_clock::now()) {
    //   i++;
    //   Vector3f linear_accel = co_await imu_linear_acceleration(m_imu_if);
    //   lax_sum += linear_accel;
    //   Vector3f angular_vel = co_await imu_angular_velocity(m_imu_if);
    //   Vector3d magnetic_field = odom_magnetic_field(m_odom_if);
    // }
    // FusionVector initial_earth = FusionAhrsGetEarthAcceleration(&ahrs);
    // auto initial_imu_offset = lax_sum / i;
    // m_madgwick_params.accelerometerOffset.array[0] = initial_imu_offset(0);
    // m_madgwick_params.accelerometerOffset.array[1] = initial_imu_offset(1);
    // m_madgwick_params.accelerometerOffset.array[2] = - 9.882 + initial_imu_offset(2);
    // spdlog::info("IMU Offset {::2.2f}", initial_imu_offset);

    // {
    //   auto linear_accel = co_await imu_linear_acceleration(m_imu_if);
    //   auto angular_vel = co_await imu_angular_velocity(m_imu_if);
    //   auto mag_field = odom_magnetic_field(m_odom_if);
    //   fusion_update(&ahrs, &offset, linear_accel, angular_vel, mag_field);
    // }

    // auto fusion_initial_quaternion = FusionAhrsGetQuaternion(&ahrs);
    // m_initial_orientation.emplace(fusion_initial_quaternion.element.w,
    //                               fusion_initial_quaternion.element.x,
    //                               fusion_initial_quaternion.element.y,
    //                               fusion_initial_quaternion.element.z);
    // spdlog::info("Got initial orientation: {:0.2f}+{:0.2f}î+{:0.2f}ĵ+{:0.2f}̂k̂",
    //              m_initial_orientation->w(),
    //              m_initial_orientation->x(),
    //              m_initial_orientation->y(),
    //              m_initial_orientation->z());

    auto odom_vel = odometry_velocity_heading(m_odom_if);
    auto ekf = EKF3();
    while (true) {
      // auto linear_accel = co_await imu_linear_acceleration(m_imu_if);
      // auto angular_vel = co_await imu_angular_velocity(m_imu_if);
      // angular_vel *= 180.0f/M_PI;
      // spdlog::debug("Angular vel: {::5.5f}", angular_vel);
      auto mag_field = odom_magnetic_field(m_odom_if);
      auto odom_vel = odometry_velocity_heading(m_odom_if);
      // fusion_update(&ahrs, &offset, linear_accel, angular_vel, mag_field);

      // auto imu_quaternion = FusionAhrsGetQuaternion(&ahrs);
      // Eigen::Quaterniond orientation(imu_quaternion.element.w,
      //                                imu_quaternion.element.x,
      //                                imu_quaternion.element.y,
      //                                imu_quaternion.element.z);
      // const FusionVector earth = FusionAhrsGetEarthAcceleration(&ahrs);
      // auto filtered_imu_accel =
      //   Eigen::Vector3d{ earth.axis.x, earth.axis.y, earth.axis.z };

      // double imu_heading = m_initial_orientation->angularDistance(orientation);
      auto ekf_control_ip =
        Eigen::Matrix<double, 4, 1>{ m_last_target_velocity(0) * cos(odom_vel(3)),
                                     m_last_target_velocity(0) * -sin(odom_vel(3)),
                                     0, m_last_target_velocity(2) };
      ekf.predict(ekf_control_ip);
      auto ekf_measurements = Eigen::Matrix<double, 4, 1>{
        odom_vel(0), odom_vel(1), odom_vel(3), odom_vel(2)
      };
      auto estimates = ekf.correct(ekf_measurements).first;
      auto dt = duration<double>(steady_clock::now() - m_timestamp).count();

      m_state = estimates;
      auto odom_pos = odometry_position(m_odom_if);
      m_position_heading(0) = odom_pos(0) + args.get<float>("-ox");
      m_position_heading(1) = odom_pos(1) + args.get<float>("-oy");
      m_position_heading(2) = estimates(2);

      m_timestamp = steady_clock::now();

      spdlog::debug(
        "Current position: x {:2.2f} y {:2.2f} {:2.2f}° vel {:2.2f} | Odom: x {:2.2f} y "
        "{:2.2f} vel {:2.2f} {:2.2f} heading {:2.2f}°",
        m_position_heading(0),
        m_position_heading(1),
        std::fmod(m_position_heading(2) * 180 / M_PI, 360.0f),
        estimates(0),
        odom_pos(0),
        odom_pos(1),
        odom_vel(0),
        odom_vel(1),
        std::fmod(odom_vel(3) * 180 / M_PI, 360.0f));

      // position_file << estimates(0) << " " << estimates(1) << '\n';
      timer.expires_after(10ms);
      co_await timer.async_wait(use_nothrow_awaitable);
    }
  }
  auto loop() -> asio::awaitable<void>
  {
    std::vector<Eigen::Vector2d> lsq {
      {1.2, -2.5},
      {2.0, -0.6},
      {3.0, -0.5},
      {4.5, -1.2},
      {7.3, -1.0},
    };
    std::vector<Eigen::Vector2d> lcir{
      { 2.7, -0.7 },
      { 4.5, -3.0 },
      { 8.5, -0.9 },
    };
    RrtMotionPlanner rrt(std::vector<Eigen::Vector2d>(),
                         lsq,
                         std::vector<Eigen::Vector2d>(),
                         std::vector<Eigen::Vector2d>());
    rrt.plan({ 0, 0 }, { 7, -1.3 });
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
      spdlog::info("Wrote tree: {}", m_tree.write("m_tree.ot"));
      m_map_cloud.clear();

      if (m_move.utility(this) > 0) {
        m_desired_pos =
          targets[target_index] +
          Eigen::Vector2d{ args.get<float>("-ox"), args.get<float>("-oy") };
        spdlog::info("Moving to WP#{} {::2.2f}", target_index, m_desired_pos);
        co_await m_move.action(this);
      } else if (target_index < targets.size()) {
        spdlog::info(
          "Reached waypoint#{} {::2.2f}", target_index, targets[target_index]);
        target_index++;
      }

      m_iterations++;
      if (m_iterations % 10)
        spdlog::info("Wrote tree: {}", m_tree.write("m_tree.ot"));
      timer.expires_after(1000ms);
      co_await timer.async_wait(use_nothrow_awaitable);
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
    .default_value(0.5f)
    .help("Linear speed in m/s")
    .scan<'g', float>();
  args.add_argument("-ox")
    .default_value(-1.0f)
    .help("Waypoint offset in x")
    .scan<'g', float>();
  args.add_argument("-oy")
    .default_value(-0.3f)
    .help("Waypoint offset in y")
    .scan<'g', float>();
  args.add_argument("-a", "--clearance")
    .default_value(0.2f)
    .help("Minimum clearance from an obstacle (in m)")
    .scan<'g', float>();
  args.add_argument("-kp")
    .default_value(1.0f)
    .scan<'g', float>();
  args.add_argument("-kd")
    .default_value(0.04f)
    .scan<'g', float>();
  args.add_argument("-ki")
    .default_value(0.008f)
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
    dev_serial.set_option(asio::serial_port_base::baud_rate(921600));
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
