#include "ekf2.hpp"
#include <Fusion/Fusion.h>
#include <gz/msgs.hh>
#include <gz/transport/Node.hh>
#include <Eigen/Dense>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

Eigen::Vector3d imu_calib_lx{0,0,0}, imu_lx, imu_ang_vel, odom_position{0,0,0}, mag_field, imu_vel{0,0,0}, odom_vel;
double yaw, odom_heading, odom_ang_vel;
std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_update,
  last_update_imu, last_update_odom;
Eigen::Vector3d control_lin_vel{ 0, 0, 0 }, control_ang_vel{ 0, 0, 0 };
std::optional<Eigen::Quaternion<double>> initial_orientation;
int imu_calib_samples = 0;

int main() {

  gz::transport::Node node;
  std::string topic_cmd_vel = "/cmd_vel";
  std::string topic_imu = "/world/default/model/rover/link/base_link/sensor/imu_sensor/imu";
  std::string topic_odom = "/model/rover/odometry";
  std::string topic_magnet = "/world/default/model/rover/link/base_link/sensor/magnet/magnetometer";
  // Initialise algorithms
  FusionOffset offset;
  FusionAhrs ahrs;
  const FusionMatrix gyroscopeMisalignment = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                               0.0f, 0.0f, 0.0f, 1.0f };
  const FusionVector gyroscopeSensitivity = { 1.0f, 1.0f, 1.0f };
  const FusionVector gyroscopeOffset = { 0.0f, 0.0f, 0.0f };
  const FusionMatrix accelerometerMisalignment = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                   0.0f, 0.0f, 0.0f, 1.0f };
  const FusionVector accelerometerSensitivity = { 4.0f, 1.0f, 1.0f };
  const FusionVector accelerometerOffset = { 0.0f, 0.0f, 0.0f };
  const FusionMatrix softIronMatrix = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f };
  const FusionVector hardIronOffset = { 0.0f, 0.0f, 0.0f };

  const int SAMPLE_RATE = 100;
  FusionOffsetInitialise(&offset, SAMPLE_RATE);
  FusionAhrsInitialise(&ahrs);

  // Set AHRS algorithm settings
  const FusionAhrsSettings settings = {
    .convention = FusionConventionNwu,
    .gain = 0.5f,
    .gyroscopeRange =
      2000.0f, /* replace this with actual gyroscope range in degrees/s */
    .accelerationRejection = 10.0f,
    .magneticRejection = 10.0f,
    .recoveryTriggerPeriod = 5 * SAMPLE_RATE, /* 5 seconds */
  };
  FusionAhrsSetSettings(&ahrs, &settings);

  // Subscribe to a topic by registering a callback.

  if (node.Subscribe(
        topic_imu, std::function([&](const gz::msgs::IMU& imu) {
          imu_lx << imu.linear_acceleration().x(),
            imu.linear_acceleration().y(), imu.linear_acceleration().z();
          imu_ang_vel << imu.angular_velocity().x(), imu.angular_velocity().y(),
            imu.angular_velocity().z();
          FusionVector gyroscope = {
            float(imu_ang_vel(0)), float(imu_ang_vel(1)), float(imu_ang_vel(2))
          }; // replace this with actual gyroscope data in degrees/s
          FusionVector accelerometer = {
            float(imu_lx(0)), float(imu_lx(1)), float(imu_lx(2))
          }; // replace this with actual accelerometer data in g
          FusionVector magnetometer = {
            float(mag_field(0)), float(mag_field(1)), float(mag_field(2))
          }; // replace this with actual magnetometer data in arbitrary units

          gyroscope = FusionCalibrationInertial(gyroscope,
                                                gyroscopeMisalignment,
                                                gyroscopeSensitivity,
                                                gyroscopeOffset);
          accelerometer = FusionCalibrationInertial(accelerometer,
                                                    accelerometerMisalignment,
                                                    accelerometerSensitivity,
                                                    accelerometerOffset);
          magnetometer = FusionCalibrationMagnetic(
            magnetometer, softIronMatrix, hardIronOffset);

          // Update gyroscope offset correction algorithm
          gyroscope = FusionOffsetUpdate(&offset, gyroscope);
          auto now = std::chrono::steady_clock::now();
          double diff_time = 0.0;
          if (not last_update) diff_time = 0.01;
          else
            diff_time = std::chrono::duration<double>(
                          std::chrono::duration_cast<std::chrono::seconds>(
                            now - *last_update))
                          .count();
          last_update.emplace(now);

          FusionAhrsUpdate(
            &ahrs, gyroscope, accelerometer, magnetometer, diff_time);
          const FusionEuler euler =
            FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));
          const FusionVector earth = FusionAhrsGetEarthAcceleration(&ahrs);
          yaw = euler.angle.yaw;
          if (imu_calib_samples < 1000) {
            Eigen::Vector3d lx{ earth.axis.x, earth.axis.y, earth.axis.z };
            imu_calib_lx += lx;
            if (++imu_calib_samples == 1000) {
              imu_calib_lx = imu_calib_lx / 1000;
            }
            return;
          }
          // printf("Roll %0.1f, Pitch %0.1f, Yaw %0.1f, X %0.1f, Y %0.1f, Z"
          // "%0.1f\n",
          //                euler.angle.roll, euler.angle.pitch,
          //                euler.angle.yaw, earth.axis.x, earth.axis.y,
          //                earth.axis.z);
          double update_diff = 0.0;
          if (not last_update_imu) update_diff = 0.01;
          else update_diff = std::chrono::duration<double>(
                          std::chrono::duration_cast<std::chrono::seconds>(
                            now - *last_update_imu))
                          .count();
          if (update_diff < 0.0001) return;
          last_update_imu.emplace(now);
          Eigen::Vector3d better_lx{ earth.axis.x, earth.axis.y, earth.axis.z };
          better_lx -= imu_calib_lx;
          if (not initial_orientation) return;
          better_lx = initial_orientation->_transformVector(better_lx);
          imu_vel = imu_vel + better_lx * update_diff;
          // spdlog::info("Vel: {::02.2f} | Accel: {::02.2f} | Yaw: {:02.2f}°", imu_vel, better_lx, yaw);
        }))) {
    std::cerr << "Subscribing to topic [" << topic_imu << "]" << std::endl;
  }

  if (node.Subscribe(topic_odom, std::function([&](const gz::msgs::Odometry& odom) {
                       
    Eigen::Vector3d current_position{double(odom.pose().position().x()),double( odom.pose().position().y()),double( odom.pose().position().z())};
    double diff_time = 0.0;
    auto now = std::chrono::steady_clock::now();
    if (not last_update_odom) diff_time = 0.01;
    else  diff_time = std::chrono::duration<double>(
                          std::chrono::duration_cast<std::chrono::seconds>(
                            now - *last_update_odom))
                          .count();

    Eigen::Quaterniond odom_quaternion{ double(odom.pose().orientation().w()), 0,0,double(odom.pose().orientation().z())};
    odom_heading = odom_quaternion.angularDistance(Eigen::Quaterniond::Identity());
    odom_ang_vel = odom.twist().angular().z();
    // if (diff_time < 0.0001) return;
    last_update_odom.emplace(now);
    auto lin_vel = odom.twist().linear().x();
    odom_vel << lin_vel*cos(odom_heading), lin_vel*sin(odom_heading), 0;
    odom_position = current_position;
    // spdlog::info("Odom vel: {::2.2f}, heading: {:2.2f}°", odom_vel, odom_heading*180/M_PI);
    })))
  {
    std::cerr << "Subscribing to topic [" << topic_odom << "]" << std::endl;
  }

  if (node.Subscribe(topic_cmd_vel, std::function([&](const gz::msgs::Twist& cmd) {
                     control_lin_vel << cmd.linear().x(), cmd.linear().y(), cmd.linear().z();
                     control_ang_vel << cmd.angular().x(), cmd.angular().y(), cmd.angular().z();
                     return;
                     })))
  {
    std::cerr << "Subscribing to topic [" << topic_cmd_vel << "]" << std::endl;
  }

  if (node.Subscribe(topic_magnet, std::function([&](const gz::msgs::Magnetometer& mag) {
                       mag_field << mag.field_tesla().x(), mag.field_tesla().y(), mag.field_tesla().z();
                     })))
  {
    std::cerr << "Subscribing to topic [" << topic_magnet << "]" << std::endl;
  }

  auto ekf = EKF2();
  // Hold the rover still for a second
  std::this_thread::sleep_for(std::chrono::seconds(1));
  // Get initial orientation quaternion
  auto initial_quaternion = FusionAhrsGetQuaternion(&ahrs);
  initial_orientation.emplace(initial_quaternion.element.w,
                              initial_quaternion.element.x,
                              initial_quaternion.element.y,
                              initial_quaternion.element.z);
  while (true) {
    auto control_ip = Matrix6f{ 0, control_lin_vel(0), 0, control_lin_vel(1),
                                0, control_ang_vel(2) };
    ekf.predict(control_ip);

    auto measurements = Matrix4f{ 0.95 * odom_vel(0) + 0.05 * imu_vel(0),
                                  0.95 * odom_vel(1) + 0.05 * imu_vel(1),
                                  yaw,
                                  odom_ang_vel };
    auto estimates = ekf.correct(measurements).first;

    // measurements = Matrix4f{ odom_vel(0), odom_vel(1), odom_heading, odom_ };

    spdlog::info("Position: {}×{}", estimates(0), estimates(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
}
