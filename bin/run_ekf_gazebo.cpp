#include "ekf2.hpp"
#include <Fusion/Fusion.h>
#include <gz/msgs.hh>
#include <gz/transport/Node.hh>
#include <Eigen/Dense>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

Eigen::Vector3f imu_lx, imu_ang_vel, odom_position{0,0,0}, mag_field, imu_vel{0,0,0}, odom_vel;
float yaw, odom_heading, odom_ang_vel;
std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_update, last_update_odom; 
Eigen::Vector3f control_lin_vel{0,0,0}, control_ang_vel{0,0,0};
std::optional<Eigen::Quaternion<float>> initial_orientation;

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
            imu_ang_vel(0), imu_ang_vel(1), imu_ang_vel(2)
          }; // replace this with actual gyroscope data in degrees/s
          FusionVector accelerometer = {
            imu_lx(0), imu_lx(1), imu_lx(2)
          }; // replace this with actual accelerometer data in g
          FusionVector magnetometer = {
            mag_field(0), mag_field(1), mag_field(2)
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
          float diff_time = 0.0;
          if (not last_update) diff_time = 0.01;
          else
            diff_time = std::chrono::duration<float>(
                          std::chrono::duration_cast<std::chrono::seconds>(
                            now - *last_update))
                          .count();
          last_update.emplace(now);

          FusionAhrsUpdate(
            &ahrs, gyroscope, accelerometer, magnetometer, diff_time);
          const FusionEuler euler =
            FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));
          const FusionVector earth = FusionAhrsGetEarthAcceleration(&ahrs);
          // printf("Roll %0.1f, Pitch %0.1f, Yaw %0.1f, X %0.1f, Y %0.1f, Z"
          // "%0.1f\n",
          //                euler.angle.roll, euler.angle.pitch,
          //                euler.angle.yaw, earth.axis.x, earth.axis.y,
          //                earth.axis.z);
          Eigen::Vector3f better_lx{ earth.axis.x, earth.axis.y, earth.axis.z };
          if (not initial_orientation) return;
          better_lx = initial_orientation->_transformVector(better_lx);
          imu_vel = imu_vel + better_lx * 0.1;
          yaw = euler.angle.yaw;
          // spdlog::info("Vel: {} | Accel: {} | Yaw: {}°", better_lx*0.1, better_lx, yaw);
        }))) {
    std::cerr << "Subscribing to topic [" << topic_imu << "]" << std::endl;
  }

  if (node.Subscribe(topic_odom, std::function([&](const gz::msgs::Odometry& odom) {
                       
    Eigen::Vector3f current_position{float(odom.pose().position().x()),float( odom.pose().position().y()),float( odom.pose().position().z())};
    float diff_time = 0.0;
    auto now = std::chrono::steady_clock::now();
    if (not last_update_odom) diff_time = 0.01;
    else  diff_time = std::chrono::duration<float>(
                          std::chrono::duration_cast<std::chrono::seconds>(
                            now - *last_update_odom))
                          .count();
    last_update_odom.emplace(now);

    Eigen::Quaternionf odom_quaternion{ float(odom.pose().orientation().w()), 0,0,float(odom.pose().orientation().z())};
    odom_heading = odom_quaternion.angularDistance(Eigen::Quaternionf::Identity()) * 360/M_PI;
    odom_vel = (current_position - odom_position) / diff_time;
    odom_position = current_position;
    odom_ang_vel = odom.twist().angular().z();
    spdlog::info("Odom vel: {}", odom_vel);
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

    auto measurements = Matrix4f{ imu_vel(0), imu_vel(1), yaw, imu_ang_vel(2)};
    auto estimates = ekf.correct(measurements).first;

    // measurements = Matrix4f{ odom_vel(0), odom_vel(1), odom_heading, odom_ };

    spdlog::info("Position: {}×{}", estimates(0), estimates(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}
