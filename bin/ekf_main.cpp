#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cmath>
#include <stack>
#include <stdlib.h>
#include <thread>
#include <tuple>
#include <gz/msgs.hh>
#include <gz/transport/Node.hh>

using namespace std;
using namespace Eigen;

const float deg_to_rad = 0.01745329251;

class EKF {
public:

  //Covariance Matrix
  Matrix<float, 4, 4> Q;
  Matrix<float, 2, 2> R;

  //ip noise
  Matrix<float, 3, 3> ip_noise;
  
  //measurement matrix	
  Matrix<float, 2, 4> H;

  // acceleration & gyro variables
  Eigen::Vector3f gyro_msg;
  Eigen::Vector3f accel_msg;
  Eigen::Vector3f odom_msg;
  float distance;
  float accel_net;

  //time-step 
  float dt;

  EKF() {
    // Covariance Matrix
    Q << 0.1, 0.0, 0.0, 0.0, 
	 0.0, 0.1, 0.0, 0.0, 
	 0.0, 0.0, (1*deg_to_rad), 0.0, 
	 0.0, 0.0, 0.0, 1.0;

    R << 0.1, 0, 
	 0, 0.1;

    // input noise
    ip_noise << 1.0, 0.0, 0.0,
                0.0, (30*deg_to_rad), 0.0,
                0.0, 0.0, 1.0;

    // measurement matrix
    H << 1, 0, 0, 0, 
         0, 1, 0, 0;
    
    // acceleration
    accel_net = 0.0;

    //time-step
    dt = 0.1;
  }
  
  tuple<MatrixXf, MatrixXf> observation(MatrixXf xTrue, MatrixXf u) {
    xTrue = state_model(xTrue, u);

    Matrix<float, 3, 1> ud;
    ud = u + (ip_noise * MatrixXf::Random(3, 1));

    return make_tuple(xTrue, ud);
  }

  MatrixXf state_model(MatrixXf x, MatrixXf u) {
    Matrix<float, 4, 4> A;
    A << 1, 0, 0, 0, 
         0, 1, 0, 0, 
         0, 0, 1, 0, 
         0, 0, 0, 0;

    Matrix<float, 4, 3> B;
    B << (dt * cos(x.coeff(2, 0))), 0, u(0)+cos(x.coeff((2,0))), 
         (dt * sin(x.coeff(2, 0))), 0, u(0)+sin(x.coeff((2,0))),
                                 0,dt, u(1)/u(0),
                                 1, 0, u(0)*dt;

    x = (A * x) + (B * u);

    return x;
  }

  MatrixXf jacob_f(MatrixXf x, MatrixXf u) {
    float yaw = x.coeff(2, 0);

    float v = u.coeff(0, 0);

    Matrix<float, 4, 4> jF;
    jF << 1.0, 0.0, (-dt * v * sin(yaw)), (dt * cos(yaw)), 
	  0.0, 1.0, (dt * v * cos(yaw)), (dt * sin(yaw)), 
          0.0, 0.0, 1.0, 0.0, 
          0.0, 0.0, 0.0, 1.0;

    return jF;
  }

  MatrixXf observation_model(MatrixXf x) {
    Matrix<float, 2, 1> z;

    z = H * x;

    return z;
  }

  tuple<MatrixXf, MatrixXf> ekf_estimation(MatrixXf xEst, MatrixXf PEst,
                                           MatrixXf z, MatrixXf u) {
    // Predict
    Matrix<float, 4, 1> xPred;
    xPred = state_model(xEst, u);

    // state vector
    Matrix<float, 4, 4> jF;
    jF = jacob_f(xEst, u);

    Matrix<float, 4, 4> PPred;
    PPred = (jF * PEst * jF.transpose()) + Q;

    // Update
    Matrix<float, 2, 1> zPred;
    zPred = observation_model(xPred);

    Matrix<float, 2, 1> y;
    y = z - zPred; // measurement residual

    Matrix<float, 2, 2> S;
    S = (H * PPred * H.transpose()) + R; // Innovation Covariance

    Matrix<float, 4, 2> K;
    K = PPred * H.transpose() * S.inverse(); // Kalman Gain

    xEst = xPred + K * y; // update step

    PEst = (MatrixXf::Identity(4, 4) - (K * H)) * PPred;

    return make_tuple(xEst, PEst);
  }

  float complementary(float IMU_vel, float EC_vel){
    
    float compl_vel;

    //accelerometer wt.
    float alpha = 0.80;

    //complementary velocity 
    compl_vel = (alpha * IMU_vel) + (1 - alpha)*(compl_vel + IMU_vel);
    
    return compl_vel;
  }

  void IMU_cb(const gz::msgs::IMU &imu) {

    accel_msg << imu.linear_acceleration().x(), imu.linear_acceleration().y(),
        imu.linear_acceleration().z();

    gyro_msg << imu.angular_velocity().x(), imu.angular_velocity().y(),
        imu.angular_velocity().z();

    accel_net = accel_net + accel_msg.norm();
  }

  void Odometry_cb(const gz::msgs::Odometry &odom) {

    odom_msg << odom.pose().position().x(), odom.pose().position().y(),
        odom.pose().position().z();

    distance = odom_msg.norm();
  }
};

int main() {

  EKF obj;

  gz::transport::Node node;
  std::string topic_imu = "/world/default/model/rover/link/base_link/sensor/imu_sensor/imu";
  std::string topic_odom = "/model/rover/odometry";

  // Subscribe to a topic by registering a callback.

  if (node.Subscribe(topic_imu, &EKF::IMU_cb, &obj))
  {
    std::cerr << "Subscribing to topic [" << topic_imu << "]" << endl;
  }

  if (node.Subscribe(topic_odom, &EKF::Odometry_cb, &obj))
  {
    std::cerr << "Subscribing to topic [" << topic_odom << "]" << endl;
  }

  //velocity variables
  float imu_vel = 0.0, odom_vel = 0.0, vel = 0.0; 
	
  bool print_to_cout = true;

  // state vector
  Matrix<float, 4, 1> xEst = MatrixXf::Zero(4, 1);
  Matrix<float, 4, 1> xTrue = MatrixXf::Zero(4, 1);
  
  // Predicted Covariance
  Matrix<float, 4, 4> PEst = MatrixXf::Identity(4, 4);
  
  // control input
  Matrix<float, 3, 1> u;
  Matrix<float, 3, 1> ud = MatrixXf::Zero(3, 1);
  
  // observation vector 
  Matrix<float, 2, 1> z = MatrixXf::Zero(2, 1);

  // history
  Matrix<float, 4, 1> hxEst = MatrixXf::Zero(4, 1);
  Matrix<float, 4, 1> hxTrue = MatrixXf::Zero(4, 1);

  while (true) {
   
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // calculating IMU velocity 
    imu_vel = obj.accel_net * obj.dt;

    // calculating encoder veclotiy 
    odom_vel = obj.distance/obj.dt;

    vel = obj.complementary(imu_vel, odom_vel);

    // control input
    u << vel, obj.gyro_msg.y(), obj.distance;
    
    float time = time + obj.dt;

    tie(xTrue, ud) = obj.observation(xTrue, u);

    z = obj.observation_model(xTrue);

    tie(xEst, PEst) = obj.ekf_estimation(xEst, PEst, z, ud);

    // store datat history
    hxEst = xEst;
    hxTrue = xTrue;
  
    //visualisation
    if (print_to_cout) {
      
      // synchronising with python visualisation  
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      
      //Estimation + True
      cout << hxEst(0) << " " << hxEst(1) << " " 
           << hxTrue(0) << " " << hxTrue(1) << endl;
    }
  }
  return 0;
}
