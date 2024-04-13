#include <Eigen/Dense>
#include <chrono>

using Eigen::Matrix;
using Eigen::MatrixXf;
const float deg_to_rad = 0.01745329251;

class EKF
{
public:
  // Covariance Matrix
  Matrix<float, 4, 4> m_Q;
  Matrix<float, 2, 2> m_R;

  // ip noise
  Matrix<float, 2, 2> m_ip_noise;

  // measurement matrix
  Matrix<float, 2, 4> m_H;

  // acceleration & gyro variables
  Eigen::Vector3f m_gyro_msg;
  Eigen::Vector3f m_accel_msg;
  Eigen::VectorXf m_odom_msg;
  float m_distance;
  float m_accel_net;

  EKF()
  {
    // Covariance Matrix
    m_Q << 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, (1 * deg_to_rad),
      0.0, 0.0, 0.0, 0.0, 1.0;

    m_R << 0.1, 0, 0, 0.1;

    // input noise
    m_ip_noise << 1.0, 0, 0, (30 * deg_to_rad);

    // measurement matrix
    m_H << 1, 0, 0, 0, 0, 1, 0, 0;

    // acceleration
    m_accel_net = 0.0;
  }

  // time-step
  float dt = 0.1;

  std::tuple<MatrixXf, MatrixXf> observation(MatrixXf xTrue, MatrixXf u)
  {
    xTrue = state_model(xTrue, u);

    Matrix<float, 2, 1> ud;
    ud = u + (m_ip_noise * MatrixXf::Random(2, 1));

    return std::make_tuple(xTrue, ud);
  }

  MatrixXf state_model(MatrixXf x, MatrixXf u)
  {
    Matrix<float, 4, 4> A;
    A << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;

    Matrix<float, 4, 3> B;
    B << (dt * cos(x.coeff(2, 0))), 0, u(0) + cos(x.coeff((2, 0))),
      (dt * sin(x.coeff(2, 0))), 0, u(0) + sin(x.coeff((2, 0))), 0, dt,
      u(1) / u(0), 1, 0, u(0) * dt;

    x = (A * x) + (B * u);

    return x;
  }

  MatrixXf jacob_f(MatrixXf x, MatrixXf u)
  {
    float yaw = x.coeff(2, 0);

    float v = u.coeff(0, 0);

    Matrix<float, 4, 4> jF;
    jF << 1.0, 0.0, (-dt * v * sin(yaw)), (dt * cos(yaw)), 0.0, 1.0,
      (dt * v * cos(yaw)), (dt * sin(yaw)), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      1.0;

    return jF;
  }

  MatrixXf observation_model(MatrixXf x)
  {
    Matrix<float, 2, 1> z;

    z = m_H * x;

    return z;
  }

  std::tuple<MatrixXf, MatrixXf> ekf_estimation(MatrixXf xEst,
                                                MatrixXf PEst,
                                                MatrixXf z,
                                                MatrixXf u)
  {
    // Predict
    Matrix<float, 4, 1> xPred;
    xPred = state_model(xEst, u);

    // state vector
    Matrix<float, 4, 4> jF;
    jF = jacob_f(xEst, u);

    Matrix<float, 4, 4> PPred;
    PPred = (jF * PEst * jF.transpose()) + m_Q;

    // Update
    Matrix<float, 2, 1> zPred;
    zPred = observation_model(xPred);

    Matrix<float, 2, 1> y;
    y = z - zPred; // measurement residual

    Matrix<float, 2, 2> S;
    S = (m_H * PPred * m_H.transpose()) + m_R; // Innovation Covariance

    Matrix<float, 4, 2> K;
    K = PPred * m_H.transpose() * S.inverse(); // Kalman Gain

    xEst = xPred + K * y; // update step

    PEst = (MatrixXf::Identity(4, 4) - (K * m_H)) * PPred;

    return std::make_tuple(xEst, PEst);
  }

  float complementary(float IMU_vel, float EC_vel)
  {
    float compl_vel;

    // accelerometer wt.
    float alpha = 0.98;

    // complementary velocity
    compl_vel = (alpha * IMU_vel) + (1 - alpha) * (compl_vel + IMU_vel);

    return compl_vel;
  }
};
