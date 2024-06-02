#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include "ekf2.hpp"
#include <numeric>
#include <math.h>

int main() {
  int simulation_time = 40;
  float dt = 1;
  int steps = simulation_time / dt;
  std::vector<float> tspan(steps);
  for (int i = 0; i < steps; i++) tspan[i] = i * dt;

  int gain = 3;
  std::vector<float> Uvx(steps-1);
  std::vector<float> Uvy(steps-1);
  std::vector<float> Uvtheta(steps-1);

  for (int i = 1; i < steps; i++) {
    Uvx[i] = gain*cos(tspan[i]);
    Uvy[i] = gain*-sin(tspan[i]);
    Uvtheta[i] = gain*pow(1/cos(tspan[i]), 2);
  }
  Matrix6f initial_state{0,1,1,0,0,1};
  Eigen::Matrix<float, 6, Eigen::Dynamic> true_states(6, steps);
  true_states.col(0) = initial_state;
  EKF2 ekf;
  for (int i = 1; i < steps; i++) {
    Matrix6f u{0, Uvx[i], 0, Uvy[i], 0, Uvtheta[i]};
    true_states.col(i) = ekf.m_const_A * true_states.col(i-1) + ekf.m_const_B * u;
    // true_states.col(i) = true_states.col(i-1);
  }
  Eigen::LLT<Eigen::MatrixXf> lltOfR(ekf.m_const_R);
      Eigen::MatrixXf L = lltOfR.matrixL(); // Lower triangular matrix from Cholesky decomposition

      // Generate random noise
      std::default_random_engine generator(2024);
      std::normal_distribution<float> distribution;
      Eigen::Matrix<float, 4, Eigen::Dynamic> randn(4, steps);
      for(int i = 0; i < 4; ++i) {
          for(int j = 0; j < steps; ++j) {
              randn(i, j) = distribution(generator);
          }
      }

      // Compute Measurements
      Eigen::MatrixXf Measurements = ekf.m_const_H * true_states + L * randn;
      Eigen::MatrixXf estimates(6, steps);
      for (int i = 1; i < steps; i++) {
        auto control_ip = Matrix6f{0, Uvx[i], 0, Uvy[i], 0, Uvtheta[i]};
        ekf.predict(control_ip);
        estimates.col(i) = ekf.correct(Measurements.col(i)).first;
        // std::cout << "Estimate: " << estimates.col(i) << " Measurements: " << Measurements.col(i) << '\n';
      }

      std::ofstream estimatef("estimates.txt"), truef("trues.txt");
      for (int i = 1; i < steps; i++) {
        estimatef << estimates(0, i) << " " << estimates(2, i) << '\n';
        truef << true_states( 0,i) << " " << true_states( 2,i) << '\n';
      }
}


