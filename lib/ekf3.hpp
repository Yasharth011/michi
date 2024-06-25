
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

using Matrix4by4f = Eigen::Matrix<double, 4, 4>;
using Matrix3f = Eigen::Matrix<double, 3, 1>;
using Matrix4f = Eigen::Matrix<double, 4, 1>;
using namespace std::chrono;

class EKF3 {
  time_point<steady_clock> m_timestamp;
  double m_dt = 0.01;
  Matrix4f m_state;
  Matrix4by4f m_covariance;
  public:
  Matrix4by4f m_const_A;
  Matrix4by4f m_const_B;
  Eigen::Matrix<double, 4, 4> m_const_H;
  private:
  Matrix4by4f m_const_Q;
  public:
  Eigen::Matrix<double, 4, 4> m_const_R;
  public:
    EKF3(double initial_vx=0, double initial_vy=0)
      : m_timestamp{steady_clock::now()}
      , m_state{initial_vx,initial_vy,0,0}
      , m_covariance(Matrix4by4f::Identity())
      // clang-format off
      , m_const_A{ { 1, 0, 0,   0 },
                   { 0, 1, 0,   0 },
                   { 0, 0, 1, m_dt},
                   { 0, 0, 0,    1}}
      , m_const_H(Matrix4by4f::Identity())
      // clang-format on
      , m_const_B(Matrix4by4f::Identity())
      , m_const_Q(Matrix4by4f::Identity() * 1)
      , m_const_R(Eigen::Matrix<double, 4, 4>::Identity() * 0.1)
    {
    }

    auto predict(Matrix4f control_input) {
      if (m_state == Matrix4f{0,0,0,0}) {
        m_state = control_input;
      }
      m_const_A(2,3) = m_dt;
      control_input(0) -= m_state(0);
      control_input(1) -= m_state(1);
      control_input(3) -= m_state(3);
      m_state = m_const_A * m_state + m_const_B * control_input;
      m_covariance = m_const_A * m_covariance * m_const_A.transpose() + m_const_Q;
      // return std::make_pair(predicted_state, predicted_cov);
    }
    auto correct(Matrix4f measurements) -> std::pair<Matrix4f, Matrix4by4f> {
      m_dt = duration<double>(steady_clock::now() - m_timestamp).count();
      m_timestamp = steady_clock::now();

      auto innovation_cov =
        m_const_H * m_covariance * m_const_H.transpose() + m_const_R;
      auto filter_gain =
        m_covariance * m_const_H.transpose() * innovation_cov.inverse();

      // std::cout << "Filter Gain\n" << filter_gain << '\n';
      auto corrected_state =
        m_state + filter_gain * (measurements - m_const_H * m_state);
      auto corrected_cov =
        m_covariance - filter_gain * innovation_cov * filter_gain.transpose();
      m_state = corrected_state;
      m_covariance = corrected_cov;
      return std::make_pair(m_state, m_covariance);
    }
};
