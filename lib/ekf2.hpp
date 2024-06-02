#pragma once

#include <Eigen/Dense>

using Matrix6f = Eigen::Matrix<float, 6, 1>;
using Matrix6by6f = Eigen::Matrix<float, 6, 6>;
using Matrix3f = Eigen::Matrix<float, 3, 1>;
using Matrix4f = Eigen::Matrix<float, 4, 1>;

class EKF2 {
  const float m_dt = 0.01;
  Matrix6f m_state;
  Matrix6by6f m_covariance;
  public:
  Matrix6by6f m_const_A;
  Matrix6by6f m_const_B;
  Eigen::Matrix<float, 4, 6> m_const_H;
  private:
  Matrix6by6f m_const_Q;
  public:
  Eigen::Matrix<float, 4, 4> m_const_R;
  public:
    EKF2()
      : m_state{0,1,1,0,0,1}
      , m_covariance(Matrix6by6f::Identity())
      , m_const_A{ { 1, m_dt, 0, 0, 0, 0 },    { 0, 1, 0, 0, 0, 0 },
                   { 0, 0, 1, m_dt, 0, 0 },    { 0, 0, 0, 1, 0, 0 },
                   { 0, 0, 0, 0, 1, m_dt },    { 0, 0, 0, 0, 0, 1 } }
      , m_const_H{ {0, 1, 0, 0, 0, 0},
                   {0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 1}}
      , m_const_B(Matrix6by6f::Zero())
      , m_const_Q(Matrix6by6f::Identity() * 1)
      , m_const_R(Eigen::Matrix<float, 4, 4>::Identity() * 0.1)
    {
      m_const_B(1, 1) = 0.3;
      m_const_B(3, 3) = 0.3;
      m_const_B(5, 5) = 0.3;
    }

    auto predict(Matrix6f control_input) {
      m_state = m_const_A * m_state + m_const_B * control_input;
      m_covariance = m_const_A * m_covariance * m_const_A.transpose() + m_const_Q;
      // return std::make_pair(predicted_state, predicted_cov);
    }
    auto correct(Matrix4f measurements) -> std::pair<Matrix6f, Matrix6by6f> {
      auto innovation_cov = m_const_H*m_covariance*m_const_H.transpose() + m_const_R;
      auto filter_gain = m_covariance * m_const_H.transpose() * innovation_cov.inverse();

      auto corrected_state = m_state + filter_gain * (measurements - m_const_H*m_state);
      auto corrected_cov = m_covariance - filter_gain*innovation_cov*filter_gain.transpose();
      m_state = corrected_state;
      m_covariance = corrected_cov;
      return std::make_pair(m_state, m_covariance);
    }
};
