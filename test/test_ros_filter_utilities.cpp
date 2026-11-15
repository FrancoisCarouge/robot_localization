/*
 * Copyright (c) 2026, Charles River Analytics, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "robot_localization/ros_filter_utilities.hpp"

namespace robot_localization
{
namespace ros_filter_utilities
{

// A differential pose source turns two poses dt apart into a velocity, (p - p_previous) / dt:
// its variance is (σ² + σ²_previous) / dt², in (m/s)² and (rad/s)².
TEST(DifferentialTwistCovarianceTest, DividesBySquaredTimeDelta)
{
  Eigen::MatrixXd current = Eigen::MatrixXd::Identity(6, 6) * 2.0;
  current(0, 1) = current(1, 0) = 0.5;
  current(5, 5) = 0.2;
  const Eigen::MatrixXd previous = Eigen::MatrixXd::Identity(6, 6) * 2.0;

  const Eigen::MatrixXd twist = differentialTwistCovariance(current, previous, 0.1);

  EXPECT_DOUBLE_EQ(twist(0, 0), 400.0);
  EXPECT_DOUBLE_EQ(twist(0, 1), 50.0);
  EXPECT_DOUBLE_EQ(twist(1, 0), 50.0);
  EXPECT_DOUBLE_EQ(twist(5, 5), 220.0);
  EXPECT_DOUBLE_EQ(twist(2, 3), 0.0);
}

// Poses twice as far apart in time give a velocity four times as certain.
TEST(DifferentialTwistCovarianceTest, ScalesWithInverseSquaredTimeDelta)
{
  const Eigen::MatrixXd pose = Eigen::MatrixXd::Identity(6, 6);

  const Eigen::MatrixXd near = differentialTwistCovariance(pose, pose, 0.05);
  const Eigen::MatrixXd far = differentialTwistCovariance(pose, pose, 0.1);

  for (Eigen::Index i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(near(i, i), 4.0 * far(i, i));
  }
}

}  // namespace ros_filter_utilities
}  // namespace robot_localization
