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

#ifndef TYPED_COVARIANCE_HPP_
#define TYPED_COVARIANCE_HPP_

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Eigen/Dense"
#include "fcarouge/sparse_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "mp-units/framework/quantity.h"
#include "mp-units/systems/si.h"
#include "tf2/LinearMath/Matrix3x3.hpp"

namespace fcarouge
{
/// The adapter of the typed linear algebra to mp-units elements: a quantity element is stored as
/// its numerical value in its own unit, and a stored value is read back as a quantity of that unit.
template<typename To, mp_units::Quantity From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
    return value.numerical_value_in(From::unit);
  }
};

template<mp_units::Quantity To, typename From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};
}  // namespace fcarouge

namespace robot_localization
{
namespace typed_covariance
{
template<auto Reference>
using Quantity = mp_units::quantity<Reference, double>;

/// The elements of a pose, a twist, and an acceleration: linear then angular.
template<auto Linear, auto Angular>
using Screw = std::tuple<Quantity<Linear>, Quantity<Linear>, Quantity<Linear>,
    Quantity<Angular>, Quantity<Angular>, Quantity<Angular>>;
template<auto Linear, auto Angular>
using InverseScrew = std::tuple<Quantity<mp_units::one / Linear>, Quantity<mp_units::one / Linear>,
    Quantity<mp_units::one / Linear>, Quantity<mp_units::one / Angular>,
    Quantity<mp_units::one / Angular>, Quantity<mp_units::one / Angular>>;

inline constexpr auto metre = mp_units::si::metre;
inline constexpr auto radian = mp_units::si::radian;
inline constexpr auto second = mp_units::si::second;

/// The covariance of a pose (m², m·rad, rad²) and of a twist ((m/s)², m·rad/s², (rad/s)²).
template<auto Linear, auto Angular>
using Covariance = fcarouge::typed_matrix<
  fcarouge::sparse_matrix<double, fcarouge::sparse::dense<6, 6>>, Screw<Linear, Angular>,
  Screw<Linear, Angular>>;

/// The rotation of a pose or a twist: the same rotation of the linear and the angular parts, two
/// blocks of a pattern of 18 elements.
inline constexpr fcarouge::sparse::pattern rotation_blocks{fcarouge::sparse::shape<6, 6>,
  [](std::size_t row, std::size_t column) {return (row < 3) == (column < 3);}};
template<auto Linear, auto Angular>
using Rotation = fcarouge::typed_matrix<
  fcarouge::sparse_matrix<double, rotation_blocks>, Screw<Linear, Angular>,
  InverseScrew<Linear, Angular>>;

/// The covariance of an acceleration, (m/s²)², and its rotation.
inline constexpr auto metre_per_second_squared = metre / (second * second);
using Acceleration = std::tuple<Quantity<metre_per_second_squared>,
    Quantity<metre_per_second_squared>, Quantity<metre_per_second_squared>>;
using InverseAcceleration = std::tuple<Quantity<mp_units::one / metre_per_second_squared>,
    Quantity<mp_units::one / metre_per_second_squared>,
    Quantity<mp_units::one / metre_per_second_squared>>;
using AccelerationCovariance = fcarouge::typed_matrix<
  fcarouge::sparse_matrix<double, fcarouge::sparse::dense<3, 3>>, Acceleration, Acceleration>;
using AccelerationRotation = fcarouge::typed_matrix<
  fcarouge::sparse_matrix<double, fcarouge::sparse::dense<3, 3>>, Acceleration,
  InverseAcceleration>;

/// The typed matrix of an untyped one, each element read in its own unit.
template<typename Typed>
Typed typed(const Eigen::MatrixXd & value)
{
  Typed result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (result.template at<Index / Typed::columns, Index % Typed::columns>(
      value(Index / Typed::columns, Index % Typed::columns) *
      fcarouge::tla::element_at<Typed, Index / Typed::columns,
      Index % Typed::columns>::reference), ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}

/// The untyped matrix of a typed one, each element in its own unit.
template<typename Typed>
Eigen::MatrixXd untyped(const Typed & value)
{
  Eigen::MatrixXd result(Typed::rows, Typed::columns);
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    ((result(Index / Typed::columns, Index % Typed::columns) =
    value.template at<Index / Typed::columns, Index % Typed::columns>().numerical_value_in(
      fcarouge::tla::element_at<Typed, Index / Typed::columns, Index % Typed::columns>::unit)),
    ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}

/// The covariance `covariance` of a pose (Linear = m, Angular = rad) or a twist (m/s, rad/s)
/// rotated by `rot`: R6 Σ R6ᵀ, R6 the block-diagonal rotation of both parts.
template<auto Linear, auto Angular>
Eigen::MatrixXd rotate(const Eigen::MatrixXd & covariance, const tf2::Matrix3x3 & rot)
{
  Rotation<Linear, Angular> rotation;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (rotation.template at<Index / 3, Index % 3>(rot[Index / 3][Index % 3] * mp_units::one), ...);
    (rotation.template at<Index / 3 + 3, Index % 3 + 3>(
      rot[Index / 3][Index % 3] * mp_units::one), ...);
  }(std::make_index_sequence<9>{});
  return untyped(Covariance<Linear, Angular>{rotation *
             typed<Covariance<Linear, Angular>>(covariance) * fcarouge::transposed(rotation)});
}

/// The covariance `covariance` of an acceleration rotated by `rot`: R Σ Rᵀ.
inline Eigen::MatrixXd rotate_acceleration(
  const Eigen::MatrixXd & covariance, const tf2::Matrix3x3 & rot)
{
  AccelerationRotation rotation;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (rotation.template at<Index / 3, Index % 3>(rot[Index / 3][Index % 3] * mp_units::one), ...);
  }(std::make_index_sequence<9>{});
  return untyped(AccelerationCovariance{rotation * typed<AccelerationCovariance>(covariance) *
             fcarouge::transposed(rotation)});
}
}  // namespace typed_covariance
}  // namespace robot_localization

#endif  // TYPED_COVARIANCE_HPP_
