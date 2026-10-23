// GHOST hardware: what the real and the simulated OpenArm v1.0 arm share.
//
// Both systems take the full Damiao MIT command per joint, every cycle:
//   tau = stiffness * (position - q) + damping * (velocity - dq) + effort
// so a controller can change kp / kd as fast as it runs (EMG stiffness).
#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ghost_hardware
{

constexpr size_t N_ARM = 7;
// peak torque per motor (N m): DM8009 x2, DM4340 x2, DM4310 x3, gripper DM4310
constexpr std::array<double, N_ARM> TAU_PEAK = {40, 40, 27, 27, 7, 7, 7};
constexpr double GRIP_TAU_PEAK = 7.0;
// pure damping used when a command is missing (NaN) or while switching off
constexpr std::array<double, N_ARM> KD_IDLE = {1.0, 1.0, 0.8, 0.8, 0.3, 0.3, 0.3};
constexpr std::array<double, N_ARM> KD_DAMP = {5.0, 5.0, 4.0, 4.0, 2.0, 1.5, 1.5};

// One joint's MIT command as read from the command interfaces.
struct MitCommand
{
  double q = NAN, dq = 0, tau = 0, kp = NAN, kd = NAN;
  bool valid() const { return std::isfinite(q) && std::isfinite(kp) && std::isfinite(kd); }
};

// Joint names "<prefix>joint1..7" and the gripper GPIO "<prefix>gripper", checked against the URDF.
inline bool check_names(
  const hardware_interface::HardwareInfo & info, const std::string & prefix,
  std::array<std::string, N_ARM> & joints, std::string & gripper, const rclcpp::Logger & log)
{
  if (info.joints.size() != N_ARM) {
    RCLCPP_ERROR(log, "expected %zu joints, got %zu", N_ARM, info.joints.size());
    return false;
  }
  for (size_t i = 0; i < N_ARM; ++i) {
    joints[i] = prefix + "joint" + std::to_string(i + 1);
    if (info.joints[i].name != joints[i]) {
      RCLCPP_ERROR(log, "joint %zu is %s, expected %s", i, info.joints[i].name.c_str(), joints[i].c_str());
      return false;
    }
  }
  gripper = prefix + "gripper";
  bool has_gripper = false;
  for (const auto & g : info.gpios) {
    has_gripper |= g.name == gripper;
  }
  if (!has_gripper) {
    gripper.clear();
  }
  return true;
}

inline std::string param(
  const hardware_interface::HardwareInfo & info, const std::string & key, const std::string & dflt)
{
  auto it = info.hardware_parameters.find(key);
  return it == info.hardware_parameters.end() ? dflt : it->second;
}

}  // namespace ghost_hardware
