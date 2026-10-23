// ghost_hardware/OpenArmMITSystem: one real OpenArm v1.0 arm (7 Damiao motors + gripper) over SocketCAN.
//
// Unlike enactic's openarm_hardware it does not fix kp / kd at start-up and never moves on its own:
// stiffness / damping come from the command interfaces every cycle. A joint whose command is missing
// (NaN, e.g. before a controller is active) gets pure damping, so the arm hangs as if switched off.
// Parameters: can_interface (can_left), can_fd (false), prefix (openarm_left_),
//             max_rest_deg (25; refuse to configure if a joint rests further from zero, < 0 = no check).
#include <chrono>
#include <memory>
#include <thread>

#include "ghost_hardware/mit_common.hpp"
#include "openarm/can/socket/openarm.hpp"
#include "openarm/damiao_motor/dm_motor_constants.hpp"

namespace ghost_hardware
{
namespace dm = openarm::damiao_motor;
using CallbackReturn = hardware_interface::CallbackReturn;
using hardware_interface::return_type;

class OpenArmMITSystem : public hardware_interface::SystemInterface
{
public:
  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override
  {
    if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    const auto & info = get_hardware_info();
    can_ = param(info, "can_interface", "can_left");
    can_fd_ = param(info, "can_fd", "false") == "true";
    max_rest_ = std::stod(param(info, "max_rest_deg", "25")) * M_PI / 180.0;
    if (!check_names(info, param(info, "prefix", "openarm_left_"), joints_, gripper_, get_logger())) {
      return CallbackReturn::ERROR;
    }
    try {
      arm_ = std::make_unique<openarm::can::socket::OpenArm>(can_, can_fd_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "cannot open %s: %s (is it up? ip -br link)", can_.c_str(), e.what());
      return CallbackReturn::ERROR;
    }
    arm_->init_arm_motors(
      {dm::MotorType::DM8009, dm::MotorType::DM8009, dm::MotorType::DM4340, dm::MotorType::DM4340,
       dm::MotorType::DM4310, dm::MotorType::DM4310, dm::MotorType::DM4310},
      {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}, {0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17});
    arm_->init_gripper_motor(dm::MotorType::DM4310, 0x08, 0x18);
    arm_->set_callback_mode_all(dm::CallbackMode::STATE);
    RCLCPP_INFO(get_logger(), "OpenArm on %s (%s)", can_.c_str(), can_fd_ ? "CAN-FD" : "classic CAN");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    for (int i = 0; i < 5; ++i) {
      arm_->refresh_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      arm_->recv_all(2000);
    }
    const auto m = motors();
    for (size_t i = 0; i < m.size(); ++i) {
      if (m[i].get_state_tmos() == 0 && m[i].get_state_trotor() == 0) {
        RCLCPP_ERROR(get_logger(), "motor %zu does not reply on %s (power, IDs, CAN mode?)", i + 1, can_.c_str());
        return CallbackReturn::ERROR;
      }
    }
    for (size_t i = 0; i < N_ARM; ++i) {
      if (max_rest_ >= 0 && std::abs(m[i].get_position()) > max_rest_) {
        RCLCPP_ERROR(
          get_logger(), "%s rests at %.0f deg, not near zero: wrong arm on %s, or a joint turned by hand? "
          "(skip_rest_check:=true starts anyway)", joints_[i].c_str(), m[i].get_position() * 180 / M_PI, can_.c_str());
        return CallbackReturn::ERROR;
      }
    }
    copy_states();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    arm_->enable_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    arm_->recv_all(2000);
    RCLCPP_INFO(get_logger(), "motors enabled (pure damping until a controller commands them)");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    RCLCPP_WARN(get_logger(), "damping 1.5 s (support the arm), then motors off");
    for (int k = 0; k < 300; ++k) {
      std::vector<dm::MITParam> p;
      const auto m = arm_->get_arm().get_motors();
      for (size_t i = 0; i < N_ARM; ++i) {
        p.push_back({0.0, KD_DAMP[i], m[i].get_position(), 0.0, 0.0});
      }
      arm_->get_arm().mit_control_all(p);
      arm_->recv_all(1000);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (int i = 0; i < 3; ++i) {
      arm_->disable_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      arm_->recv_all(2000);
    }
    return CallbackReturn::SUCCESS;
  }

  return_type read(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    if (!arm_->is_bus_healthy()) {
      RCLCPP_ERROR(get_logger(), "CAN bus on %s is not healthy (bus-off?)", can_.c_str());
      return return_type::ERROR;
    }
    arm_->recv_all(0);          // replies to last cycle's commands; write() already waited for them
    copy_states();
    return return_type::OK;
  }

  return_type write(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    const auto m = arm_->get_arm().get_motors();
    std::vector<dm::MITParam> p;
    for (size_t i = 0; i < N_ARM; ++i) {
      const std::string & j = joints_[i];
      MitCommand c{get_command(j + "/position"), get_command(j + "/velocity"), get_command(j + "/effort"),
                   get_command(j + "/stiffness"), get_command(j + "/damping")};
      if (!c.valid()) {
        p.push_back({0.0, KD_IDLE[i], m[i].get_position(), 0.0, 0.0});
        continue;
      }
      const double tau = std::clamp(std::isfinite(c.tau) ? c.tau : 0.0, -TAU_PEAK[i], TAU_PEAK[i]);
      p.push_back({c.kp, c.kd, c.q, std::isfinite(c.dq) ? c.dq : 0.0, tau});
    }
    arm_->get_arm().mit_control_all(p);
    if (!gripper_.empty()) {
      const double q = get_command(gripper_ + "/position"), kp = get_command(gripper_ + "/stiffness"),
                   kd = get_command(gripper_ + "/damping");
      const double q_now = arm_->get_gripper().get_motors()[0].get_position();
      if (std::isfinite(q) && std::isfinite(kp)) {
        arm_->get_gripper().mit_control_all({{kp, std::isfinite(kd) ? kd : 0.3, q, 0.0, 0.0}});
      } else {
        arm_->get_gripper().mit_control_all({{0.0, 0.3, q_now, 0.0, 0.0}});
      }
    }
    arm_->recv_all(400);
    return return_type::OK;
  }

private:
  std::vector<dm::Motor> motors()
  {
    auto m = arm_->get_arm().get_motors();
    for (const auto & g : arm_->get_gripper().get_motors()) {
      m.push_back(g);
    }
    return m;
  }

  void copy_states()
  {
    const auto m = arm_->get_arm().get_motors();
    for (size_t i = 0; i < N_ARM; ++i) {
      set_state(joints_[i] + "/position", m[i].get_position());
      set_state(joints_[i] + "/velocity", m[i].get_velocity());
      set_state(joints_[i] + "/effort", m[i].get_torque());
    }
    if (!gripper_.empty()) {
      const auto g = arm_->get_gripper().get_motors()[0];
      set_state(gripper_ + "/position", g.get_position());
      set_state(gripper_ + "/effort", g.get_torque());
    }
  }

  std::unique_ptr<openarm::can::socket::OpenArm> arm_;
  std::string can_, gripper_;
  bool can_fd_ = false;
  double max_rest_ = 0;
  std::array<std::string, N_ARM> joints_;
};

}  // namespace ghost_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(ghost_hardware::OpenArmMITSystem, hardware_interface::SystemInterface)
