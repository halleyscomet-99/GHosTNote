// ghost_hardware/MujocoMITSystem: the same MIT interface as OpenArmMITSystem, simulated in MuJoCo.
//
// Loads the scene exported from sim/drum_sim.py (both arms, stick on the left tcp, pad under its tip) and,
// every control cycle, steps the physics for one period (0.5 ms steps) applying the Damiao law per joint:
//   tau = clamp(kp (q_des - q) + kd (dq_des - dq) + tau_ff, +-peak)
// recomputed every physics step, like the motor's on-board PD. Joints this system does not own (right arm,
// fingers) are held at zero with a stiff PD + gravity, as in drum_sim.simulate.
// The pad only collides once switched on (service <component>/set_pad, std_srvs/SetBool), like the real one
// that is placed after the arm reaches the drumming pose: otherwise the 40 cm stick sweeps through it.
// Parameters: model (path to the scene XML), prefix (openarm_left_), pad_enabled (false).
#include <mujoco/mujoco.h>

#include <atomic>
#include <memory>

#include "ghost_hardware/mit_common.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace ghost_hardware
{
using CallbackReturn = hardware_interface::CallbackReturn;
using hardware_interface::return_type;

class MujocoMITSystem : public hardware_interface::SystemInterface
{
public:
  ~MujocoMITSystem() override
  {
    if (d_) mj_deleteData(d_);
    if (m_) mj_deleteModel(m_);
  }

  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override
  {
    if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    const auto & info = get_hardware_info();
    if (!check_names(info, param(info, "prefix", "openarm_left_"), joints_, gripper_, get_logger())) {
      return CallbackReturn::ERROR;
    }
    const std::string path = param(info, "model", "");
    char err[1000] = "";
    m_ = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    if (!m_) {
      RCLCPP_ERROR(get_logger(), "cannot load %s: %s", path.c_str(), err);
      return CallbackReturn::ERROR;
    }
    d_ = mj_makeData(m_);
    for (size_t i = 0; i < N_ARM; ++i) {
      const int j = mj_name2id(m_, mjOBJ_JOINT, joints_[i].c_str());
      const int a = mj_name2id(m_, mjOBJ_ACTUATOR, ("tau_" + joints_[i]).c_str());
      if (j < 0 || a < 0) {
        RCLCPP_ERROR(get_logger(), "%s or its actuator is missing from %s", joints_[i].c_str(), path.c_str());
        return CallbackReturn::ERROR;
      }
      qadr_[i] = m_->jnt_qposadr[j];
      vadr_[i] = m_->jnt_dofadr[j];
      act_[i] = a;
    }
    for (int j = 0; j < m_->njnt; ++j) {
      const std::string name = mj_id2name(m_, mjOBJ_JOINT, j);
      const int a = mj_name2id(m_, mjOBJ_ACTUATOR, ("tau_" + name).c_str());
      if (a >= 0 && std::find(joints_.begin(), joints_.end(), name) == joints_.end()) {
        others_.push_back({m_->jnt_qposadr[j], m_->jnt_dofadr[j], a});
      }
    }
    head_ = mj_name2id(m_, mjOBJ_GEOM, "head");
    if (head_ >= 0) {
      head_type_ = {m_->geom_contype[head_], m_->geom_conaffinity[head_]};
      pad_req_ = param(info, "pad_enabled", "false") == "true";
      apply_pad();
    }
    mj_forward(m_, d_);
    RCLCPP_INFO(get_logger(), "MuJoCo %s: %ld dof, dt %.4f s", path.c_str(), static_cast<long>(m_->nv), m_->opt.timestep);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    if (get_node() && head_ >= 0) {
      pad_srv_ = get_node()->create_service<std_srvs::srv::SetBool>(
        "~/set_pad", [this](const std_srvs::srv::SetBool::Request::SharedPtr q, std_srvs::srv::SetBool::Response::SharedPtr r) {
          pad_req_ = q->data;
          r->success = true;
          r->message = q->data ? "pad collides" : "pad off";
        });
    }
    copy_states();        // the state interfaces exist only after on_init
    return CallbackReturn::SUCCESS;
  }

  return_type read(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    copy_states();
    return return_type::OK;
  }

  return_type write(const rclcpp::Time &, const rclcpp::Duration & period) override
  {
    std::array<MitCommand, N_ARM> c;
    for (size_t i = 0; i < N_ARM; ++i) {
      const std::string & j = joints_[i];
      c[i] = {get_command(j + "/position"), get_command(j + "/velocity"), get_command(j + "/effort"),
              get_command(j + "/stiffness"), get_command(j + "/damping")};
    }
    apply_pad();
    const int steps = std::max(1, static_cast<int>(std::lround(period.seconds() / m_->opt.timestep)));
    for (int s = 0; s < steps; ++s) {
      for (size_t i = 0; i < N_ARM; ++i) {
        const double q = d_->qpos[qadr_[i]], dq = d_->qvel[vadr_[i]];
        double t = c[i].valid() ?
          c[i].kp * (c[i].q - q) + c[i].kd * ((std::isfinite(c[i].dq) ? c[i].dq : 0.0) - dq) +
          (std::isfinite(c[i].tau) ? c[i].tau : 0.0) :
          -KD_IDLE[i] * dq;
        tau_[i] = std::clamp(t, -TAU_PEAK[i], TAU_PEAK[i]);
        d_->ctrl[act_[i]] = tau_[i];
      }
      for (const auto & o : others_) {
        d_->ctrl[o.act] = std::clamp(
          -60.0 * d_->qpos[o.qadr] - 1.5 * d_->qvel[o.vadr] + d_->qfrc_bias[o.vadr], -20.0, 20.0);
      }
      mj_step(m_, d_);
    }
    return return_type::OK;
  }

private:
  struct Other { int qadr, vadr, act; };

  void apply_pad()
  {
    if (head_ < 0 || pad_req_ == pad_on_) return;
    pad_on_ = pad_req_;
    m_->geom_contype[head_] = pad_on_ ? head_type_.first : 0;
    m_->geom_conaffinity[head_] = pad_on_ ? head_type_.second : 0;
    RCLCPP_INFO(get_logger(), "pad %s", pad_on_ ? "on (collides)" : "off");
  }

  void copy_states()
  {
    for (size_t i = 0; i < N_ARM; ++i) {
      set_state(joints_[i] + "/position", d_->qpos[qadr_[i]]);
      set_state(joints_[i] + "/velocity", d_->qvel[vadr_[i]]);
      set_state(joints_[i] + "/effort", tau_[i]);
    }
    if (!gripper_.empty()) {      // the stick is welded to the tcp in the sim: the gripper only echoes
      const double q = get_command(gripper_ + "/position");
      set_state(gripper_ + "/position", std::isfinite(q) ? q : 0.0);
      set_state(gripper_ + "/effort", 0.0);
    }
  }

  mjModel * m_ = nullptr;
  mjData * d_ = nullptr;
  std::array<std::string, N_ARM> joints_;
  std::string gripper_;
  std::array<int, N_ARM> qadr_{}, vadr_{}, act_{};
  std::array<double, N_ARM> tau_{};
  std::vector<Other> others_;
  int head_ = -1;
  std::pair<int, int> head_type_{1, 1};
  std::atomic<bool> pad_req_{false};
  bool pad_on_ = true;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pad_srv_;
};

}  // namespace ghost_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(ghost_hardware::MujocoMITSystem, hardware_interface::SystemInterface)
