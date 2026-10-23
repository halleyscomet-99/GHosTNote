// ghost_controller/GhostImpedanceController: sim2real.py's control law as a ros2_control controller.
//
// Per cycle, for the 7 arm joints it writes the full Damiao MIT command
//   position = q_des, velocity = dq_des, effort = tau_ff, stiffness = kp(k), damping = kd(k)
// where
//   k in [0, 1] (topic k_topic, e.g. Myo EMG) scales kp of the stiff_joints by kp_lo_frac + k (1 - kp_lo_frac)
//     and their kd by its square root;
//   tau_ff = MuJoCo inverse dynamics of (q_des, dq_des, ddq_des) on ff_model, the gravity part scaled per joint
//     by g_scale, plus friction * tanh(dq_des / friction_vel), clipped to ff_max_frac of peak.
// The reference comes from one of
//   ~/trajectory  trajectory_msgs/JointTrajectory (q, dq, ddq at times from start), played back once; its
//                 first point must be within traj_start_tol of the current reference;
//   ~/target      sensor_msgs/JointState stream (teleop): low-passed (lowpass s), then followed by a critically
//                 damped second-order filter (stream_wn rad/s) limited to vmax and amax, so the reference has
//                 a smooth dq / ddq (no velocity steps that would fling a soft wrist).
// The gripper motor holds ~/gripper std_msgs/Float64MultiArray [q (motor rad, NaN = where it is), kp].
// Safety (latched until ~/reset): tracking error > clamp(err_tau_frac peak / kp, max_err, err_cap),
// |dq| > dq_base + dq_gain * (fastest |dq_des| of any joint over the last ~1 s; a soft wrist is thrown by
// the elbow), |tau| > tau_frac peak  ->  pure damping, reason on ~/status.
// Out: ~/command (JointState: q_des, dq_des, tau_ff per joint; "k"; "gripper" measured) every cycle,
//      ~/status (String) at 10 Hz.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "realtime_tools/realtime_thread_safe_box.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace ghost_controller
{
using CallbackReturn = controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::interface_configuration_type;
using Vec = std::vector<double>;
constexpr size_t N = 7;
constexpr const char * CMD[] = {"position", "velocity", "effort", "stiffness", "damping"};

// one monotonic clock for message ages and playback (ROS / system time types must not be mixed)
inline double mono()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Target { Vec q; double stamp = 0; };
struct Point { double t; Vec q, dq, ddq; };
struct Trajectory { std::vector<Point> pts; };
struct TrajSlot { std::shared_ptr<Trajectory> p; };
struct KValue { double k = NAN; double stamp = 0; };

class GhostImpedanceController : public controller_interface::ControllerInterface
{
public:
  ~GhostImpedanceController() override
  {
    if (dff_) mj_deleteData(dff_);
    if (mff_) mj_deleteModel(mff_);
  }

  CallbackReturn on_init() override
  {
    std::vector<std::string> j;
    for (size_t i = 1; i <= N; ++i) j.push_back("openarm_left_joint" + std::to_string(i));
    auto_declare<std::vector<std::string>>("joints", j);
    auto_declare<std::string>("gripper", "openarm_left_gripper");
    auto_declare<Vec>("kp", {70, 70, 70, 60, 10, 10, 10});
    auto_declare<Vec>("kd", {3.5, 3.0, 2.5, 2.5, 1.0, 0.8, 0.8});
    auto_declare<Vec>("kd_damp", {5.0, 5.0, 4.0, 4.0, 2.0, 1.5, 1.5});
    auto_declare<Vec>("tau_peak", {40, 40, 27, 27, 7, 7, 7});
    auto_declare<std::vector<int64_t>>("stiff_joints", {3, 6});
    auto_declare<double>("kp_lo_frac", 0.08);
    auto_declare<std::string>("k_topic", "/ghost/k");
    auto_declare<double>("k_default", 1.0);
    auto_declare<double>("k_timeout", 0.0);
    auto_declare<double>("k_min", 0.0);
    auto_declare<std::string>("ff_model", "");
    auto_declare<bool>("use_ff", true);
    auto_declare<double>("ff_max_frac", 0.4);
    auto_declare<Vec>("g_scale", Vec(N, 1.0));
    auto_declare<Vec>("friction", Vec(N, 0.0));
    auto_declare<double>("friction_vel", 0.05);
    auto_declare<bool>("friction_on_stream", false);
    auto_declare<double>("max_err", 0.3);
    auto_declare<double>("err_tau_frac", 0.25);
    auto_declare<double>("err_cap", 0.7);
    auto_declare<double>("tau_frac", 0.8);
    auto_declare<double>("dq_base", 3.0);
    auto_declare<double>("dq_gain", 1.5);
    auto_declare<Vec>("vmax", {2, 2, 2, 3, 3, 3, 3});
    auto_declare<Vec>("amax", {15, 15, 15, 30, 30, 30, 30});
    auto_declare<double>("stream_wn", 20.0);
    auto_declare<double>("lowpass", 0.04);
    auto_declare<double>("target_timeout", 0.3);
    auto_declare<double>("traj_start_tol", 0.1);
    auto_declare<double>("ramp_time", 1.5);
    auto_declare<double>("gripper_kp", 8.0);
    auto_declare<double>("gripper_kd", 0.5);
    return CallbackReturn::SUCCESS;
  }

  InterfaceConfiguration command_interface_configuration() const override
  {
    InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & j : joints_) for (const char * i : CMD) c.names.push_back(j + "/" + i);
    if (!gripper_.empty()) {
      for (const char * i : {"position", "stiffness", "damping"}) c.names.push_back(gripper_ + "/" + i);
    }
    return c;
  }

  InterfaceConfiguration state_interface_configuration() const override
  {
    InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & j : joints_) for (const char * i : {"position", "velocity", "effort"}) c.names.push_back(j + "/" + i);
    if (!gripper_.empty()) {
      c.names.push_back(gripper_ + "/position");
      c.names.push_back(gripper_ + "/effort");
    }
    return c;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    auto n = get_node();
    joints_ = n->get_parameter("joints").as_string_array();
    gripper_ = n->get_parameter("gripper").as_string();
    if (joints_.size() != N) {
      RCLCPP_ERROR(n->get_logger(), "need %zu joints", N);
      return CallbackReturn::ERROR;
    }
    for (const char * p : {"kp", "kd", "kd_damp", "tau_peak", "g_scale", "friction", "vmax", "amax"}) {
      if (n->get_parameter(p).as_double_array().size() != N) {
        RCLCPP_ERROR(n->get_logger(), "parameter %s needs %zu values", p, N);
        return CallbackReturn::ERROR;
      }
    }
    kp0_ = n->get_parameter("kp").as_double_array();
    kd0_ = n->get_parameter("kd").as_double_array();
    kd_damp_ = n->get_parameter("kd_damp").as_double_array();
    peak_ = n->get_parameter("tau_peak").as_double_array();
    g_scale_ = n->get_parameter("g_scale").as_double_array();
    friction_ = n->get_parameter("friction").as_double_array();
    vmax_ = n->get_parameter("vmax").as_double_array();
    amax_ = n->get_parameter("amax").as_double_array();
    stiff_.clear();
    const auto stiff = n->get_parameter("stiff_joints").as_integer_array();
    for (auto i : stiff) stiff_.push_back(static_cast<size_t>(i));
    for (const char * p : {"kp_lo_frac", "k_default", "k_timeout", "k_min", "ff_max_frac", "friction_vel", "max_err",
                           "err_tau_frac", "err_cap", "tau_frac", "dq_base", "dq_gain", "lowpass", "stream_wn", "target_timeout",
                           "traj_start_tol", "ramp_time", "gripper_kp", "gripper_kd"}) {
      num_[p] = n->get_parameter(p).as_double();
    }
    use_ff_ = n->get_parameter("use_ff").as_bool();
    friction_on_stream_ = n->get_parameter("friction_on_stream").as_bool();
    if (use_ff_ && !load_ff(n->get_parameter("ff_model").as_string())) return CallbackReturn::ERROR;

    traj_sub_ = n->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "~/trajectory", 10, [this](trajectory_msgs::msg::JointTrajectory::SharedPtr m) { on_trajectory(*m); });
    target_sub_ = n->create_subscription<sensor_msgs::msg::JointState>(
      "~/target", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::JointState::SharedPtr m) {
        if (m->position.size() >= N) target_.set(Target{Vec(m->position.begin(), m->position.begin() + N), mono()});
      });
    k_sub_ = n->create_subscription<std_msgs::msg::Float64>(
      n->get_parameter("k_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64::SharedPtr m) { k_.set(KValue{m->data, mono()}); });
    grip_sub_ = n->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/gripper", 10, [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
        if (!m->data.empty()) grip_cmd_.set(std::vector<double>(m->data));
      });
    reset_srv_ = n->create_service<std_srvs::srv::Trigger>(
      "~/reset", [this](const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) {
        reset_.store(true);
        r->success = true;
        r->message = "fault cleared; holding where the arm is";
      });
    cmd_pub_ = std::make_shared<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
      n->create_publisher<sensor_msgs::msg::JointState>("~/command", rclcpp::SystemDefaultsQoS()));
    status_pub_ = std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::String>>(
      n->create_publisher<std_msgs::msg::String>("~/status", 10));
    cmd_msg_.name = joints_;
    cmd_msg_.name.push_back("k");            // position = k in use
    cmd_msg_.name.push_back("gripper");      // position / effort = measured gripper motor rad / N m
    cmd_msg_.position.assign(N + 2, 0.0);
    cmd_msg_.velocity.assign(N + 2, 0.0);
    cmd_msg_.effort.assign(N + 2, 0.0);
    RCLCPP_INFO(n->get_logger(), "configured: feedforward %s, stiffness on %zu joints, k from %s",
                use_ff_ ? "ON" : "OFF", stiff_.size(), n->get_parameter("k_topic").as_string().c_str());
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    hold_here();
    fault_.clear();
    ramp_ = 0.0;
    traj_.reset();
    grip_q_ = gripper_.empty() ? NAN : state(N * 3);
    grip_kp_ = num_["gripper_kp"];
    RCLCPP_INFO(get_node()->get_logger(), "active: holding the arm where it is, gains ramping in over %.1f s",
                num_["ramp_time"]);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    for (auto & c : command_interfaces_) {      // NaN: the hardware falls back to pure damping
      std::ignore = c.set_value(std::numeric_limits<double>::quiet_NaN());
    }
    return CallbackReturn::SUCCESS;
  }

  controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    const double dt = std::max(period.seconds(), 1e-4), now = mono();
    Vec q(N), dq(N), tau(N);
    for (size_t i = 0; i < N; ++i) {
      q[i] = state(3 * i); dq[i] = state(3 * i + 1); tau[i] = state(3 * i + 2);
    }
    if (reset_.exchange(false)) {
      hold_here();
      fault_.clear();
      ramp_ = 0.0;
      traj_.reset();
    }
    // stiffness
    double k = num_["k_default"];
    if (auto kv = k_.try_get(); kv && std::isfinite(kv->k)) {
      const bool stale = num_["k_timeout"] > 0 && now - kv->stamp > num_["k_timeout"];
      if (!stale) k = kv->k;
    }
    k = std::clamp(std::max(k, num_["k_min"]), 0.0, 1.0);
    // reference
    Vec ddqd(N, 0.0);
    bool streaming = false;
    if (fault_.empty()) {
      if (auto t = pending_traj_.try_get(); t && t->p) {
        start_trajectory(t->p, now);
        pending_traj_.set(TrajSlot{});
      }
      if (traj_) {
        play(now, ddqd);
      } else {
        // stream (or, once it stops, the last streamed goal): the filter brings the reference to rest smoothly
        const auto tg = target_.try_get();
        streaming = tg && !tg->q.empty() && now - tg->stamp < num_["target_timeout"];
        const double lp = dt / (num_["lowpass"] + dt), wn = num_["stream_wn"];
        for (size_t i = 0; i < N; ++i) {
          if (streaming) qf_[i] += lp * (tg->q[i] - qf_[i]);
          const double acc = std::clamp(wn * wn * (qf_[i] - qd_[i]) - 2.0 * wn * dqd_[i], -amax_[i], amax_[i]);
          dqd_[i] = std::clamp(dqd_[i] + acc * dt, -vmax_[i], vmax_[i]);
          qd_[i] += dqd_[i] * dt;
          ddqd[i] = acc;
        }
      }
    }
    // gains
    Vec kp = kp0_, kd = kd0_;
    const double frac = num_["kp_lo_frac"] + k * (1.0 - num_["kp_lo_frac"]);
    for (size_t i : stiff_) {
      kp[i] *= frac;
      kd[i] *= std::sqrt(frac);
    }
    ramp_ = std::min(1.0, ramp_ + dt / std::max(num_["ramp_time"], 1e-3));
    const double ramp_scale = 0.3 + 0.7 * ramp_;
    for (auto & x : kp) x *= ramp_scale;
    // feedforward
    Vec ff(N, 0.0);
    if (use_ff_ && fault_.empty() && ramp_ >= 1.0) {
      feedforward(qd_, dqd_, ddqd, !streaming || friction_on_stream_, ff);
    }
    // safety
    double fastest = 0.0;
    for (double v : dqd_) fastest = std::max(fastest, std::abs(v));
    dq_peak_ = std::max(fastest, dq_peak_ * std::exp(-dt / 1.0));
    if (fault_.empty()) check(q, dq, tau, kp);
    // write
    for (size_t i = 0; i < N; ++i) {
      const bool f = !fault_.empty();
      set_cmd(5 * i + 0, f ? q[i] : qd_[i]);
      set_cmd(5 * i + 1, f ? 0.0 : dqd_[i]);
      set_cmd(5 * i + 2, f ? 0.0 : ff[i]);
      set_cmd(5 * i + 3, f ? 0.0 : kp[i]);
      set_cmd(5 * i + 4, f ? kd_damp_[i] : kd[i]);
    }
    if (!gripper_.empty()) {
      if (auto g = grip_cmd_.try_get(); g && !g->empty()) {
        grip_q_ = std::isfinite((*g)[0]) ? (*g)[0] : state(N * 3);
        if (g->size() > 1 && std::isfinite((*g)[1])) grip_kp_ = (*g)[1];
        grip_cmd_.set(std::vector<double>());
      }
      set_cmd(5 * N + 0, grip_q_);
      set_cmd(5 * N + 1, grip_kp_);
      set_cmd(5 * N + 2, num_["gripper_kd"]);
    }
    publish(time, dt, k, ff);
    return controller_interface::return_type::OK;
  }

private:
  double state(size_t i) const { return state_interfaces_[i].get_optional().value_or(NAN); }
  void set_cmd(size_t i, double v) { std::ignore = command_interfaces_[i].set_value(v); }

  void hold_here()
  {
    qd_.assign(N, 0.0); dqd_.assign(N, 0.0);
    for (size_t i = 0; i < N; ++i) qd_[i] = state(3 * i);
    qf_ = qd_;
  }

  bool load_ff(std::string path)
  {
    if (path.rfind("package://", 0) == 0) {      // package://<pkg>/<file> -> that package's share directory
      const auto slash = path.find('/', 10);
      try {
        path = ament_index_cpp::get_package_share_directory(path.substr(10, slash - 10)) + path.substr(slash);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_node()->get_logger(), "ff_model %s: %s", path.c_str(), e.what());
        return false;
      }
    }
    char err[1000] = "";
    mff_ = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    if (!mff_) {
      RCLCPP_ERROR(get_node()->get_logger(), "ff_model %s: %s", path.c_str(), err);
      return false;
    }
    dff_ = mj_makeData(mff_);
    for (size_t i = 0; i < N; ++i) {
      const int j = mj_name2id(mff_, mjOBJ_JOINT, joints_[i].c_str());
      if (j < 0) {
        RCLCPP_ERROR(get_node()->get_logger(), "%s not in %s", joints_[i].c_str(), path.c_str());
        return false;
      }
      qadr_[i] = mff_->jnt_qposadr[j];
      vadr_[i] = mff_->jnt_dofadr[j];
    }
    return true;
  }

  // MuJoCo inverse dynamics, the gravity part scaled per joint, + Coulomb friction (sim2real.Model.tau)
  void feedforward(const Vec & q, const Vec & dq, const Vec & ddq, bool friction, Vec & out)
  {
    mju_zero(dff_->qpos, mff_->nq); mju_zero(dff_->qvel, mff_->nv); mju_zero(dff_->qacc, mff_->nv);
    for (size_t i = 0; i < N; ++i) dff_->qpos[qadr_[i]] = q[i];
    mj_inverse(mff_, dff_);
    Vec g(N);
    for (size_t i = 0; i < N; ++i) g[i] = dff_->qfrc_inverse[vadr_[i]];
    for (size_t i = 0; i < N; ++i) { dff_->qvel[vadr_[i]] = dq[i]; dff_->qacc[vadr_[i]] = ddq[i]; }
    mj_inverse(mff_, dff_);
    for (size_t i = 0; i < N; ++i) {
      double t = dff_->qfrc_inverse[vadr_[i]] + (g_scale_[i] - 1.0) * g[i];
      if (friction) t += friction_[i] * std::tanh(dq[i] / num_["friction_vel"]);
      const double lim = num_["ff_max_frac"] * peak_[i];
      out[i] = std::clamp(t, -lim, lim);
    }
  }

  void on_trajectory(const trajectory_msgs::msg::JointTrajectory & m)
  {
    auto t = std::make_shared<Trajectory>();
    std::vector<size_t> col(N);
    for (size_t i = 0; i < N; ++i) {
      auto it = std::find(m.joint_names.begin(), m.joint_names.end(), joints_[i]);
      if (it == m.joint_names.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "trajectory lacks %s; ignored", joints_[i].c_str());
        return;
      }
      col[i] = static_cast<size_t>(it - m.joint_names.begin());
    }
    for (const auto & p : m.points) {
      Point pt{rclcpp::Duration(p.time_from_start).seconds(), Vec(N), Vec(N, 0.0), Vec(N, 0.0)};
      for (size_t i = 0; i < N; ++i) {
        pt.q[i] = p.positions.at(col[i]);
        if (p.velocities.size() == m.joint_names.size()) pt.dq[i] = p.velocities[col[i]];
        if (p.accelerations.size() == m.joint_names.size()) pt.ddq[i] = p.accelerations[col[i]];
      }
      t->pts.push_back(pt);
    }
    if (t->pts.empty()) return;
    pending_traj_.set(TrajSlot{t});
  }

  void start_trajectory(const std::shared_ptr<Trajectory> & t, double now)
  {
    double off = 0;
    for (size_t i = 0; i < N; ++i) off = std::max(off, std::abs(t->pts.front().q[i] - qd_[i]));
    if (off > num_["traj_start_tol"]) {
      status_note_ = "trajectory rejected: starts " + std::to_string(off) + " rad from the current reference";
      return;
    }
    traj_ = t;
    traj_t0_ = now;
    status_note_.clear();
  }

  void play(double now, Vec & ddqd)
  {
    const auto & p = traj_->pts;
    const double t = now - traj_t0_;
    size_t k = 0;
    while (k + 1 < p.size() && p[k + 1].t <= t) ++k;
    if (k + 1 >= p.size()) {
      qd_ = p.back().q;
      std::fill(dqd_.begin(), dqd_.end(), 0.0);
      qf_ = qd_;
      traj_.reset();
      return;
    }
    const double s = std::clamp((t - p[k].t) / std::max(p[k + 1].t - p[k].t, 1e-9), 0.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
      qd_[i] = p[k].q[i] + s * (p[k + 1].q[i] - p[k].q[i]);
      dqd_[i] = p[k].dq[i] + s * (p[k + 1].dq[i] - p[k].dq[i]);
      ddqd[i] = p[k].ddq[i] + s * (p[k + 1].ddq[i] - p[k].ddq[i]);
    }
    qf_ = qd_;
  }

  void check(const Vec & q, const Vec & dq, const Vec & tau, const Vec & kp)
  {
    for (size_t i = 0; i < N; ++i) {
      const double lim = std::clamp(num_["err_tau_frac"] * peak_[i] / std::max(kp[i], 1e-6), num_["max_err"], num_["err_cap"]);
      const double e = std::abs(q[i] - qd_[i]);
      const double dq_lim = num_["dq_base"] + num_["dq_gain"] * dq_peak_;
      char buf[200] = "";
      if (!std::isfinite(q[i])) {
        snprintf(buf, sizeof(buf), "no state from %s", joints_[i].c_str());
      } else if (e > lim) {
        snprintf(buf, sizeof(buf), "%s tracking error %.1f deg > %.1f (q %+.1f, target %+.1f)", joints_[i].c_str(),
                 e * 57.2958, lim * 57.2958, q[i] * 57.2958, qd_[i] * 57.2958);
      } else if (std::abs(dq[i]) > dq_lim) {
        snprintf(buf, sizeof(buf), "%s speed %.2f rad/s > %.2f", joints_[i].c_str(), dq[i], dq_lim);
      } else if (std::abs(tau[i]) > num_["tau_frac"] * peak_[i]) {
        snprintf(buf, sizeof(buf), "%s torque %.2f N m > %.0f %% of peak", joints_[i].c_str(), tau[i],
                 100 * num_["tau_frac"]);
      }
      if (buf[0]) {
        fault_ = std::string("SAFETY STOP: ") + buf + " -> damping; call ~/reset to resume";
        traj_.reset();
        RCLCPP_ERROR(get_node()->get_logger(), "%s", fault_.c_str());
        return;
      }
    }
  }

  void publish(const rclcpp::Time & time, double dt, double k, const Vec & ff)
  {
    cmd_msg_.header.stamp = time;
    for (size_t i = 0; i < N; ++i) {
      cmd_msg_.position[i] = qd_[i]; cmd_msg_.velocity[i] = dqd_[i]; cmd_msg_.effort[i] = ff[i];
    }
    cmd_msg_.position[N] = k;
    if (!gripper_.empty()) {
      cmd_msg_.position[N + 1] = state(3 * N);
      cmd_msg_.effort[N + 1] = state(3 * N + 1);
    }
    cmd_pub_->try_publish(cmd_msg_);
    status_age_ += dt;
    if (status_age_ >= 0.1) {
      status_age_ = 0.0;
      std_msgs::msg::String s;
      const char * mode = !fault_.empty() ? "FAULT" : traj_ ? "TRAJECTORY" : ramp_ < 1.0 ? "RAMP" : "HOLD/STREAM";
      s.data = std::string(mode) + " k=" + std::to_string(k).substr(0, 4) +
               (fault_.empty() ? "" : " | " + fault_) + (status_note_.empty() ? "" : " | " + status_note_);
      status_pub_->try_publish(s);
    }
  }

  std::vector<std::string> joints_;
  std::string gripper_;
  Vec kp0_, kd0_, kd_damp_, peak_, g_scale_, friction_, vmax_, amax_;
  std::vector<size_t> stiff_;
  std::map<std::string, double> num_;
  bool use_ff_ = true, friction_on_stream_ = false;
  mjModel * mff_ = nullptr;
  mjData * dff_ = nullptr;
  std::array<int, N> qadr_{}, vadr_{};

  Vec qd_, dqd_, qf_;
  double ramp_ = 0.0, grip_q_ = NAN, grip_kp_ = 8.0, dq_peak_ = 0.0;
  std::string fault_, status_note_;
  std::shared_ptr<Trajectory> traj_;
  double traj_t0_ = 0, status_age_ = 0;

  realtime_tools::RealtimeThreadSafeBox<Target> target_;
  realtime_tools::RealtimeThreadSafeBox<KValue> k_;
  realtime_tools::RealtimeThreadSafeBox<TrajSlot> pending_traj_;
  realtime_tools::RealtimeThreadSafeBox<std::vector<double>> grip_cmd_;
  std::atomic<bool> reset_{false};

  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr k_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr grip_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  std::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>> cmd_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::String>> status_pub_;
  sensor_msgs::msg::JointState cmd_msg_;
};

}  // namespace ghost_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(ghost_controller::GhostImpedanceController, controller_interface::ControllerInterface)
