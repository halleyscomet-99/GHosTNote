# GHosTNote ROS 2 workspace

Six packages; enactic's `openarm_description` and `openarm_can` are imported into `src/` from `ghost.repos`. Setup and run commands are in the [top-level README](../README.md).

## Packages

| Package | Type | Contents |
|---|---|---|
| `ghost_description` | data | `urdf/ghost.urdf.xacro` (args `hardware_type` sim / real, `can_interface`, `stick`, `skip_rest_check`); `scripts/export_mjcf.py` writes `mujoco/scene_*.xml` and `mujoco/ff_*.xml` from `sim/drum_sim.py` |
| `ghost_hardware` | C++ plugins | `OpenArmMITSystem` (real arm), `MujocoMITSystem` (twin) |
| `ghost_controller` | C++ plugin | `GhostImpedanceController` |
| `ghost_bringup` | launch / config | `launch/ghost.launch.xml`, `config/*.yaml`, `scripts/export_calibration.py` |
| `ghost_sim2real` | Python | `run`, `twin_viewer`, `client.ArmClient` |
| `ghost_teleop` | Python | `myo_bridge`, `imu_teleop` |

## Hardware interface

Each left-arm joint `openarm_left_jointN` exports the full Damiao MIT command, so a controller can change stiffness every cycle:

| Command interface | Meaning |
|---|---|
| `position`, `velocity` | q_des, dq_des |
| `effort` | tau_ff (feedforward torque) |
| `stiffness`, `damping` | kp, kd |

State interfaces: `position`, `velocity`, `effort` (measured torque on the real arm, applied torque in the twin). The gripper is the GPIO `openarm_left_gripper` (motor radians: `position`, `stiffness`, `damping`; state `position`, `effort`). A joint without a valid command gets pure damping. The right arm and the fingers come from mock hardware and only provide TF. ros2_control's own limit enforcement is off for these interfaces (`<limits enable="false"/>`); the controller and the motors enforce theirs.

- **OpenArmMITSystem** — params `can_interface` (can_left), `can_fd` (false), `prefix`, `max_rest_deg` (25: refuses to configure if a joint rests further from zero). Reports an error on CAN bus-off. Deactivation damps for 1.5 s, then disables the motors.
- **MujocoMITSystem** — param `model`. Applies `tau = clamp(kp (q_des - q) + kd (dq_des - dq) + tau_ff, ±peak)` every 0.5 ms physics step. The scene carries the real arm's identified Coulomb friction. Service `/ghost_left_arm/set_pad` (std_srvs/SetBool) turns pad collisions on; `run` does this once the arm is at the drumming pose.

## Controller

`ghost_controller/GhostImpedanceController`, 250 Hz.

| Topic / service | Type | |
|---|---|---|
| `/ghost_controller/trajectory` | trajectory_msgs/JointTrajectory | played back once (q, dq, ddq); must start within 0.1 rad of the current reference |
| `/ghost_controller/target` | sensor_msgs/JointState | teleop stream: low-pass, then a critically damped second-order filter limited by `vmax` / `amax` |
| `/ghost/k` | std_msgs/Float64 | stiffness k in [0, 1] |
| `/ghost_controller/gripper` | std_msgs/Float64MultiArray | [q (motor rad, NaN = hold), kp] |
| `/ghost_controller/reset` | std_srvs/Trigger | clears a safety stop; holds where the arm is |
| `/ghost_controller/command` | sensor_msgs/JointState | out, every cycle: q_des, dq_des, tau_ff per joint, then `k`, then `gripper` (measured) |
| `/ghost_controller/status` | std_msgs/String | out, 10 Hz: mode, k, fault reason |

Control law per joint: kp, kd are the real arm's defaults; on `stiff_joints` (J4, J7) kp is scaled by `kp_lo_frac + k (1 - kp_lo_frac)` and kd by its square root. tau_ff is MuJoCo inverse dynamics of the reference on `ff_model`, with the gravity part scaled by `g_scale` and Coulomb `friction · tanh(dq / friction_vel)` added, clipped to `ff_max_frac` of peak torque.

Safety stop (latched, pure damping): tracking error above `clamp(err_tau_frac · peak / kp, max_err, err_cap)`, any joint speed above `dq_base + dq_gain ·` (fastest reference speed over the last second), or measured torque above `tau_frac` of peak.

Parameters live in `ghost_bringup/config/ghost_controllers.yaml`; the launch file overlays `ff_v1_left[_stick].yaml`, the calibration (`calibration_v1_left.yaml` real, `calibration_v1_left_sim.yaml` twin) and `teleop_{true,false}.yaml` (`k_timeout`).

## Logging

`run` writes `data/robot/<time>_ros_<cmd>.csv` (columns as `robot/sim2real.py`: phase, t, qcmd_*, q_*, dq_*, tau_*, ff_*). For anything else: `ros2 bag record /joint_states /ghost_controller/command /ghost/k`.
