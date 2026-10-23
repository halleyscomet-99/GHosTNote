# GHosTNote

**Animation supplies the motion, EMG supplies the stiffness.** GHosTNote drives an OpenArm v1.0 arm that plays a drum. The motion comes from a planned stroke or from the operator's arm. The joint stiffness comes from the operator's forearm muscles, read live by a Myo armband.

Everything runs through ROS 2 (Kilted) and ros2_control. The real arm and its MuJoCo twin use the same launch file and the same controller; only the hardware plugin changes.

```
Myo armbands ─► live_stiffness.py ─UDP─► myo_bridge ─► /ghost/k, /ghost/imu/*
                                                            │
run (strokes, k sweeps) / imu_teleop (whole-arm teleop) ────┤  trajectory / target stream
                                                            ▼
                 ghost_controller (250 Hz): k -> kp, kd; MuJoCo feedforward; safety stop
                                                            │  per joint: q, dq, tau_ff, kp, kd
                                 ┌──────────────────────────┴──────────────────────────┐
                 OpenArmMITSystem (real arm, SocketCAN)                MujocoMITSystem (twin)
```

## What is here

| Path | What it is |
|---|---|
| `ros2_ws/src/ghost_description` | URDF (enactic OpenArm v1.0 + full Damiao MIT interfaces per joint), MuJoCo scene / feedforward export |
| `ros2_ws/src/ghost_hardware` | ros2_control systems: the real arm (openarm_can) and a MuJoCo twin with the same interface |
| `ros2_ws/src/ghost_controller` | `GhostImpedanceController`: stiffness from k every cycle, inverse-dynamics feedforward, trajectory and teleop-stream references, latched safety stop |
| `ros2_ws/src/ghost_bringup` | `ghost.launch.xml`, controller parameters, identified calibration |
| `ros2_ws/src/ghost_sim2real` | `run goto / stroke / contact` (strokes, interleaved k sweeps, MIDI + wrist-rebound scoring, real-vs-sim plots), `twin_viewer` |
| `ros2_ws/src/ghost_teleop` | `myo_bridge` (armbands -> ROS topics), `imu_teleop` (two-Myo whole-arm teleop) |
| `sim/drum_sim.py` | MuJoCo drumming model and stroke plan (the source of every pose and trajectory) |
| `sim/ded70.py`, `sim/ded70_layout.yaml` | Donner DED-70 kit for the twin and RViz: parametric model from the product spec (four 6.5" mesh pads, three 9" rubber cymbals, tube rack; the kick pedal is left out); where each piece sits and how its face is tilted is in the layout file |
| `robot/` | Python tools the ROS nodes reuse: `sim2real.py` (control law, scoring, gravity / friction check), `arm_teleop.py` (IMU calibration + IK), `arm_bringup.py` (motor bring-up), `udev/` (fixed CAN names) |
| `scripts/` | Myo pipeline: `live_stiffness.py` (EMG -> co-contraction -> k), `calibrate.py` (MVC), `new_session.py`, `myo_scan.py` |
| `data/robot/gravity_check_v1_left.json` | Identified gravity scale and Coulomb friction of our left arm |

The ROS workspace README ([ros2_ws/README.md](ros2_ws/README.md)) lists topics, services and parameters.

## Setup

Ubuntu 24.04, ROS 2 Kilted. `GHosTNote` and enactic's `openarm_mujoco` must sit side by side:

```bash
# system (skip what you already have; run apt update first)
sudo apt update
sudo apt install ros-kilted-desktop ros-kilted-ros2-control ros-kilted-ros2-controllers ros-kilted-xacro \
                 python3-colcon-common-extensions python3-vcstool libcli11-dev cmake ninja-build

# code
cd ~/Github
git clone git@github.com:halleyscomet-99/GHosTNote.git
git clone https://github.com/enactic/openarm_mujoco.git
cd GHosTNote/ros2_ws && vcs import src < ghost.repos && cd ..

# Python environment: create it BEFORE building (the C++ plugins link its MuJoCo)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# MuJoCo models (once per machine) and controller calibration
.venv/bin/python ros2_ws/src/ghost_description/scripts/export_mjcf.py
python3 ros2_ws/src/ghost_bringup/scripts/export_calibration.py

# build
cd ros2_ws
source /opt/ros/kilted/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

Every new terminal: `source /opt/ros/kilted/setup.bash && source ~/Github/GHosTNote/ros2_ws/install/setup.bash`. The Python nodes switch to `.venv` by themselves.

## Run

### 1. Simulation (no hardware)

```bash
ros2 launch ghost_bringup ghost.launch.xml stick:=true viewer:=true            # terminal A
ros2 run ghost_sim2real run goto                                                # terminal B: to the drumming pose, Enter, back
ros2 run ghost_sim2real run stroke --slow 4                                     # one slow stroke + real-vs-sim plot
ros2 run ghost_sim2real run stroke --slow 1 --ks 1,0.5,0.2,0.08 --n 3 --period 2  # interleaved stiffness sweep onto the pad
```

The twin viewer draws the arm as measured, with the controller's reference as a translucent blue arm, playing the DED-70 kit; RViz shows the same kit (`/ghost/kit` markers). The kit only collides once the arm is at the drumming pose (`run` switches it on through `/ghost_left_arm/set_pad`). Logs and plots go to `data/robot/`.

### 2. Real arm

One-time:
1. Motors: IDs 0x01–0x08 (replies 0x11–0x18), classic CAN 1 Mbit/s, zero position set with the arm hanging (openarm_can `setup/` scripts).
2. CAN names: put your adapters' USB serials into `robot/udev/80-openarm-can.rules` (`udevadm info -a -p /sys/class/net/can0 | grep -m1 serial`), then `sudo cp robot/udev/80-openarm-can.rules /etc/udev/rules.d/ && sudo udevadm control --reload` and re-plug. Without the rule: `sudo ip link set can0 up type can bitrate 1000000`.
3. Gravity / friction check of your arm (replaces ours): build the openarm_can Python binding (`source .venv/bin/activate && bash ros2_ws/src/openarm_can/python/build.sh`), run `python robot/sim2real.py goto --can can_left`, then rerun the two export commands above.

Each session:
```bash
python robot/arm_bringup.py read --can can_left --no-fd                         # 8 motors reply, joints near 0 deg
ros2 launch ghost_bringup ghost.launch.xml hardware_type:=real can:=can_left viewer:=true
ros2 run ghost_sim2real run goto --frac 0.3                                     # first time: no stick, hand on the arm
ros2 run ghost_sim2real run stroke --slow 1 --ks 1,0.5,0.2,0.08 --n 8 --aim 1 --midi --stick   # relaunch with stick:=true
```
Place the pad with `ros2 run ghost_sim2real run contact --stick`. A safety stop damps the arm and is reported on `/ghost_controller/status`; resume with `ros2 service call /ghost_controller/reset std_srvs/srv/Trigger`.

### 3. Teleop with Myo armbands

Forearm Myo (EMG + IMU) and upper-arm Myo (IMU), each on its own BLED112 dongle:
```bash
python scripts/myo_scan.py                                                      # which Myo each dongle sees
export MYO_TTY=/dev/ttyACM1 MYO_MAC=<forearm MAC> MYO2_TTY=/dev/ttyACM0 MYO2_MAC=<upper-arm MAC>
cd scripts && python new_session.py && python calibrate.py --session <session>  # MVC, once per wearing
python live_stiffness.py --session <session> --udp 127.0.0.1:9100 --myo2       # window: relaxed grip L, hard grip H
```
Then, in the ROS terminals:
```bash
ros2 launch ghost_bringup ghost.launch.xml teleop:=true [hardware_type:=real]
ros2 run ghost_teleop myo_bridge
ros2 run ghost_teleop imu_teleop                       # whole arm: two held poses, then the arm follows you
ros2 run ghost_sim2real run stroke --teleop --n 20 --slow 1 --period 2   # or: planned strokes, stiffness from your grip
```
`teleop:=true` makes stale EMG (0.5 s) fall back to the stiff default.

## Changing the kit layout

Everything about the kit's placement is in `sim/ded70_layout.yaml` (robot frame: origin at the robot base, x toward the kit, y to the robot's left, z up; metres and degrees):

| To change | Edit |
|---|---|
| where the snare is (= robot vs kit) | `snare.position`: `auto` puts it under the stick tip at the drumming pose; `[x, y, z]` is a measured position of the head's top centre |
| how a face is angled | `tilt_deg` (lean toward the robot) and `rotate_deg` (turns that lean about the vertical) of the snare, any pad or cymbal |
| the whole kit's heading | `kit_rotate_deg` (about the snare) |
| a tom or cymbal | its `offset` from the snare, in the kit frame |
| rack, floor | `rack`, `floor_z`, `robot_stand` |

Then regenerate and relaunch (no rebuild needed):
```bash
.venv/bin/python ros2_ws/src/ghost_description/scripts/export_mjcf.py
ros2 launch ghost_bringup ghost.launch.xml stick:=true viewer:=true      # MuJoCo + RViz show the new layout
```
The export prints where the planned stroke lands on the snare (e.g. `stroke lands 4.9 cm from its centre`) and warns when it misses the 6.5" head. The stroke itself comes from the drumming pose in `sim/drum_sim.py` (`POSE`, `stroke_plan`), so a snare moved far from `auto` also needs that pose changed.

## Results so far

- Real arm (Python path, `robot/sim2real.py`): tracking at full stroke speed 0.6–0.7 deg (elbow) and 1.1 deg (wrist). On the pad, the wrist rebound falls monotonically with stiffness: +1.26, +1.02, +0.68, +0.41 rad/s at k = 0.08, 0.2, 0.5, 1 (8 interleaved strokes each, sd <= 0.08). MIDI velocity does not change with k.
- ROS 2 twin: 250 Hz, stroke tracking 0.05 deg (elbow) and 0.44 deg (wrist); k sweeps, Mode 1 and whole-arm teleop run end to end.

## Known limits

- The ROS 2 real-arm plugin has not run on the robot yet (the Python path has).
- The twin's pad contact is not identified: its rebound (~3.5 rad/s) does not depend on k yet.
- No CAD of the DED-70 exists, so `sim/ded70.py` is built from the spec: pad and cymbal sizes are real, their positions are a typical layout until you measure yours (see "Changing the kit layout").
- The Myo cannot see the wrist; in whole-arm teleop J6 / J7 hold `--wrist`.
- Real-time scheduling is off unless your user may use FIFO priority (ros2_control prints a warning).
