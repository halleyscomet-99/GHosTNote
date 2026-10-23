"""Export the MuJoCo models the ROS 2 stack loads, built by sim/drum_sim.py so ROS and the Python sim match.

  .venv/bin/python ros2_ws/src/ghost_description/scripts/export_mjcf.py

Writes mujoco/<name>.xml (meshes stay in openarm_mujoco, referenced by absolute path):
  scene_v1_left[_stick].xml   physics for the sim hardware: both arms, pad under the left tip, stick geom;
                              the left arm's joints get the real arm's identified Coulomb friction
                              (data/robot/gravity_check_v1_left.json) and almost no viscous damping, so the
                              twin needs the same feedforward as the real arm
  ff_v1_left[_stick].xml      feedforward for ghost_controller: no contacts, no joint limits, no damping or
                              frictionloss (rigid-body gravity + inertia only, exactly sim2real.Model)
"""
import json
import sys
from pathlib import Path

import mujoco

GHOST = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GHOST / "sim"))
import drum_sim as ds  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "mujoco"


CHECK = GHOST / "data" / "robot" / "gravity_check_v1_left.json"
FRIC_MAX = [2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 0.5]     # as robot/sim2real.py
DAMPING = 0.05                                     # N m s / rad, instead of the MJCF's guessed 0.9-1.0


def main(model="v1", side="left"):
    OUT.mkdir(exist_ok=True)
    fric = [0.0] * 7
    if CHECK.exists():
        fric = [min(max(f, 0.0), m) for f, m in zip(json.load(open(CHECK)).get("friction", fric), FRIC_MAX)]
    print("left-arm friction (N m):", [round(f, 3) for f in fric])
    stick_m = ds.STICK_M
    ds.use_model(model, side)
    for stick in (False, True):
        ds.STICK_M = stick_m if stick else 1e-4
        tag = f"{model}_{side}" + ("_stick" if stick else "")
        # pad under the tip at the contact pose of THIS arm: drum_sim's poses are right-arm angles, the left
        # arm is their mirror image (robot/sim2real.py MIRROR)
        contact = ds.POSE.copy()
        contact[ds.WRIST] = ds.WRIST_DOWN * 0.20
        if side == "left":
            contact *= [-1, -1, -1, 1, -1, -1, -1]
        scene = ds.build(contact_q=contact, as_spec=True)
        if not stick:            # drum_sim always adds the stick geom; without a stick it must not collide
            g = scene.geom("stick")
            g.contype, g.conaffinity, g.rgba = 0, 0, [0, 0, 0, 0]
        for i in range(7):
            j = scene.joint(f"openarm_{side}_joint{i + 1}")
            j.damping, j.frictionloss = [DAMPING, 0.0, 0.0], fric[i]
        ff = ds.build(pad_raise=-1.0, as_spec=True)
        ff.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        for j in ff.joints:
            j.damping, j.frictionloss = [0.0, 0.0, 0.0], 0.0
        for name, spec in (("scene", scene), ("ff", ff)):
            spec.meshdir = str(ds.MJCF.parent / "meshes")
            path = OUT / f"{name}_{tag}.xml"
            path.write_text(spec.to_xml())
            m = mujoco.MjModel.from_xml_path(str(path))         # round trip: loads from anywhere
            print(f"{path.name}: nq {m.nq}, nu {m.nu}, dt {m.opt.timestep}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
