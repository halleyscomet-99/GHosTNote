"""Gravity / friction check of the real arm (sim2real.py goto) -> ghost_controller parameters.

  python3 ros2_ws/src/ghost_bringup/scripts/export_calibration.py [data/robot/gravity_check_v1_left.json]

Same rule as robot/sim2real.py Model.calibrate: per joint, gravity scale = measured / model torque
(0.3x..2x) where the sign agreed and the model torque is >= 0.5 N m; Coulomb friction capped per joint.
Writes config/calibration_v1_left.yaml (real arm: gravity scale + friction) and
config/calibration_v1_left_sim.yaml (MuJoCo twin: friction only; its scene has that friction, its gravity is
the model's) next to this script's package, overlaid on ghost_controllers.yaml by ghost.launch.xml.
"""
import json
import sys
from pathlib import Path

FRIC_MAX = [2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 0.5]
GHOST = Path(__file__).resolve().parents[4]
CFG = Path(__file__).resolve().parent.parent / "config"


def main(path=GHOST / "data" / "robot" / "gravity_check_v1_left.json"):
    c = json.load(open(path))
    g_scale = [1.0] * 7
    for i, ok in enumerate(c["sign_ok"]):
        if ok and abs(c["model_g"][i]) >= 0.5:
            g_scale[i] = min(max(c["measured_tau"][i] / c["model_g"][i], 0.3), 2.0)
    fric = [min(max(f, 0.0), m) for f, m in zip(c.get("friction", [0.0] * 7), FRIC_MAX)]
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"  # noqa: E731
    head = f"# from {Path(path).name} ({c.get('time', '?')}, stick {c.get('stick')}), all signs ok: {c['all_signs_ok']}\n"
    for name, scale in (("calibration_v1_left.yaml", g_scale), ("calibration_v1_left_sim.yaml", [1.0] * 7)):
        (CFG / name).write_text(
            head + "ghost_controller:\n  ros__parameters:\n"
            f"    use_ff: {'true' if c['all_signs_ok'] or scale != g_scale else 'false'}\n"
            f"    g_scale: {fmt(scale)}\n    friction: {fmt(fric)}\n")
        print(name + ":\n" + (CFG / name).read_text())


if __name__ == "__main__":
    main(*sys.argv[1:])
