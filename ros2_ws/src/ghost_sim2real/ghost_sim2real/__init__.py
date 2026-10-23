"""GHOST sim2real on ROS 2: the robot/sim2real.py experiments, driven through ghost_controller."""
import os
import sys
from pathlib import Path


def ghost_root():
    """The GHOST repo (sim/, robot/, scripts/, data/): GHOST_ROOT, else the first parent holding sim/drum_sim.py."""
    if os.environ.get("GHOST_ROOT"):
        return Path(os.environ["GHOST_ROOT"])
    for p in Path(__file__).resolve().parents:
        if (p / "sim" / "drum_sim.py").exists():
            return p
    return Path.home() / "Github" / "GHOST"


ROOT = ghost_root()

# ROS starts console scripts with /usr/bin/python3; mujoco, pandas etc. live in the GHOST .venv. Re-run this
# process under the venv interpreter (ROS's PYTHONPATH is inherited, so rclpy keeps working).
try:
    import mujoco  # noqa: F401
except ImportError:
    _venv = ROOT / ".venv" / "bin" / "python"
    if _venv.exists() and Path(sys.prefix).resolve() != _venv.parent.parent.resolve() and not os.environ.get("GHOST_REEXEC"):
        os.environ["GHOST_REEXEC"] = "1"
        os.execv(str(_venv), [str(_venv)] + sys.argv)
    raise
for sub in ("sim", "robot", "scripts"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
