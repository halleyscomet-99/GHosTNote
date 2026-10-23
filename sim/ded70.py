"""Donner DED-70 electronic drum kit as MuJoCo primitives, placed around the snare the robot plays.

No CAD of the DED-70 is published, so this is a parametric model from the product spec: four 6.5" mesh pads
(snare + three toms), three 9" rubber cymbals (hi-hat, crash, ride), a kick pad, and a tube rack. Positions
relative to the snare are a typical kids' kit layout seen from the drummer (the robot faces +x, its left is
+y); measure your kit and edit LAYOUT to match it. Only the snare's head ("head", body "pad") is scored;
every other pad, cymbal and tube collides with the stick too.

  spec = mujoco.MjSpec.from_file(...);  add_ded70(spec, snare_top=[x, y, z], head_solref=[-3e4, -8])
"""
import numpy as np
import mujoco

INCH = 0.0254
PAD_R = 6.5 * INCH / 2            # mesh pads
CYM_R = 9.0 * INCH / 2            # rubber cymbals
TUBE_R = 0.0125                   # rack tubes, 25 mm

# offsets of each pad / cymbal centre from the snare head's top centre (m), and its tilt toward the drummer (deg)
LAYOUT = {
    "tom1":   dict(kind="pad", off=[0.18, 0.08, 0.12], tilt=15),
    "tom2":   dict(kind="pad", off=[0.18, -0.18, 0.12], tilt=15),
    "tom3":   dict(kind="pad", off=[0.00, -0.42, -0.03], tilt=5),
    "hihat":  dict(kind="cymbal", off=[0.02, 0.28, 0.10], tilt=5),
    "crash":  dict(kind="cymbal", off=[0.24, 0.30, 0.32], tilt=15),
    "ride":   dict(kind="cymbal", off=[0.20, -0.50, 0.26], tilt=10),
}
RACK_X = 0.30                        # posts and front bar this far behind the snare (m)
RACK_Y = (0.42, -0.64)               # left / right post, relative to the snare
BAR_DZ = 0.0                         # front bar height relative to the snare head

BLACK = [0.08, 0.08, 0.09, 1]
SHELL = [0.16, 0.16, 0.18, 1]
MESH = [0.78, 0.78, 0.76, 1]
RUBBER = [0.22, 0.22, 0.24, 1]


def _tilt_quat(deg):
    """Rotate about y so the pad's top faces the drummer (-x)."""
    a = np.radians(deg) / 2
    return [np.cos(a), 0.0, -np.sin(a), 0.0]


def _tube(body, a, b, name=None, rgba=BLACK):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[TUBE_R, 0, 0],
                  fromto=[*a, *b], rgba=rgba)


def _pad(body, name, head_solref, solimp):
    """A 6.5" mesh pad: collidable head on top of a shell, rubber rim (visual)."""
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R, 0.01, 0], pos=[0, 0, -0.01],
                  rgba=MESH, solref=head_solref, solimp=solimp)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R + 0.012, 0.022, 0], pos=[0, 0, -0.034],
                  rgba=SHELL, contype=0, conaffinity=0)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R + 0.014, 0.004, 0], pos=[0, 0, -0.004],
                  rgba=BLACK, contype=0, conaffinity=0)


def add_ded70(spec, snare_top, head_solref=(-3.0e4, -8.0), floor_z=-0.5):
    """Add the kit to spec.worldbody with the snare head's top centre at snare_top. Returns the snare body."""
    solimp = [0.95, 0.99, 0.001, 0.5, 2]
    s = np.asarray(snare_top, float)
    wb = spec.worldbody
    snare = wb.add_body(name="pad", pos=s.tolist())
    _pad(snare, "head", list(head_solref), solimp)

    rack = wb.add_body(name="ded70_rack", pos=[0, 0, 0])
    bar_z = s[2] + BAR_DZ
    posts = [np.array([s[0] + RACK_X, s[1] + y, 0.0]) for y in RACK_Y]
    for i, p in enumerate(posts):
        _tube(rack, [p[0], p[1], floor_z], [p[0], p[1], bar_z + 0.12], name=f"ded70_post{i}")
        _tube(rack, [p[0] - 0.18, p[1], floor_z + 0.02], [p[0] + 0.18, p[1], floor_z + 0.02])     # feet
    _tube(rack, [posts[0][0], posts[0][1], bar_z], [posts[1][0], posts[1][1], bar_z], name="ded70_bar")
    _tube(rack, [s[0] + RACK_X, s[1], bar_z], [s[0] + 0.05, s[1], s[2] - 0.05])                   # snare arm
    rack.add_geom(name="ded70_module", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.09, 0.06],
                  pos=[posts[0][0] - 0.05, posts[0][1], bar_z + 0.05], rgba=BLACK)

    for name, it in LAYOUT.items():
        c = s + np.asarray(it["off"], float)
        b = wb.add_body(name=f"ded70_{name}", pos=c.tolist(), quat=_tilt_quat(it["tilt"]))
        if it["kind"] == "pad":
            _pad(b, f"ded70_{name}_head", list(head_solref), solimp)
            _tube(rack, [s[0] + RACK_X, c[1], bar_z], [c[0] + 0.03, c[1], c[2] - 0.05])
        else:
            b.add_geom(name=f"ded70_{name}_cymbal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CYM_R, 0.003, 0],
                       rgba=RUBBER, solref=[-1.0e4, -5.0], solimp=solimp)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.008, 0], pos=[0, 0, 0.008],
                       rgba=BLACK, contype=0, conaffinity=0)
            _tube(rack, [s[0] + RACK_X, c[1], bar_z], [c[0], c[1], c[2] - 0.02])

    # the robot stands on a platform: its base is at z = 0, and a hanging 40 cm stick reaches ~0.32 m below it
    wb.add_geom(name="ded70_robot_stand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.18, 0.22, -floor_z / 2],
                pos=[0, 0, floor_z / 2], rgba=[0.25, 0.25, 0.27, 1], contype=0, conaffinity=0)
    kick = wb.add_body(name="ded70_kick", pos=[s[0] + 0.05, s[1] - 0.18, floor_z + 0.12],
                       quat=[np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0])           # face the drummer
    _pad(kick, "ded70_kick_head", list(head_solref), solimp)
    _tube(rack, [s[0] + 0.12, s[1] - 0.18, floor_z], [s[0] + 0.12, s[1] - 0.18, floor_z + 0.2])
    return snare
