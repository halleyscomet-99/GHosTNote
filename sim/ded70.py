"""Donner DED-70 electronic drum kit as MuJoCo primitives, laid out by sim/ded70_layout.yaml.

No CAD of the DED-70 is published, so this is a parametric model from the product spec: four 6.5" mesh pads
(snare + three toms), three 9" rubber cymbals (hi-hat, crash, ride) and a tube rack. The kick is a trigger
pedal on the floor that the robot never touches, so it is left out. Where everything sits (and how each face
is tilted) comes from the layout file; see its header. Only the snare's head ("head", body "pad") is scored;
every other pad, cymbal and tube collides with the stick too.

  spec = mujoco.MjSpec.from_file(...);  add_ded70(spec, auto_snare_top=[x, y, z], head_solref=[-3e4, -8])
"""
from pathlib import Path

import mujoco
import numpy as np
import yaml

LAYOUT = Path(__file__).resolve().parent / "ded70_layout.yaml"
INCH = 0.0254
PAD_R = 6.5 * INCH / 2            # mesh pads
CYM_R = 9.0 * INCH / 2            # rubber cymbals
TUBE_R = 0.0125                   # rack tubes, 25 mm

BLACK = [0.08, 0.08, 0.09, 1]
SHELL = [0.16, 0.16, 0.18, 1]
MESH = [0.78, 0.78, 0.76, 1]
RUBBER = [0.22, 0.22, 0.24, 1]


def load_layout(path=None):
    return yaml.safe_load(open(path or LAYOUT))


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _yaw(deg):
    a = np.radians(deg) / 2
    return [np.cos(a), 0.0, 0.0, np.sin(a)]


def face_quat(tilt_deg, rotate_deg):
    """Orientation of a pad / cymbal: its face leans tilt_deg toward the robot (-x), that lean turned by
    rotate_deg about the vertical."""
    a = -np.radians(tilt_deg) / 2
    return _qmul(_yaw(rotate_deg), [np.cos(a), 0.0, np.sin(a), 0.0])


def _rotz(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _below(c, q, depth):
    """Point `depth` under a body's centre along its own axis (where its holder arm ends)."""
    w, x, y, z = q
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return c - depth * axis


def _tube(body, a, b, name=None):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[TUBE_R, 0, 0], fromto=[*a, *b], rgba=BLACK)


def _pad(body, name, head_solref, solimp):
    """A 6.5" mesh pad: collidable head on top of a shell, rubber rim (visual)."""
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R, 0.01, 0], pos=[0, 0, -0.01],
                  rgba=MESH, solref=head_solref, solimp=solimp)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R + 0.012, 0.022, 0], pos=[0, 0, -0.034],
                  rgba=SHELL, contype=0, conaffinity=0)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R + 0.014, 0.004, 0], pos=[0, 0, -0.004],
                  rgba=BLACK, contype=0, conaffinity=0)


def snare_top(layout, auto_snare_top):
    p = layout["snare"]["position"]
    return np.asarray(auto_snare_top if p == "auto" else p, float)


def add_ded70(spec, auto_snare_top, head_solref=(-3.0e4, -8.0), layout=None):
    """Add the kit to spec.worldbody. auto_snare_top: the snare head's top centre used when the layout says
    position: auto (under the stick tip at the contact pose). Returns the snare's top centre."""
    L = layout or load_layout()
    solimp = [0.95, 0.99, 0.001, 0.5, 2]
    floor_z, yaw = float(L["floor_z"]), float(L.get("kit_rotate_deg", 0))
    s = snare_top(L, auto_snare_top)
    wb = spec.worldbody
    sq = face_quat(L["snare"].get("tilt_deg", 0), L["snare"].get("rotate_deg", 0))
    snare = wb.add_body(name="pad", pos=s.tolist(), quat=sq)
    _pad(snare, "head", list(head_solref), solimp)

    R = L["rack"]
    rack = wb.add_body(name="ded70_rack", pos=[0, 0, 0])
    bar_z = s[2] + R["bar_dz"]
    kit = lambda off: s + _rotz(np.asarray(off, float), yaw)                  # noqa: E731  kit frame -> world
    posts = [kit([R["post_x"], y, 0.0]) for y in R["post_y"]]
    for i, p in enumerate(posts):
        _tube(rack, [p[0], p[1], floor_z], [p[0], p[1], bar_z + 0.12], name=f"ded70_post{i}")
        f0, f1 = kit([R["post_x"] - 0.18, R["post_y"][i], 0]), kit([R["post_x"] + 0.18, R["post_y"][i], 0])
        _tube(rack, [f0[0], f0[1], floor_z + 0.02], [f1[0], f1[1], floor_z + 0.02])           # feet
    _tube(rack, [posts[0][0], posts[0][1], bar_z], [posts[1][0], posts[1][1], bar_z], name="ded70_bar")
    on_bar = lambda c: kit([R["post_x"], _rotz(c - s, -yaw)[1], 0])[:2].tolist() + [bar_z]   # noqa: E731
    _tube(rack, on_bar(s), _below(s, sq, 0.05))                                              # snare arm
    m = kit([R["post_x"] - 0.05, R["post_y"][0], 0])
    rack.add_geom(name="ded70_module", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.09, 0.06],
                  pos=[m[0], m[1], bar_z + 0.05], quat=_yaw(yaw), rgba=BLACK)

    for group, kind in (("pads", "pad"), ("cymbals", "cymbal")):
        for name, it in L[group].items():
            c = kit(it["offset"])
            q = _qmul(_yaw(yaw), face_quat(it.get("tilt_deg", 0), it.get("rotate_deg", 0)))
            b = wb.add_body(name=f"ded70_{name}", pos=c.tolist(), quat=q)
            if kind == "pad":
                _pad(b, f"ded70_{name}_head", list(head_solref), solimp)
                _tube(rack, on_bar(c), _below(c, q, 0.05))
            else:
                b.add_geom(name=f"ded70_{name}_cymbal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[CYM_R, 0.003, 0], rgba=RUBBER, solref=[-1.0e4, -5.0], solimp=solimp)
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.008, 0], pos=[0, 0, 0.008],
                           rgba=BLACK, contype=0, conaffinity=0)
                _tube(rack, on_bar(c), _below(c, q, 0.02))

    if L.get("robot_stand", True):
        # the robot's base is at z = 0, and a hanging 40 cm stick reaches ~0.32 m below it
        wb.add_geom(name="ded70_robot_stand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.18, 0.22, -floor_z / 2],
                    pos=[0, 0, floor_z / 2], rgba=[0.25, 0.25, 0.27, 1], contype=0, conaffinity=0)
    return s
