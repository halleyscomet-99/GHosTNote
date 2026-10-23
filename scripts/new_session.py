"""Create a session folder: data/raw/<YYYY-MM-DD>_S<NN>/ with metadata template.

One session = one wearing of the Myo. Re-donning the armband = new session
(MVC normalization is only valid inside one wearing).
"""
import argparse
import datetime
from ghost_io import DATA_RAW, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default="halley")
ap.add_argument("--arm", default="right")
ap.add_argument("--notes", default="")
a = ap.parse_args()

day = datetime.date.today().isoformat()
n = 1
while (DATA_RAW / f"{day}_S{n:02d}").exists():
    n += 1
name = f"{day}_S{n:02d}"
d = DATA_RAW / name
for sub in ["calib", "takes", "video", "photos"]:
    (d / sub).mkdir(parents=True)

save_json(d / "session_meta.json", {
    "session": name,
    "subject": a.subject,
    "arm": a.arm,
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "myo": {
        "placement": "upper forearm, logo on dorsal midline (extensor side)",
        "landmark_cm_from_elbow_crease": None,
        "photo": "photos/myo_position.jpg",
        "flexor_channels": None,
        "extensor_channels": None
    },
    "ded70": {
        "note_map": {},
        "pad_settings": {"S_sensitivity": None, "T_threshold": None, "U_curve": None},
        "double_stroke_retrigger_ok": None
    },
    "stick": "",
    "camera": {
        "type": "RealSense (ghost_studio.py) + optional phone slow-mo",
        "position": "right-front of the player, 30-45 deg off the side view",
        "distance_m": None, "height_m": None, "tilt_deg": None,
        "photo": "photos/camera_position.jpg"
    },
    "notes": a.notes
})
print(f"Created {d}")
print(f"SESSION={name}")
