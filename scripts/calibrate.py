"""Step 2: MVC calibration + functional channel localization (one Myo, right forearm).

Protocol: fist / wrist flexion / wrist extension, each 3 x 5 s MAXIMAL effort
against resistance, 30 s rest between. Writes:
  calib/mvc/{emg,imu,midi}.csv, calib/mvc/segments.json, calib/mvc.json
and fills flexor/extensor channels into session_meta.json.
"""
import argparse
import os
import queue
import threading
import time
import numpy as np
from ghost_io import wait_for_myo, MyoStream, TakeWriter, session_dir, load_json, save_json, now
from emg_proc import load_emg, envelope, CH, FS

TASKS = [("fist", "FIST: thumb up, squeeze as hard as you can (thumb outside), wrist straight"),
         ("flex", "FLEXION: palm UP, other hand presses on palm; curl palm up against it, wrist stays still, fingers open"),
         ("ext", "EXTENSION: palm DOWN, other hand presses on back of hand; lift back of hand against it, wrist stays still, fingers open")]

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--hold", type=float, default=5.0)
ap.add_argument("--rest", type=float, default=30.0)
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

sd = session_dir(a.session)
cdir = sd / "calib" / "mvc"


def countdown(sec, label):
    end = time.time() + sec
    while (left := end - time.time()) > 0:
        print(f"\r  {label} {left:4.1f}s ", end="", flush=True)
        if not a.analyze_only and myo.last_emg_t is not None and now() - myo.last_emg_t > 1.0:
            print("\n\n!! EMG stream stopped (battery dead or Myo disconnected). Calibration aborted.")
            print("   Charge / reconnect, then re-run calibrate.py (this folder will be overwritten).")
            myo.close()
            os._exit(1)
        time.sleep(0.1)
    print()


if not a.analyze_only:
    q = queue.Queue()
    myo = MyoStream(q)
    myo.start()
    print("Connecting to Myo ...")
    wait_for_myo(myo)
    time.sleep(1.0)
    while not q.empty():
        q.get_nowait()
    writer = TakeWriter(q, cdir)
    writer.start()
    segs = []
    input("Sit relaxed, arm resting. Press Enter to record 5 s of REST baseline ...")
    t0 = now(); countdown(5, "REST (relax completely)"); segs.append({"task": "rest", "rep": 0, "t0": t0, "t1": now()})
    for task, cue in TASKS:
        for rep in range(1, a.reps + 1):
            print(f"\n[{task} {rep}/{a.reps}] {cue}")
            countdown(3, "get ready")
            print("  GO! MAX EFFORT")
            t0 = now(); countdown(a.hold, "HOLD"); t1 = now()
            segs.append({"task": task, "rep": rep, "t0": t0, "t1": t1})
            if not (task == TASKS[-1][0] and rep == a.reps):
                countdown(a.rest, "rest")
    writer.close()
    myo.close()
    save_json(cdir / "segments.json", segs)
    print(f"\nEMG samples: {writer.counts['emg']}")

# ---------- analysis ----------
# Robust to one bad rep (armband bump, motion artifact): per task take the
# MEDIAN over reps, then MVC = max over tasks of that median 1-s peak.
segs = load_json(cdir / "segments.json")
df = load_emg(cdir / "emg.csv")
env = envelope(df[CH].values)
t = df["t"].values
win = int(FS)  # 1 s sliding mean of the envelope
kern = np.ones(win) / win
env1s = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, env)
raw = df[CH].values

peaks, means, clip = {}, {}, {}
for s in segs:
    m = (t >= s["t0"]) & (t <= s["t1"])
    if not m.any():
        raise SystemExit(f"No EMG in segment {s['task']}{s['rep']} - stream dropped. Re-run calibration.")
    peaks.setdefault(s["task"], []).append(env1s[m].max(axis=0))
    means.setdefault(s["task"], []).append(env[m].mean(axis=0))
    clip.setdefault(s["task"], []).append((np.abs(raw[m]) >= 127).mean(axis=0))
task_mean = {k: np.median(v, axis=0) for k, v in means.items()}
task_peak = {k: np.median(v, axis=0) for k, v in peaks.items() if k != "rest"}
mvc = np.max(np.stack(list(task_peak.values())), axis=0)
rest = task_mean["rest"]
snr = mvc / np.maximum(rest, 1e-9)
dead = [i for i in range(8) if snr[i] < 3]
ratio = task_mean["flex"] / np.maximum(task_mean["ext"], 1e-9)

print("\nchannel   rest    fist    flex     ext     MVC  MVC/rest  flex/ext  clip%")
for i in range(8):
    cmax = max(np.max(v, axis=0)[i] for v in clip.values()) * 100
    tag = "  <-- DEAD (no signal above rest)" if i in dead else ""
    print(f"  ch{i}  {rest[i]:6.2f}  {task_mean['fist'][i]:6.2f}  {task_mean['flex'][i]:6.2f}"
          f"  {task_mean['ext'][i]:6.2f}  {mvc[i]:6.2f}  {snr[i]:7.1f}  {ratio[i]:7.2f}  {cmax:5.1f}{tag}")

# per-rep consistency: coefficient of variation of the rep means
for k in ("fist", "flex", "ext"):
    v = np.array(means[k])
    cv = (v.std(axis=0) / np.maximum(v.mean(axis=0), 1e-9))[[i for i in range(8) if i not in dead]]
    if np.median(cv) > 0.35:
        print(f"WARNING: {k} reps inconsistent (median CV {np.median(cv):.2f}) - same posture & effort each rep")

live = [i for i in range(8) if i not in dead]
order = sorted(live, key=lambda i: ratio[i])
ext_ch, flex_ch = sorted(order[:3]), sorted(order[-3:])
print(f"\nSuggested flexor channels:   {flex_ch}")
print(f"Suggested extensor channels: {ext_ch}")
if dead:
    print(f"WARNING: dead channels {dead} - wet the skin under those pods / tighten the band, then re-run")
if min(ratio[i] for i in ext_ch) >= 1.0:
    print("WARNING: no channel is extensor-dominant (all flex/ext >= 1). Extension effort was weak or the "
          "band is rotated - redo extension with palm DOWN, fingers open, maximal lift.")
if max(np.max(v, axis=0).max() for v in clip.values()) > 0.15:
    print("NOTE: >15% clipping at max effort; MVC is underestimated (OK for relative comparisons).")

save_json(sd / "calib" / "mvc.json", {"mvc": mvc.tolist(), "rest": rest.tolist(),
                                       "task_mean": {k: v.tolist() for k, v in task_mean.items()},
                                       "flexor_channels": flex_ch, "extensor_channels": ext_ch,
                                       "dead_channels": dead, "method": "median over reps",
                                       "fs_est": df.attrs["fs_est"]})
# the latest calibration is the source of truth for this wearing
meta = load_json(sd / "session_meta.json")
meta["myo"]["flexor_channels"], meta["myo"]["extensor_channels"] = flex_ch, ext_ch
meta["myo"]["dead_channels"] = dead
save_json(sd / "session_meta.json", meta)
print("Written to calib/mvc.json and session_meta.json (edit session_meta by hand if you disagree).")
