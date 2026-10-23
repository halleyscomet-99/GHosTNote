"""Live stiffness display: Myo EMG -> CCI -> normalized stiffness k in [0, 1].

  python live_stiffness.py --session 2026-10-08_S01                         # live Myo (calibrated THIS wearing)
  python live_stiffness.py --session 2026-10-08_S01 --replay T07_alt_medium_r3   # replay a recorded take
  python live_stiffness.py --session ... --udp 127.0.0.1:9100                 # also stream k over UDP
  python live_stiffness.py --session ... --udp 127.0.0.1:9100 --myo2          # + upper-arm Myo IMU (robot/arm_teleop.py)

k = clip((CCI - lo) / (hi - lo), 0, 1). By default lo / hi are the median pre-activation CCI of
free / dead hits in --ref-session (alternating takes if present), so a typical free stroke reads ~0
and a typical dead stroke ~1. Keys in the plot window: L = set lo from the last 2 s (hold the stick
loosely), H = set hi from the last 2 s (grip hard), R = back to the reference values.

Online processing is causal (sosfilt with state + trailing 30 ms RMS), so it lags the offline
zero-phase pipeline by a few ms; values are otherwise the same definition.
"""
import argparse
import collections
import queue
import socket
import sys
import threading
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from emg_proc import FS, CH
from ghost_io import MyoStream, wait_for_myo, session_dir, load_json, now, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, help="session whose calib/mvc.json matches the CURRENT wearing")
ap.add_argument("--ref-session", default=None, help="session with strikes.csv for lo/hi (default: --session)")
ap.add_argument("--replay", default=None, help="take folder name to replay instead of a live Myo")
ap.add_argument("--udp", default=None, help="host:port to stream 't,k,cci' (one line per update) and every IMU "
                "sample as 'imu,t,qw,qx,qy,qz' (forearm) / 'imu2,...' (upper arm)")
ap.add_argument("--myo2", action="store_true", help="also connect the upper-arm Myo (MYO2_TTY / MYO2_MAC) for its IMU")
ap.add_argument("--log", default=None, help="CSV path to log every sample")
ap.add_argument("--window", type=float, default=5.0)
ap.add_argument("--no-plot", action="store_true", help="print values instead (testing)")
ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
a = ap.parse_args()

sd = session_dir(a.session)
meta = load_json(sd / "session_meta.json")
if not (sd / "calib" / "mvc.json").exists():
    raise SystemExit(f"{a.session} has no calib/mvc.json (calibrate.py did not finish). Use a session that has one, "
                     "e.g. --session 2026-10-15_S02, and set lo / hi with L / H.")
mvc = np.array(load_json(sd / "calib" / "mvc.json")["mvc"])
F, E = meta["myo"]["flexor_channels"], meta["myo"]["extensor_channels"]

# ---------- reference lo / hi from recorded free vs dead hits ----------
ref = a.ref_session or a.session
sp = ROOT / "data" / "processed" / ref / "strikes.csv"
LO_REF, HI_REF = 0.3, 1.0
if sp.exists():
    st = pd.read_csv(sp)
    st = st[(st.role == "primary") & st.stroke.isin(["free", "dead"])]
    if "design" in st and (st.design == "alt").any():
        st = st[st.design == "alt"]
    if len(st):
        LO_REF = float(st[st.stroke == "free"].pre_cci.median())
        HI_REF = float(st[st.stroke == "dead"].pre_cci.median())
lohi = {"lo": LO_REF, "hi": HI_REF}
print(f"k mapping: lo = {LO_REF:.3f} (median free pre-CCI), hi = {HI_REF:.3f} (median dead pre-CCI) from {ref}")
if not a.replay:
    print("NOTE: MVC must come from the CURRENT wearing. If you re-donned the Myo, run calibrate.py first.")

# ---------- source: live Myo or replay ----------
q = queue.Queue()
stop = threading.Event()
midi_marks = collections.deque()
if a.replay:
    td = sd / "takes" / a.replay
    rec = pd.read_csv(td / "emg.csv")
    mid = pd.read_csv(td / "midi.csv")
    on_t = (mid[mid.type == "note_on"].t.values - rec.t.values[0])

    def replay():
        x = rec[CH].values
        t0 = time.perf_counter()
        for i in range(len(x)):
            target = t0 + i / FS
            while time.perf_counter() < target:
                time.sleep(0.0005)
            if stop.is_set():
                return
            q.put(("emg", i / FS, i, x[i].tolist()))
        stop.set()

    threading.Thread(target=replay, daemon=True).start()
    clock = lambda: None
    print(f"replaying {a.replay} ({len(rec) / FS:.0f} s) at real-time speed")
else:
    myo = MyoStream(q)
    myo.start()
    print("Connecting to Myo ...")
    wait_for_myo(myo)
    if a.myo2:
        myo2 = MyoStream(q, tag="2")
        myo2.start()
        print("Connecting to the upper-arm Myo ...")
        wait_for_myo(myo2)
    on_t = None

udp = None
if a.udp:
    host, port = a.udp.rsplit(":", 1)
    udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
logf = open(a.log, "w") if a.log else None
if logf:
    logf.write("t,flex,ext,cci,k\n")

# ---------- causal pipeline ----------
bp = butter(4, [20 / (FS / 2), 90 / (FS / 2)], btype="band", output="sos")
nt = tf2sos(*iirnotch(60, 30, FS))
sos = np.vstack([bp, nt])
zi = np.repeat(sosfilt_zi(sos)[:, :, None], 8, axis=2)  # (sections, 2, channels)
RMS_N = int(round(0.030 * FS))
sq_hist = np.zeros((RMS_N, 8))
N = int(a.window * FS)
buf = {k: collections.deque(maxlen=N) for k in ("t", "flex", "ext", "cci", "k")}
first = None


def kmap(c):
    return float(np.clip((c - lohi["lo"]) / max(lohi["hi"] - lohi["lo"], 1e-6), 0, 1))


def pump():
    """Drain the queue, filter new samples, append to the display buffers."""
    global zi, sq_hist, first
    rows = []
    while not q.empty():
        it = q.get_nowait()
        if it[0] == "emg":
            rows.append((it[1], it[3]))
        elif it[0] in ("imu", "imu2") and udp:
            udp[0].sendto((f"{it[0]},{it[1]:.4f}," + ",".join(f"{v:.5f}" for v in it[2]) + "\n").encode(), udp[1])
    if not rows:
        return
    t = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows], float)
    if first is None:
        first = t[0]
    y = np.empty_like(x)
    for ch in range(8):
        y[:, ch], zi[:, :, ch] = sosfilt(sos, x[:, ch], zi=zi[:, :, ch])
    sq = np.vstack([sq_hist, y ** 2])
    cs = np.cumsum(sq, axis=0)
    rms = np.sqrt((cs[RMS_N:] - cs[:-RMS_N]) / RMS_N)
    sq_hist = sq[-RMS_N:]
    env = rms / mvc
    f, e = env[:, F].mean(1), env[:, E].mean(1)
    lo, hi = np.minimum(f, e), np.maximum(f, e)
    c = np.where(hi > 1e-9, lo / np.maximum(hi, 1e-9), 0) * (f + e)
    for i in range(len(t)):
        k = kmap(c[i])
        for key, val in (("t", t[i] - first), ("flex", f[i]), ("ext", e[i]), ("cci", c[i]), ("k", k)):
            buf[key].append(val)
        if logf:
            logf.write(f"{t[i]:.4f},{f[i]:.4f},{e[i]:.4f},{c[i]:.4f},{k:.4f}\n")
    if udp:
        udp[0].sendto(f"{buf['t'][-1]:.4f},{buf['k'][-1]:.4f},{buf['cci'][-1]:.4f}\n".encode(), udp[1])


def recent(key, sec=2.0):
    t = np.array(buf["t"])
    v = np.array(buf[key])
    return v[t >= t[-1] - sec] if len(t) else v


def finish():
    stop.set()
    if logf:
        logf.close()
    if not a.replay:
        myo.close()
        if a.myo2:
            myo2.close()


t_start = time.time()
if a.no_plot:
    try:
        while not stop.is_set() and (a.duration is None or time.time() - t_start < a.duration):
            time.sleep(0.25)
            pump()
            if buf["t"]:
                print(f"t {buf['t'][-1]:6.2f}  flex {buf['flex'][-1]:.2f}  ext {buf['ext'][-1]:.2f}  "
                      f"CCI {buf['cci'][-1]:.2f}  k {buf['k'][-1]:.2f}")
    except KeyboardInterrupt:
        pass
    finish()
    sys.exit(0)

# ---------- live plot ----------
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(3, 2, width_ratios=[6, 1], hspace=0.35)
ax_env, ax_cci, ax_k = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
ax_bar = fig.add_subplot(gs[:, 1])
l_f, = ax_env.plot([], [], color="#2a78d6", lw=1.8, label="flexor")
l_e, = ax_env.plot([], [], color="#eb6834", lw=1.8, label="extensor")
ax_env.set_ylim(0, 1.4); ax_env.set_ylabel("envelope ÷MVC"); ax_env.legend(loc="upper left", ncol=2, frameon=False)
l_c, = ax_cci.plot([], [], color="#111418", lw=1.8)
h_lo = ax_cci.axhline(lohi["lo"], color="#1baf7a", ls="--", lw=1, label="lo (k = 0)")
h_hi = ax_cci.axhline(lohi["hi"], color="#4a3aa7", ls="--", lw=1, label="hi (k = 1)")
ax_cci.set_ylim(0, 2); ax_cci.set_ylabel("CCI"); ax_cci.legend(loc="upper left", ncol=2, frameon=False)
l_k, = ax_k.plot([], [], color="#4a3aa7", lw=2)
ax_k.set_ylim(-0.05, 1.05); ax_k.set_ylabel("stiffness k"); ax_k.set_xlabel("time (s)")
for ax in (ax_env, ax_cci, ax_k):
    ax.grid(alpha=0.3)
bar = ax_bar.bar([0], [0], color="#4a3aa7", width=0.6)[0]
ax_bar.set_ylim(0, 1); ax_bar.set_xlim(-0.6, 0.6); ax_bar.set_xticks([]); ax_bar.set_title("k")
k_txt = ax_bar.text(0, 0.5, "", ha="center", va="center", fontsize=22, fontweight="bold", color="white")
mark_lines = []
title = fig.suptitle("")


def on_key(ev):
    if ev.key in ("l", "L") and buf["cci"]:
        lohi["lo"] = float(np.median(recent("cci")))
    elif ev.key in ("h", "H") and buf["cci"]:
        lohi["hi"] = float(np.percentile(recent("cci"), 75))
    elif ev.key in ("r", "R"):
        lohi.update(lo=LO_REF, hi=HI_REF)
    h_lo.set_ydata([lohi["lo"]] * 2); h_hi.set_ydata([lohi["hi"]] * 2)


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("close_event", lambda ev: finish())


def update(_):
    pump()
    if not buf["t"]:
        title.set_text("waiting for EMG ...")
        return
    t = np.array(buf["t"])
    x0, x1 = t[-1] - a.window, t[-1]
    for ln, key in ((l_f, "flex"), (l_e, "ext"), (l_c, "cci"), (l_k, "k")):
        ln.set_data(t, np.array(buf[key]))
    for ax in (ax_env, ax_cci, ax_k):
        ax.set_xlim(x0, x1)
    for ln in mark_lines:
        ln.remove()
    mark_lines.clear()
    if on_t is not None:
        for ot in on_t[(on_t > x0) & (on_t < x1)]:
            for ax in (ax_env, ax_cci, ax_k):
                mark_lines.append(ax.axvline(ot, color="#7b8591", lw=0.8, alpha=0.6))
    k = buf["k"][-1]
    bar.set_height(k)
    k_txt.set_text(f"{k:.2f}"); k_txt.set_y(max(k / 2, 0.06)); k_txt.set_color("white" if k > 0.12 else "#111418")
    src = f"replay {a.replay}" if a.replay else "live Myo"
    title.set_text(f"{src}  |  lo {lohi['lo']:.2f}  hi {lohi['hi']:.2f}  |  keys: L set lo, H set hi, R reset"
                   + ("  |  grey lines = MIDI hits" if on_t is not None else ""))
    if a.duration and time.time() - t_start > a.duration:
        plt.close(fig)


ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()
finish()
