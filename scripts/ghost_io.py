"""Shared acquisition layer: Myo EMG/IMU + DED-70 MIDI (+ optional audio) on ONE clock.

Every sample is stamped with time.perf_counter() - T0 at arrival on the host.
One process, several threads, one clock -> no cross-device sync problem.
EMG also carries a running sample index so processing can rebuild an exact
200 Hz timeline (BLE delivers packets in bursts, host stamps jitter ~10-30 ms).
"""
import atexit
import csv
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config"

T0 = time.perf_counter()
T0_WALL = time.time()


def now():
    return time.perf_counter() - T0


# Myo IMU raw-unit scales (myohw.h)
QUAT_SCALE, ACC_SCALE, GYRO_SCALE = 16384.0, 2048.0, 16.0  # -, g, deg/s


def mac_to_addr(mac):
    """'F9:2E:A0:A5:2F:12' -> pyomyo address list (BLE byte order is reversed)."""
    return [int(b, 16) for b in reversed(mac.split(":"))]


class MyoStream(threading.Thread):
    """Raw 200 Hz EMG (int8, 8 ch) + 50 Hz IMU into a queue.

    Two Myos: give each its own BLED112 (tty) and MAC, and tag="2" for the second one; its items are
    queued as "emg2"/"imu2" and TakeWriter stores them in emg2.csv / imu2.csv.
    Env defaults: MYO_TTY / MYO_MAC (first Myo), MYO2_TTY / MYO2_MAC (second, tag "2").
    """

    def __init__(self, q, tty=None, mac=None, tag=""):
        super().__init__(daemon=True)
        env = "MYO2" if tag == "2" else "MYO"
        self.q, self.tag = q, tag
        self.tty = tty or os.environ.get(f"{env}_TTY")
        self.mac = mac or os.environ.get(f"{env}_MAC")
        self.emg_idx = 0
        self.last_emg_t = None
        self.ready = threading.Event()
        self.error = None
        self.stop_flag = threading.Event()
        self.m = None
        # The BLED112 keeps a BLE link alive after the process dies, leaving the
        # Myo "connected" and invisible to the next scan. Always release it.
        atexit.register(self.close)

    def close(self):
        self.stop_flag.set()
        if self.m is not None:
            # raw BGAPI connection_disconnect(h); fire-and-forget, because
            # bt.disconnect() blocks on a reply the reader thread may swallow
            try:
                for h in (0, 1, 2):
                    self.m.bt.ser.write(bytes([0, 1, 3, 0, h]))
                self.m.bt.ser.flush()
            except Exception:
                pass

    def run(self):
        try:
            import builtins
            import pyomyo.pyomyo as pm
            from pyomyo import Myo, emg_mode
            # silence pyomyo's per-advertisement "scan response" spam
            pm.print = lambda *a, **k: None if a and a[0] == "scan response:" else builtins.print(*a, **k)
            m = Myo(tty=self.tty, mode=emg_mode.RAW)
            self.m = m
            m.connect(mac_to_addr(self.mac) if self.mac else None)
            m.add_emg_handler(self._on_emg)
            m.add_imu_handler(self._on_imu)
            m.sleep_mode(1)       # never sleep
            # forearm Myo blue, upper-arm Myo orange: easy to tell apart when both are on
            col = [255, 100, 0] if self.tag == "2" else [0, 128, 255]
            m.set_leds(col, col)
            m.vibrate(1)
            try:  # battery characteristic: last payload byte = percent
                self.battery = m.read_attr(0x11).payload[-1]
                print(f"Myo{self.tag or ''} battery: {self.battery}%" + ("  <-- LOW, charge first!" if self.battery < 30 else ""))
            except Exception:
                self.battery = None
            self.ready.set()
            while not self.stop_flag.is_set():
                m.run()
            m.disconnect()
        except Exception as e:  # surface to main thread
            self.error = e
            self.ready.set()

    def _on_emg(self, emg, moving):
        t = now()
        self.last_emg_t = t
        self.q.put(("emg" + self.tag, t, self.emg_idx, list(emg)))
        self.emg_idx += 1

    def _on_imu(self, quat, acc, gyro):
        self.q.put(("imu" + self.tag, now(),
                    [v / QUAT_SCALE for v in quat],
                    [v / ACC_SCALE for v in acc],
                    [v / GYRO_SCALE for v in gyro]))


def scan_myos(tty=None, seconds=6.0):
    """List advertising Myos seen by one BLED112 -> [(mac, rssi)]. A Myo on USB charge or asleep is silent."""
    import builtins
    import pyomyo.pyomyo as pm
    from pyomyo import Myo, emg_mode
    pm.print = lambda *a, **k: None if a and a[0] in ("scan response:", "scanning...") else builtins.print(*a, **k)
    m = Myo(tty=tty, mode=emg_mode.RAW)
    m.bt.ser.timeout = 0.5            # recv_packet returns None instead of blocking when nothing advertises
    for h in (0, 1, 2):               # drop a link a crashed process may have left on this dongle
        m.bt.ser.write(bytes([0, 1, 3, 0, h]))
    m.bt.end_scan()
    m.bt.discover()
    seen, t_end = {}, time.time() + seconds
    sig = b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'   # Myo service UUID
    while time.time() < t_end:
        p = m.bt.recv_packet()
        if p is None or p.cls != 6 or p.cmd != 0:
            continue
        if p.payload.endswith(sig):
            mac = ":".join(f"{b:02X}" for b in reversed(p.payload[2:8]))
            rssi = p.payload[0] - 256 if p.payload[0] > 127 else p.payload[0]
            seen[mac] = rssi
    m.bt.end_scan()
    m.bt.ser.close()
    return sorted(seen.items(), key=lambda kv: -kv[1])


def wait_for_myo(myo, timeout=180):
    """Block until the Myo connects. A sleeping Myo only advertises after it is moved."""
    t_end = time.time() + timeout
    while not myo.ready.wait(5):
        if time.time() > t_end:
            raise SystemExit("Myo not found. Is it charged, off USB, and is MYO_TTY the right dongle?")
        print("  ... still scanning - shake your arm / tap the Myo to wake it (logo should light up)")
    if myo.error:
        raise SystemExit(f"Myo{myo.tag} failed: {myo.error!r}")
    print(f"Myo{myo.tag} connected (it should have vibrated)" + (f" [{myo.mac} on {myo.tty}]" if myo.mac else "."))


def find_midi_port(hint=None):
    import mido
    names = mido.get_input_names()
    if hint:
        for n in names:
            if hint.lower() in n.lower():
                return n
        raise RuntimeError(f"No MIDI input matching {hint!r}. Available: {names}")
    names = [n for n in names if "through" not in n.lower()]
    if not names:
        raise RuntimeError("No MIDI input found. Is the DED-70 plugged in over USB-B and powered on?")
    return names[0]


class MidiStream:
    """Callback-based MIDI input; keeps everything except clock/active-sensing.

    Cheap USB-MIDI cables get reset by the host ("disabled by hub (EMI?)"); the ALSA port keeps its
    name but an already-open handle goes silent. A watchdog reopens the port whenever the set of
    USB sound devices changes, and counts the reconnects."""

    def __init__(self, q, port_hint=None):
        self.q = q
        self.hint = port_hint
        self.reconnects = 0
        self.port = None
        self._open()
        self._usb = self._usb_state()
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _open(self):
        import mido
        self.name = find_midi_port(self.hint)
        self.port = mido.open_input(self.name, callback=self._cb)

    @staticmethod
    def _usb_state():
        st = set()
        for f in Path("/proc/asound").glob("card*/usbbus"):
            try:
                st.add((f.parent.name, f.read_text().strip()))
            except OSError:
                pass
        return st

    def _watch(self):
        while not self._stop.wait(0.5):
            cur = self._usb_state()
            if cur == self._usb:
                continue
            time.sleep(0.5)                       # let ALSA finish creating the new port
            try:
                if self.port is not None:
                    self.port.close()
                self._open()
                self._usb = self._usb_state()
                self.reconnects += 1
                print(f"MIDI: USB device reset -> reopened {self.name} (reconnect #{self.reconnects})")
            except Exception as e:
                print(f"MIDI: reopen failed ({e}); will retry")
                self._usb = set()

    def _cb(self, msg):
        if msg.type in ("clock", "active_sensing"):
            return
        t = now()
        note = getattr(msg, "note", "")
        vel = getattr(msg, "velocity", "")
        typ = msg.type
        if typ == "note_on" and vel == 0:
            typ = "note_off"
        self.q.put(("midi", t, typ, note, vel, getattr(msg, "control", ""),
                    getattr(msg, "value", ""), msg.hex()))

    def close(self):
        self._stop.set()
        if self.port is not None:
            self.port.close()


class AudioStream:
    """Optional: record the module's headphone/line out (needs portaudio + sounddevice)."""

    def __init__(self, path, device=None, sr=48000):
        import sounddevice as sd
        import soundfile as sf
        self.file = sf.SoundFile(path, "w", samplerate=sr, channels=1)
        self.first_t = None
        self.n = 0

        def cb(indata, frames, tinfo, status):
            if self.first_t is None:
                self.first_t = now() - frames / sr
            self.file.write(indata.copy())
            self.n += frames

        self.stream = sd.InputStream(samplerate=sr, channels=1, device=device, callback=cb)
        self.stream.start()
        self.sr = sr

    def close(self):
        self.stream.stop()
        self.stream.close()
        self.file.close()


class TakeWriter(threading.Thread):
    """Drains the queue into emg.csv / imu.csv / midi.csv inside one take folder."""

    def __init__(self, q, take_dir, on_midi=None):
        super().__init__(daemon=True)
        self.q, self.dir, self.on_midi = q, Path(take_dir), on_midi
        self.dir.mkdir(parents=True, exist_ok=True)
        self.counts = {"emg": 0, "imu": 0, "midi": 0}
        self.stop_flag = threading.Event()
        self.files, self.w = {}, {}
        for kind in ("emg", "imu", "midi"):
            self._open(kind)

    HEAD = {"emg": ["t", "idx"] + [f"ch{i}" for i in range(8)],
            "imu": ["t", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"],
            "midi": ["t", "type", "note", "velocity", "control", "value", "hex"]}

    def _open(self, kind):
        """emg/imu/midi always; emg2/imu2 (second Myo) on their first sample."""
        self.files[kind] = open(self.dir / f"{kind}.csv", "w", newline="")
        self.w[kind] = csv.writer(self.files[kind])
        self.w[kind].writerow(self.HEAD[kind.rstrip("2")])
        self.counts.setdefault(kind, 0)

    def run(self):
        while not (self.stop_flag.is_set() and self.q.empty()):
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = item[0]
            if kind not in self.files:
                self._open(kind)
            if kind.startswith("emg"):
                self.w[kind].writerow([f"{item[1]:.6f}", item[2], *item[3]])
            elif kind.startswith("imu"):
                self.w[kind].writerow([f"{item[1]:.6f}", *[f"{v:.5f}" for v in item[2] + item[3] + item[4]]])
            elif kind == "midi":
                self.w["midi"].writerow([f"{item[1]:.6f}", *item[2:]])
                if self.on_midi:
                    self.on_midi(item)
            self.counts[kind] += 1
        for fh in self.files.values():
            fh.close()

    def close(self):
        self.stop_flag.set()
        self.join()


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def session_dir(session):
    d = DATA_RAW / session
    if not d.exists():
        sys.exit(f"Session {session} does not exist. Create it first: python scripts/new_session.py")
    return d
