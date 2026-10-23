"""List the Myos each BLED112 dongle can see, so two Myos can be pinned to (dongle, MAC) pairs.

  python myo_scan.py                 # scans with every dongle for 6 s

Wake both Myos first (tap / move them; a Myo on USB charge does not advertise). Then export, e.g.
  export MYO_TTY=/dev/ttyACM0  MYO_MAC=F9:2E:A0:A5:2F:12      # forearm (blue LED)
  export MYO2_TTY=/dev/ttyACM1 MYO2_MAC=xx:xx:xx:xx:xx:xx     # upper arm (orange LED)
and write the same into session_meta.json -> myo.mac / myo2.mac.
"""
import argparse

from serial.tools.list_ports import comports

from ghost_io import scan_myos

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=6.0)
a = ap.parse_args()
ttys = [p.device for p in comports() if "2458:0001" in (p.hwid or "").upper() or "PID=2458:1" in (p.hwid or "").upper()]
if not ttys:
    raise SystemExit("No BLED112 dongle found (USB id 2458:0001).")
macs = {}
for tty in ttys:
    print(f"scanning with {tty} for {a.seconds:.0f} s ...")
    for mac, rssi in scan_myos(tty, a.seconds):
        print(f"   {mac}   RSSI {rssi} dBm")
        macs.setdefault(mac, tty)
if not macs:
    print("No Myo advertising. Tap/move them (logo should light up), unplug from USB, and retry.")
elif len(macs) >= 2:
    (m1, _), (m2, _) = list(macs.items())[:2]
    print("\nSuggested (check which Myo is which by its LED after connecting: forearm blue, upper arm orange):")
    print(f"  export MYO_TTY={ttys[0]} MYO_MAC={m1}")
    print(f"  export MYO2_TTY={ttys[1] if len(ttys) > 1 else ttys[0]} MYO2_MAC={m2}")
    if len(ttys) < 2:
        print("  !! only one dongle: two Myos need two BLED112 dongles")
