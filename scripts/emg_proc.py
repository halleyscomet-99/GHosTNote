"""EMG processing shared by calibration and analysis (Myo: 200 Hz, so 20-90 Hz band)."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

FS = 200.0
CH = [f"ch{i}" for i in range(8)]


def load_emg(path):
    """Load emg.csv and rebuild a jitter-free timeline from the sample index.

    Host stamps = true time + positive, variable BLE latency. Fit t = a + b*idx,
    then shift onto the lower envelope (5th percentile of residuals) so the
    timeline reflects the earliest, least-delayed arrivals.
    """
    df = pd.read_csv(path)
    b, a = np.polyfit(df["idx"], df["t"], 1)
    fit = a + b * df["idx"]
    df["t_fit"] = fit + np.percentile(df["t"] - fit, 5)
    df.attrs["fs_est"] = 1.0 / b
    return df


def envelope(x, fs=FS, rms_ms=30):
    """bandpass 20-90 -> 60 Hz notch -> rectify -> moving RMS. x: (n, ch)."""
    x = np.asarray(x, float)
    x = x - x.mean(axis=0)
    b, a = butter(4, [20 / (fs / 2), 90 / (fs / 2)], btype="band")
    x = filtfilt(b, a, x, axis=0)
    b, a = iirnotch(60, 30, fs)
    x = filtfilt(b, a, x, axis=0)
    x = np.abs(x)
    w = max(1, int(round(rms_ms / 1000 * fs)))
    k = np.ones(w) / w
    return np.sqrt(np.apply_along_axis(lambda c: np.convolve(c ** 2, k, mode="same"), 0, x))


def cci(f, e):
    """Rudolph co-contraction index: (min/max) * (f + e)."""
    lo, hi = np.minimum(f, e), np.maximum(f, e)
    return np.where(hi > 1e-9, lo / np.maximum(hi, 1e-9), 0.0) * (f + e)


class OnlineEMG:
    """Causal version of envelope() for live use: sosfilt with carried state + trailing RMS.

    Lags the zero-phase offline envelope by a few ms (live_stiffness replay: r = 0.97, ~20 ms).
    feed(x (n, 8) raw counts) -> (bandpassed (n, 8), envelope / MVC (n, 8)).
    """

    def __init__(self, mvc, fs=FS, rms_ms=30):
        from scipy.signal import sosfilt_zi, tf2sos
        bp = butter(4, [20 / (fs / 2), 90 / (fs / 2)], btype="band", output="sos")
        self.sos = np.vstack([bp, tf2sos(*iirnotch(60, 30, fs))])
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, None], 8, axis=2) * 0.0
        self.n = max(1, int(round(rms_ms / 1000 * fs)))
        self.hist = np.zeros((self.n, 8))
        self.mvc = np.asarray(mvc, float)

    def feed(self, x):
        from scipy.signal import sosfilt
        x = np.asarray(x, float)
        y = np.empty_like(x)
        for ch in range(x.shape[1]):
            y[:, ch], self.zi[:, :, ch] = sosfilt(self.sos, x[:, ch], zi=self.zi[:, :, ch])
        sq = np.vstack([self.hist, y ** 2])
        cs = np.cumsum(sq, axis=0)
        cs = np.vstack([np.zeros((1, 8)), cs])
        rms = np.sqrt(np.maximum(cs[self.n + 1:] - cs[1:-self.n], 0) / self.n)
        self.hist = sq[-self.n:]
        return y, rms / self.mvc
