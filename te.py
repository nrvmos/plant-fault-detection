"""Fault detection on the Tennessee Eastman benchmark, scored the way an operator would judge it.

Both detectors learn from normal operation only (d00.dat) and are scored on the
separate test runs (dXX_te.dat), so no test data is used to fit or tune anything.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/{name}"
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_MINUTES = 3
TEST_FAULT_ONSET = 160  # test runs switch the fault on after 8 h (160 samples)
SHIFT_HOURS = 12
FAULTS = range(1, 22)

FAULT_NAMES = {
    1: "A/C feed ratio step",
    2: "B composition step",
    3: "D feed temperature step",
    4: "Reactor cooling water inlet temp step",
    5: "Condenser cooling water inlet temp step",
    6: "A feed loss",
    7: "C header pressure loss",
    8: "A, B, C feed composition random",
    9: "D feed temperature random",
    10: "C feed temperature random",
    11: "Reactor cooling water inlet temp random",
    12: "Condenser cooling water inlet temp random",
    13: "Reaction kinetics slow drift",
    14: "Reactor cooling water valve sticking",
    15: "Condenser cooling water valve sticking",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Stream 4 valve fixed at steady state",
}


TAG_NAMES = [
    "A feed", "D feed", "E feed", "A and C feed", "Recycle flow", "Reactor feed rate", "Reactor pressure",
    "Reactor level", "Reactor temperature", "Purge rate", "Separator temperature", "Separator level",
    "Separator pressure", "Separator underflow", "Stripper level", "Stripper pressure", "Stripper underflow",
    "Stripper temperature", "Stripper steam flow", "Compressor work", "Reactor cooling water outlet temp",
    "Separator cooling water outlet temp",
    *[f"Reactor feed analysis {c}" for c in "ABCDEF"],
    *[f"Purge gas analysis {c}" for c in "ABCDEFGH"],
    *[f"Product analysis {c}" for c in "DEFGH"],
    "D feed valve", "E feed valve", "A feed valve", "A and C feed valve", "Compressor recycle valve",
    "Purge valve", "Separator liquid valve", "Stripper product valve", "Stripper steam valve",
    "Reactor cooling water valve", "Condenser cooling water valve",
]
TAG_IDS = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]


def data_files() -> list[str]:
    names = ["d00.dat", "d00_te.dat"]
    names += [f"d{i:02d}_te.dat" for i in FAULTS]
    return names


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in data_files():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(DATA_URL.format(name=name), path)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    x = np.loadtxt(data_dir / name)
    # d00.dat is stored transposed (52 variables x 500 samples).
    if x.shape[0] == 52 and x.shape[1] != 52:
        x = x.T
    return x


@dataclass
class ThreeSigma:
    """Baseline: a fixed high/low alarm on every tag at mean +/- k standard deviations."""

    k: float = 3.0

    def fit(self, x: np.ndarray) -> "ThreeSigma":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        return self

    def flags(self, x: np.ndarray) -> np.ndarray:
        z = np.abs((x - self.mean) / self.std)
        return (z > self.k).any(axis=1)


@dataclass
class PCAMonitor:
    """PCA on normal data, alarming on Hotelling T^2 or the residual (SPE/Q).

    Limits are the given percentile of each statistic, taken on `calibration` data when given
    (a separate normal run) and on the training data otherwise. A model always looks calmer on the
    data it learned from, so limits taken from the training data are too tight on new runs.
    `smoothing` < 1 applies an EWMA filter to both statistics before the limit check.
    """

    variance: float = 0.90
    percentile: float = 99.0
    smoothing: float = 1.0

    def fit(self, x: np.ndarray, calibration: np.ndarray | None = None) -> "PCAMonitor":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        z = (x - self.mean) / self.std
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        eig = s**2 / (len(z) - 1)
        self.n_components = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.variance) + 1)
        self.p = vt[: self.n_components].T
        self.eig = eig[: self.n_components]
        t2, spe = self.stats(x if calibration is None else calibration)
        self.t2_limit = np.percentile(t2, self.percentile)
        self.spe_limit = np.percentile(spe, self.percentile)
        return self

    def stats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (x - self.mean) / self.std
        scores = z @ self.p
        t2 = (scores**2 / self.eig).sum(axis=1)
        resid = z - scores @ self.p.T
        spe = (resid**2).sum(axis=1)
        return ewma(t2, self.smoothing), ewma(spe, self.smoothing)

    def flags(self, x: np.ndarray) -> np.ndarray:
        t2, spe = self.stats(x)
        return (t2 > self.t2_limit) | (spe > self.spe_limit)


def ewma(x: np.ndarray, weight: float) -> np.ndarray:
    """Exponentially weighted moving average, like a first-order filter on a DCS input."""
    if weight >= 1:
        return x
    out = np.empty(len(x))
    level = x[0]
    for i, v in enumerate(x):
        level = weight * v + (1 - weight) * level
        out[i] = level
    return out


def on_delay(flags: np.ndarray, samples: int) -> np.ndarray:
    """An alarm only sounds once the raw flag has held for `samples` in a row (like a DCS on-delay)."""
    if samples <= 1:
        return flags.copy()
    out = np.zeros_like(flags)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= samples
    return out


def alarm_events(alarm: np.ndarray) -> int:
    """Count separate alarms (rising edges), which is what an operator hears."""
    a = alarm.astype(int)
    return int(a[0] + np.sum(np.diff(a) == 1))


# Runs are split in two so choices are made on one half and reported on the other.
DEV_FAULTS = [f for f in FAULTS if f % 2 == 1]
REPORT_FAULTS = [f for f in FAULTS if f % 2 == 0]
CALIBRATION_ROWS = 480  # the first 24 h of the normal test run set the alarm limits; the rest is report data


def calibration_data(data_dir: Path = DATA_DIR) -> np.ndarray:
    return load("d00_te.dat", data_dir)[:CALIBRATION_ROWS]


def first_alarm(alarm: np.ndarray, start: int) -> int | None:
    """Index of the first sample at or after `start` with the alarm sounding (the usual benchmark definition).

    An alarm that was already sounding at onset counts, so with many nuisance alarms some quick
    detections are luck; `alarm_on_share` in the score says how often that happens in normal running.
    """
    hits = np.flatnonzero(alarm[start:])
    return int(start + hits[0]) if hits.size else None


def _hours_with_alarm(alarm: np.ndarray) -> tuple[int, int]:
    per_hour = 60 // SAMPLE_MINUTES
    windows = [alarm[i : i + per_hour] for i in range(0, len(alarm) - per_hour + 1, per_hour)]
    return sum(bool(w.any()) for w in windows), len(windows)


def score(detector, delay_samples: int, split: str = "report", data_dir: Path = DATA_DIR) -> dict:
    faults = DEV_FAULTS if split == "dev" else REPORT_FAULTS
    false_events = normal_samples = on_samples = 0
    hours_with_alarm = hours_seen = 0
    if split == "report":
        # The dev split never scores the calibration run, so its false alarms are out-of-sample too.
        a = on_delay(detector.flags(load("d00_te.dat", data_dir)[CALIBRATION_ROWS:]), delay_samples)
        false_events, normal_samples, on_samples = alarm_events(a), len(a), int(a.sum())
        hours_with_alarm, hours_seen = _hours_with_alarm(a)
    rows = []
    for f in faults:
        a = on_delay(detector.flags(load(f"d{f:02d}_te.dat", data_dir)), delay_samples)
        pre = a[:TEST_FAULT_ONSET]
        false_events += alarm_events(pre)
        normal_samples += len(pre)
        on_samples += int(pre.sum())
        h, n = _hours_with_alarm(pre)
        hours_with_alarm += h
        hours_seen += n
        hit = first_alarm(a, TEST_FAULT_ONSET)
        rows.append(
            {
                "fault": f,
                "name": FAULT_NAMES[f],
                "delay_min": (hit - TEST_FAULT_ONSET) * SAMPLE_MINUTES if hit is not None else None,
                "coverage": float(a[TEST_FAULT_ONSET:].mean()),
                "pre_fault_coverage": float(pre.mean()),
            }
        )
    normal_hours = normal_samples * SAMPLE_MINUTES / 60
    return {
        "split": split,
        "false_alarms_per_shift": false_events / normal_hours * SHIFT_HOURS,
        "normal_hours": normal_hours,
        "alarm_on_share": on_samples / normal_samples,
        # How often a random hour of normal running already has an alarm in it: the chance that
        # a "caught within 1 h" is luck rather than detection.
        "chance_hour_alarms": hours_with_alarm / hours_seen,
        "faults": rows,
    }


def find_normal_zone(history: np.ndarray, percentile: float = 99.0, margin: tuple[int, int] = (12, 10), rounds: int = 20) -> np.ndarray:
    """Read the history backwards to find the samples that look like normal running. Needs no labels.

    Start from the most typical samples, fit PCA on what is currently believed normal, flag anything outside its limits for 3 samples
    in a row as an upset, drop it with a margin (36 min before, 30 min after), refit, and repeat
    until the normal zone stops changing. Returns a boolean mask of the samples kept as normal.
    """
    # Start from the most typical 75% of samples, so a long upset can't pass itself off as normal.
    t2, spe = PCAMonitor().fit(history).stats(history)
    keep = (t2 <= np.percentile(t2, 75)) & (spe <= np.percentile(spe, 75))
    for _ in range(rounds):
        model = PCAMonitor(percentile=percentile).fit(history[keep])
        upset = on_delay(model.flags(history), 3)
        drop = upset.copy()
        for i in np.flatnonzero(upset):
            drop[max(0, i - margin[0]) : i + margin[1]] = True
        new = ~drop
        if (new == keep).all():
            break
        keep = new
    return keep


@dataclass
class TagLimits:
    """High and low alarm points per tag, suggested from the normal zone of the history.

    Each limit sits at the 0.1st/99.9th percentile of normal running plus `pad` of the normal span.
    `tags` (indices) limits which tags carry an alarm at all.
    """

    pad: float = 0.10
    tags: np.ndarray | None = None

    def fit(self, normal: np.ndarray) -> "TagLimits":
        lo, hi = np.percentile(normal, 0.1, axis=0), np.percentile(normal, 99.9, axis=0)
        span = hi - lo
        self.low, self.high = lo - self.pad * span, hi + self.pad * span
        return self

    def tag_flags(self, x: np.ndarray) -> np.ndarray:
        return (x < self.low) | (x > self.high)

    def flags(self, x: np.ndarray) -> np.ndarray:
        f = self.tag_flags(x)
        return (f if self.tags is None else f[:, self.tags]).any(axis=1)


def build_history(faults=None, fault_hours: int = 4, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch the training day, the calibration run and the dev fault runs into one historian record.

    Each fault run keeps its 8 h of normal running and the first `fault_hours` of the upset, so the
    record is mostly normal with upsets in it, the way a real history is. Returns the data, the true
    upset labels (used only to check the result, never to find the normal zone) and each sample's fault id.
    """
    faults = DEV_FAULTS if faults is None else faults
    parts = [load("d00.dat", data_dir), calibration_data(data_dir)]
    labels = [np.zeros(len(p), dtype=bool) for p in parts]
    source = [np.zeros(len(p), dtype=int) for p in parts]
    for f in faults:
        run = load(f"d{f:02d}_te.dat", data_dir)[: TEST_FAULT_ONSET + fault_hours * 20]
        lab = np.zeros(len(run), dtype=bool)
        lab[TEST_FAULT_ONSET:] = True
        parts.append(run)
        labels.append(lab)
        source.append(np.full(len(run), f))
    return np.vstack(parts), np.concatenate(labels), np.concatenate(source)
