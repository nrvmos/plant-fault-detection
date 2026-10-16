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

    Limits are the given percentile of each statistic on the training data.
    """

    variance: float = 0.90
    percentile: float = 99.0

    def fit(self, x: np.ndarray) -> "PCAMonitor":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        z = (x - self.mean) / self.std
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        eig = s**2 / (len(z) - 1)
        self.n_components = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.variance) + 1)
        self.p = vt[: self.n_components].T
        self.eig = eig[: self.n_components]
        t2, spe = self.stats(x)
        self.t2_limit = np.percentile(t2, self.percentile)
        self.spe_limit = np.percentile(spe, self.percentile)
        return self

    def stats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (x - self.mean) / self.std
        scores = z @ self.p
        t2 = (scores**2 / self.eig).sum(axis=1)
        resid = z - scores @ self.p.T
        spe = (resid**2).sum(axis=1)
        return t2, spe

    def flags(self, x: np.ndarray) -> np.ndarray:
        t2, spe = self.stats(x)
        return (t2 > self.t2_limit) | (spe > self.spe_limit)


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
NORMAL_SPLIT = 480  # d00_te rows before this are dev, the rest are report


def first_alarm(alarm: np.ndarray, start: int) -> int | None:
    """Index of the first sample at or after `start` with the alarm sounding (the usual benchmark definition).

    An alarm that was already sounding at onset counts, so with many nuisance alarms some quick
    detections are luck; `alarm_on_share` in the score says how often that happens in normal running.
    """
    hits = np.flatnonzero(alarm[start:])
    return int(start + hits[0]) if hits.size else None


def score(detector, delay_samples: int, split: str = "report", data_dir: Path = DATA_DIR) -> dict:
    faults = DEV_FAULTS if split == "dev" else REPORT_FAULTS
    normal = load("d00_te.dat", data_dir)
    normal = normal[:NORMAL_SPLIT] if split == "dev" else normal[NORMAL_SPLIT:]
    a = on_delay(detector.flags(normal), delay_samples)
    false_events = alarm_events(a)
    normal_samples = len(a)
    on_samples = int(a.sum())
    rows = []
    for f in faults:
        a = on_delay(detector.flags(load(f"d{f:02d}_te.dat", data_dir)), delay_samples)
        pre = a[:TEST_FAULT_ONSET]
        false_events += alarm_events(pre)
        normal_samples += len(pre)
        on_samples += int(pre.sum())
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
        "faults": rows,
    }
