"""Usage (from the repo root):
    python run.py download
    python run.py tradeoff   # dev split, for choosing the operating point
    python run.py report     # report split, after config.py is filled in
    python run.py chart      # writes docs/fault-example.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import te  # noqa: E402

THREE_SIGMA_KS = [3.0, 3.5, 4.0, 5.0]
PCA_PERCENTILES = [99.0, 99.5, 99.9]
ON_DELAYS = [1, 3, 5, 10]
# Faults 3, 9 and 15 barely move the measured tags; they're shown but left out of the medians.
HARD_TO_SEE = {3, 9, 15}


def summary(result: dict) -> tuple[float, float, int, int]:
    rows = [r for r in result["faults"] if r["fault"] not in HARD_TO_SEE]
    delays = [r["delay_min"] if r["delay_min"] is not None else np.inf for r in rows]
    caught_in_hour = sum(d <= 60 for d in delays)
    return result["false_alarms_per_shift"], float(np.median(delays)), caught_in_hour, len(rows)


def tradeoff() -> None:
    train = te.load("d00.dat")
    print("Dev split: odd-numbered faults + first 24 h of the normal test run.\n")
    print(f"{'detector':<22}{'must hold':>9}{'false alarms/shift':>20}{'median delay (min)':>20}{'caught within 1 h':>19}")
    for name, detectors in [
        ("per-tag k", [(k, te.ThreeSigma(k=k).fit(train)) for k in THREE_SIGMA_KS]),
        ("PCA pct", [(p, te.PCAMonitor(percentile=p).fit(train)) for p in PCA_PERCENTILES]),
    ]:
        for setting, det in detectors:
            for d in ON_DELAYS:
                fa, med, caught, n = summary(te.score(det, d, "dev"))
                print(f"{name + ' ' + str(setting):<22}{d * te.SAMPLE_MINUTES:>8}m{fa:>20.2f}{med:>20.0f}{caught:>15}/{n}")


def report() -> None:
    if None in (config.THREE_SIGMA_K, config.PCA_PERCENTILE, config.THREE_SIGMA_HOLD, config.PCA_HOLD):
        sys.exit("Set THREE_SIGMA_K, PCA_PERCENTILE, THREE_SIGMA_HOLD and PCA_HOLD in config.py from the dev table first.")
    train = te.load("d00.dat")
    m = te.SAMPLE_MINUTES
    results = {
        f"Per-tag limits (k={config.THREE_SIGMA_K}, hold {config.THREE_SIGMA_HOLD * m} min)": te.score(
            te.ThreeSigma(k=config.THREE_SIGMA_K).fit(train), config.THREE_SIGMA_HOLD
        ),
        f"PCA T²/SPE ({config.PCA_PERCENTILE}th pct, hold {config.PCA_HOLD * m} min)": te.score(
            te.PCAMonitor(percentile=config.PCA_PERCENTILE).fit(train), config.PCA_HOLD
        ),
    }
    print("Report split: even-numbered faults + last 24 h of the normal test run.\n")
    print("| Detector | False alarms per 12 h shift | Alarm sounding in normal running | Median detection delay | Faults caught within 1 h |")
    print("|---|---:|---:|---:|---:|")
    for name, r in results.items():
        fa, med, caught, n = summary(r)
        print(f"| {name} | {fa:.2f} | {r['alarm_on_share']:.0%} of the time | {med:.0f} min | {caught}/{n} |")
    print("\n| Fault | What changes | " + " | ".join(results) + " |")
    print("|---|---|" + "---:|" * len(results))
    for i, row in enumerate(next(iter(results.values()))["faults"]):
        cells = []
        for r in results.values():
            delay = r["faults"][i]["delay_min"]
            cells.append("missed" if delay is None else f"{delay} min")
        mark = " (hard to see)" if row["fault"] in HARD_TO_SEE else ""
        print(f"| {row['fault']}{mark} | {row['name']} | " + " | ".join(cells) + " |")


EXAMPLE = {"k": 4.0, "pct": 99.9, "k_hold": 3, "pct_hold": 3}
CHART_FAULT = 11  # a dev-split fault, so the chart never shows report data
REACTOR_TEMP = 8  # XMEAS(9), reactor temperature


def chart(out: Path = Path(__file__).parent / "docs" / "fault-example.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = None not in (config.THREE_SIGMA_K, config.PCA_PERCENTILE, config.THREE_SIGMA_HOLD, config.PCA_HOLD)
    k = config.THREE_SIGMA_K if chosen else EXAMPLE["k"]
    pct = config.PCA_PERCENTILE if chosen else EXAMPLE["pct"]
    k_hold = config.THREE_SIGMA_HOLD if chosen else EXAMPLE["k_hold"]
    pct_hold = config.PCA_HOLD if chosen else EXAMPLE["pct_hold"]
    m = te.SAMPLE_MINUTES
    train = te.load("d00.dat")
    x = te.load(f"d{CHART_FAULT:02d}_te.dat")[:320]  # 8 h normal, then 8 h of the fault
    hours = np.arange(len(x)) * te.SAMPLE_MINUTES / 60
    onset = te.TEST_FAULT_ONSET * te.SAMPLE_MINUTES / 60
    alarms = {
        f"Per-tag limits\n(k={k}, hold {k_hold * m} min)": te.on_delay(te.ThreeSigma(k=k).fit(train).flags(x), k_hold),
        f"PCA\n({pct}th pct, hold {pct_hold * m} min)": te.on_delay(te.PCAMonitor(percentile=pct).fit(train).flags(x), pct_hold),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.45})
    axes[0].plot(hours, x[:, REACTOR_TEMP], color="#3b6ea5", lw=1)
    axes[0].set_ylabel("Reactor temp (°C)")
    axes[0].set_title(f"Fault {CHART_FAULT}: {te.FAULT_NAMES[CHART_FAULT].lower()}, switched on at hour 8", loc="left", fontsize=11)
    for ax, (name, a) in zip(axes[1:], alarms.items()):
        ax.fill_between(hours, 0, a.astype(int), step="post", color="#d9534f", alpha=0.8, lw=0)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        first = te.first_alarm(a, te.TEST_FAULT_ONSET)
        events = te.alarm_events(a[: te.TEST_FAULT_ONSET])
        note = f"first alarm {(first - te.TEST_FAULT_ONSET) * te.SAMPLE_MINUTES} min after onset" if first is not None else "no alarm after onset"
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.text(0.0, 1.04, f"{events} nuisance alarms before the fault · {note}", transform=ax.transAxes, va="bottom", fontsize=8.5, color="#333")
    for ax in axes:
        ax.axvline(onset, color="#555", ls="--", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Hours into the run")
    label = "Settings chosen in config.py" if chosen else "Example settings, not yet chosen"
    fig.text(0.01, 0.005, f"Red = alarm sounding. {label}. Tennessee Eastman test run d{CHART_FAULT:02d}_te.", fontsize=8, color="#666")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.93, bottom=0.12)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        te.download()
    elif cmd == "tradeoff":
        tradeoff()
    elif cmd == "report":
        report()
    elif cmd == "chart":
        chart()
    else:
        sys.exit(__doc__)
