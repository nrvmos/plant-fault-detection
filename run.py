"""Usage:
    python run.py download   # about 18 MB of benchmark data into data/
    python run.py steps      # dev split: the noise-reduction steps, one at a time
    python run.py tradeoff   # dev split: every per-tag and PCA setting tried
    python run.py suggest    # find the normal zone in the history and suggest alarm points per tag
    python run.py report     # report split, after config.py is filled in
    python run.py chart      # writes docs/fault-example.png
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import te  # noqa: E402

THREE_SIGMA_KS = [3.0, 3.5, 4.0, 5.0]
PCA_PERCENTILES = [99.0, 99.5, 99.9]
ON_DELAYS = [1, 2, 3, 5]
# Faults 3, 9 and 15 barely move the measured tags; they're shown but left out of the medians.
HARD_TO_SEE = {3, 9, 15}


def summary(result: dict) -> tuple[float, float, int, int, float]:
    """Nuisance alarms per shift, median minutes to alarm, faults caught within 1 h, clear faults, alarm-on share."""
    rows = [r for r in result["faults"] if r["fault"] not in HARD_TO_SEE]
    delays = [r["delay_min"] if r["delay_min"] is not None else np.inf for r in rows]
    caught_in_hour = sum(d <= 60 for d in delays)
    return result["false_alarms_per_shift"], float(np.median(delays)), caught_in_hour, len(rows), result["alarm_on_share"]


def row(name: str, result: dict) -> str:
    fa, med, caught, n, share = summary(result)
    return f"| {name} | {fa:.2f} | {share:.1%} | {result['chance_hour_alarms']:.0%} | {med:.0f} min | {caught}/{n} |"


HEADER = "| Step | Nuisance alarms per shift | Alarm on in normal running | Normal hours with an alarm | Median minutes to alarm | Faults caught within 1 h |\n|---|---:|---:|---:|---:|---:|"


def textbook_pca(train):
    return te.PCAMonitor(percentile=99.0).fit(train)


def calibrated_pca(train, cal, pct, smoothing=1.0):
    return te.PCAMonitor(percentile=pct, smoothing=smoothing).fit(train, calibration=cal)


def steps() -> None:
    train, cal = te.load("d00.dat"), te.calibration_data()
    print("Dev split: odd-numbered faults, scored on the 8 h before and the 40 h after each fault starts.\n")
    print(HEADER)
    print(row("1. Textbook PCA: limits from the training day (99th pct)", te.score(textbook_pca(train), 1, "dev")))
    print(row("2. Limits from a separate normal run (99th pct)", te.score(calibrated_pca(train, cal, 99.0), 1, "dev")))
    print(row("3. Same, limit raised to the 99.5th pct", te.score(calibrated_pca(train, cal, 99.5), 1, "dev")))
    print(row("4. Same, limit raised to the 99.9th pct", te.score(calibrated_pca(train, cal, 99.9), 1, "dev")))
    print("\nTried, and not kept:\n")
    print(HEADER)
    print(row("Smoothing both statistics (EWMA 0.3), 99.9th pct", te.score(calibrated_pca(train, cal, 99.9, 0.3), 1, "dev")))
    print(row("6-minute hold, 99.5th pct", te.score(calibrated_pca(train, cal, 99.5), 2, "dev")))
    name, result = best_per_tag(train)
    print(row(f"Per-tag limits, quietest that still catches 7/8 in 1 h ({name})", result))


def best_per_tag(train) -> tuple[str, dict]:
    """The per-tag setting with the fewest nuisance alarms that still catches 7 of 8 dev faults within an hour."""
    best = None
    for k in THREE_SIGMA_KS:
        det = te.ThreeSigma(k=k).fit(train)
        for d in ON_DELAYS:
            r = te.score(det, d, "dev")
            fa, _, caught, _, _ = summary(r)
            if caught >= 7 and (best is None or fa < best[0]):
                best = (fa, f"k={k}, hold {d * te.SAMPLE_MINUTES} min", (k, d), r)
    return best[1], best[3]


def tradeoff() -> None:
    train, cal = te.load("d00.dat"), te.calibration_data()
    print(f"{'detector':<34}{'must hold':>9}{'nuisance/shift':>16}{'median delay (min)':>20}{'caught within 1 h':>19}")
    groups = [
        ("per-tag k", [(k, te.ThreeSigma(k=k).fit(train)) for k in THREE_SIGMA_KS]),
        ("PCA, training-day limits, pct", [(p, te.PCAMonitor(percentile=p).fit(train)) for p in PCA_PERCENTILES]),
        ("PCA, separate-run limits, pct", [(p, calibrated_pca(train, cal, p)) for p in PCA_PERCENTILES]),
    ]
    for name, detectors in groups:
        for setting, det in detectors:
            for d in ON_DELAYS:
                fa, med, caught, n, _ = summary(te.score(det, d, "dev"))
                print(f"{name + ' ' + str(setting):<34}{d * te.SAMPLE_MINUTES:>8}m{fa:>16.2f}{med:>20.0f}{caught:>15}/{n}")


def report() -> None:
    if config.FINAL_PCA_PERCENTILE is None:
        sys.exit("Set FINAL_PCA_PERCENTILE in config.py from the dev split first.")
    train, cal = te.load("d00.dat"), te.calibration_data()
    tag_name, _ = best_per_tag(train)
    k, d = next((k, d) for k in THREE_SIGMA_KS for d in ON_DELAYS if f"k={k}, hold {d * te.SAMPLE_MINUTES} min" == tag_name)
    results = {
        "Before: textbook PCA": te.score(textbook_pca(train), config.HOLD_SAMPLES),
        f"After: PCA, separate-run limits ({config.FINAL_PCA_PERCENTILE}th pct)": te.score(
            calibrated_pca(train, cal, config.FINAL_PCA_PERCENTILE), config.HOLD_SAMPLES
        ),
        f"Per-tag limits ({tag_name})": te.score(te.ThreeSigma(k=k).fit(train), d),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        suggested = suggest()
    results[f"Per-tag alarm points suggested from history ({len(suggested.tags)} tags)"] = te.score(suggested, config.HOLD_SAMPLES)
    print("Report split: even-numbered faults + the last 24 h of the normal test run.\n")
    print(HEADER.replace("Step", "Detector"))
    for name, r in results.items():
        print(row(name, r))
    print("\n| Fault | What changes | " + " | ".join(results) + " |")
    print("|---|---|" + "---:|" * len(results))
    for i, f in enumerate(next(iter(results.values()))["faults"]):
        cells = []
        for r in results.values():
            delay = r["faults"][i]["delay_min"]
            cells.append("not caught" if delay is None else f"{delay} min")
        print(f"| {f['fault']} | {f['name']} | " + " | ".join(cells) + " |")


def upset_blocks(keep: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of samples the history reader dropped as upsets."""
    blocks, start = [], None
    for i, k in enumerate(np.append(keep, True)):
        if not k and start is None:
            start = i
        elif k and start is not None:
            blocks.append((start, i))
            start = None
    return blocks


def suggest(out: Path = Path(__file__).parent / "docs" / "suggested-alarms.csv", min_upsets: int = 2) -> te.TagLimits:
    history, truth, source = te.build_history()
    keep = te.find_normal_zone(history)
    found = ~keep
    blocks = upset_blocks(keep)
    clear = ~np.isin(source, list(HARD_TO_SEE))
    hours = len(history) * te.SAMPLE_MINUTES / 60
    print(f"History: {hours:.0f} h, {truth.mean():.0%} of it inside an upset (labels used only for this check).")
    coverage = {f: found[(source == f) & truth].mean() for f in te.DEV_FAULTS}
    clear_hits = [f for f, c in coverage.items() if c > 0.25]
    false_blocks = sum(not truth[a:b].any() for a, b in blocks)
    print(f"Upset periods found: {len(blocks)}, {false_blocks} of them in normal running.")
    print(f"Real upsets clearly found (over a quarter of the upset flagged): {len(clear_hits)} of {len(te.DEV_FAULTS)}: {clear_hits}.")
    print(f"They cover {found[truth & clear].mean():.0%} of the upset samples from the clear faults.")
    print(f"Normal zone kept: {keep[~truth].mean():.0%} of the truly normal samples; {(keep & truth).sum() / keep.sum():.1%} of it is really upset "
          f"({(keep & truth & clear).sum() / keep.sum():.1%} if faults 3, 9 and 15, which barely move any tag, are left out).\n")

    limits = te.TagLimits().fit(history[keep])
    flags = limits.tag_flags(history)
    normal_hours = keep.sum() * te.SAMPLE_MINUTES / 60
    rows = []
    for j in range(history.shape[1]):
        nuisance = te.alarm_events(flags[keep, j]) / normal_hours * te.SHIFT_HOURS
        fired = sum(flags[a:b, j].any() for a, b in blocks)
        rows.append((j, nuisance, fired, fired >= min_upsets))
    limits.tags = np.array([j for j, _, _, k in rows if k])
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        fh.write("tag,name,suggested_low,suggested_high,nuisance_alarms_per_shift_in_history,upsets_fired_in,keep_alarm\n")
        for j, nuisance, fired, k in rows:
            fh.write(f"{te.TAG_IDS[j]},{te.TAG_NAMES[j]},{limits.low[j]:.4g},{limits.high[j]:.4g},{nuisance:.2f},{fired},{'yes' if k else 'no'}\n")
    print(f"Suggested alarm points for all 52 tags written to {out.name}. Keep an alarm if it fired in at least {min_upsets} of the {len(blocks)} upsets: {len(limits.tags)} tags.\n")
    print("| Tag | Name | Suggested low | Suggested high | Upsets it fired in | Keep |\n|---|---|---:|---:|---:|---|")
    for j, nuisance, fired, k in sorted(rows, key=lambda r: -r[2])[:12]:
        print(f"| {te.TAG_IDS[j]} | {te.TAG_NAMES[j]} | {limits.low[j]:.4g} | {limits.high[j]:.4g} | {fired} | {'yes' if k else 'no'} |")
    return limits


CHART_FAULT = 11  # a dev-split fault, so the chart never shows report data
REACTOR_TEMP = 8  # XMEAS(9), reactor temperature


def chart(out: Path = Path(__file__).parent / "docs" / "fault-example.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, cal = te.load("d00.dat"), te.calibration_data()
    pct = config.FINAL_PCA_PERCENTILE or 99.5
    x = te.load(f"d{CHART_FAULT:02d}_te.dat")[:320]  # 8 h normal, then 8 h of the fault
    hours = np.arange(len(x)) * te.SAMPLE_MINUTES / 60
    onset = te.TEST_FAULT_ONSET * te.SAMPLE_MINUTES / 60
    alarms = {
        "Before\n(textbook limits)": te.on_delay(textbook_pca(train).flags(x), config.HOLD_SAMPLES),
        f"After\n(separate-run limits, {pct}th)": te.on_delay(calibrated_pca(train, cal, pct).flags(x), config.HOLD_SAMPLES),
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
    label = "Final setting from config.py" if config.FINAL_PCA_PERCENTILE else "Example setting, not yet chosen"
    fig.text(0.01, 0.005, f"Red = PCA alarm sounding. {label}. Tennessee Eastman test run d{CHART_FAULT:02d}_te.", fontsize=8, color="#666")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.93, bottom=0.12)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    commands = {"download": te.download, "steps": steps, "tradeoff": tradeoff, "suggest": suggest, "report": report, "chart": chart}
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in commands:
        sys.exit(__doc__)
    commands[cmd]()
