"""The operating point an operator would accept. Nathan sets this, from the dev table only.

Run `python run.py tradeoff` to see the dev-split table,
pick one row per detector, and fill in the values below. `run.py report` refuses to run
until both are set, so the report numbers are never used to make the choice.
"""

# Per-tag limits: alarm when any tag is more than K standard deviations from its normal mean.
THREE_SIGMA_K = 3.0

# PCA: alarm limit as a percentile of the training statistic.
PCA_PERCENTILE = 99.0

# On-delay in 3-minute samples for each detector: the flag must hold this long before the alarm sounds.
THREE_SIGMA_HOLD = 1  # 3 minutes, one sample
PCA_HOLD = 1

# One or two sentences, in your words, on why an operator would accept this point.
WHY = "Missing an upset is worse."  # Nathan, 2026-10-16, chosen from the dev table (rows A and E)
