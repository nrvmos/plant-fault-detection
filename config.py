"""Alarm settings, chosen by Nathan from the dev split only.

`run.py report` refuses to run until FINAL_PCA_PERCENTILE is set, so the report numbers
are never used to make the choice.

History
- Version 1 (2026-10-16): the loosest textbook limits on both detectors, because
  "Missing an upset is worse." Scored on the report split: 17 to 31 nuisance alarms a shift.
- Version 2: the same rule, but with the alarm limits set from a separate normal run.
"""

# Limit for the cleaned-up PCA detector, as a percentile of a separate normal run: 99.5 or 99.9.
FINAL_PCA_PERCENTILE = 99.5  # Balanced, picked by Nathan 2026-10-16 from the dev split

# The flag must hold this many 3-minute samples before the alarm sounds.
HOLD_SAMPLES = 1

WHY = "Missing an upset is worse."
