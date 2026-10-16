import numpy as np

import te


def test_on_delay_needs_consecutive_flags():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert te.on_delay(flags, 3).tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert te.on_delay(flags, 1).tolist() == flags.tolist()


def test_alarm_events_counts_rising_edges():
    assert te.alarm_events(np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)) == 3
    assert te.alarm_events(np.zeros(5, dtype=bool)) == 0


def test_first_alarm_at_or_after_onset():
    alarm = np.array([0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert te.first_alarm(alarm, 2) == 2
    assert te.first_alarm(alarm, 3) == 6
    assert te.first_alarm(np.zeros(5, dtype=bool), 2) is None


def test_load_untransposes_training_file(tmp_path):
    np.savetxt(tmp_path / "d00.dat", np.arange(52 * 500, dtype=float).reshape(52, 500))
    assert te.load("d00.dat", tmp_path).shape == (500, 52)


def test_detectors_flag_a_step_and_not_normal_data():
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(52, 52))
    normal = rng.normal(size=(2000, 52)) @ mix
    fault = rng.normal(size=(200, 52)) @ mix
    fault[:, 5] += 8 * normal[:, 5].std()
    for det in (te.ThreeSigma(k=4).fit(normal), te.PCAMonitor(percentile=99.9).fit(normal)):
        fresh = rng.normal(size=(2000, 52)) @ mix
        assert det.flags(fresh).mean() < 0.05
        assert det.flags(fault).mean() > 0.9


def test_calibration_run_sets_the_limits():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 52))
    wider = 2 * rng.normal(size=(500, 52))
    own = te.PCAMonitor(percentile=99).fit(train)
    cal = te.PCAMonitor(percentile=99).fit(train, calibration=wider)
    assert cal.spe_limit > own.spe_limit
    assert cal.flags(wider).mean() < own.flags(wider).mean()


def test_ewma_follows_a_step():
    out = te.ewma(np.array([0.0, 1, 1, 1]), 0.5)
    assert out.tolist() == [0.0, 0.5, 0.75, 0.875]
    assert te.ewma(np.array([3.0, 4.0]), 1.0).tolist() == [3.0, 4.0]


def test_normal_zone_drops_an_upset_and_keeps_normal():
    rng = np.random.default_rng(2)
    mix = rng.normal(size=(52, 52))
    history = rng.normal(size=(1500, 52)) @ mix
    history[700:800, 3] += 15 * history[:, 3].std()
    keep = te.find_normal_zone(history)
    assert not keep[700:800].any()
    assert keep[:600].mean() > 0.9


def test_tag_limits_cover_normal_and_flag_outside():
    normal = np.random.default_rng(3).normal(size=(5000, 4))
    lim = te.TagLimits().fit(normal)
    assert lim.flags(normal).mean() < 0.01
    assert lim.flags(np.array([[0, 0, 9, 0]])).all()
    lim.tags = np.array([0, 1])
    assert not lim.flags(np.array([[0, 0, 9, 0]])).any()
