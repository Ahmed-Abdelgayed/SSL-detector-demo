import os

import numpy as np

import ssl_detector as ssl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPTH = np.arange(0.05, 100, 0.1)


def profile(peak_z=40.0, width=4.0, seabed_z=None):
    """Background -90 dB with a Gaussian layer of -70 dB at peak_z and an optional seabed."""
    sv = -90 + 20 * np.exp(-0.5 * ((DEPTH - peak_z) / width) ** 2)
    if seabed_z is not None:
        sv[DEPTH >= seabed_z] = -20.0
    return sv


def test_finds_the_layer():
    r = ssl.detect(profile(), DEPTH)
    assert r["found"][0]
    assert abs(r["z_peak"][0] - 40.0) < 0.1
    assert r["z_u"][0] < 40.0 < r["z_l"][0]
    assert 0 < r["intensity"][0] < 1


def test_wider_rel_height_gives_wider_bounds():
    a = ssl.detect(profile(), DEPTH, rel_height=0.5)
    b = ssl.detect(profile(), DEPTH, rel_height=0.95)
    assert b["z_u"][0] < a["z_u"][0] and b["z_l"][0] > a["z_l"][0]


def test_layer_below_sv_min_is_background():
    r = ssl.detect(profile(), DEPTH, sv_min=-65.0)
    assert not r["found"][0] and r["intensity"][0] == 0.0


def school(peak_z, top, bottom):
    """A layer at peak_z with a strong echo (school) from top to bottom."""
    sv = profile(peak_z=peak_z)
    sv[(DEPTH >= top) & (DEPTH <= bottom)] = -30.0
    return sv


def test_strong_echo_is_masked():
    # a layer below a school is still found, and its bounds stay below the masked school
    # (a school must stay above sv_max after smoothing: a thin one is smoothed below it and is not masked)
    sv = school(55.0, 20.0, 35.0)
    s, z = ssl.smooth(sv, DEPTH, 2.5, 100.0, 3.1)
    top_free = z[(s >= -52.5)].max()
    r = ssl.detect(sv, DEPTH)
    assert r["found"][0] and abs(r["z_peak"][0] - 55.0) < 0.5 and r["z_u"][0] > top_free
    # a layer above a strong seabed is found, and nothing below the seabed top is kept as its bound
    r = ssl.detect(profile(peak_z=20.0, seabed_z=60.0), DEPTH)
    assert r["found"][0] and abs(r["z_peak"][0] - 20.0) < 0.5 and r["z_l"][0] < 60.0


def test_layer_inside_a_strong_echo_is_not_a_layer():
    r = ssl.detect(school(50.0, 30.0, 70.0), DEPTH)
    assert not (r["found"][0] and 30.0 <= r["z_peak"][0] <= 70.0)


def test_stretches():
    keep = np.array([True, True, False, True, False, False, True])
    assert ssl.stretches(keep) == [(0, 2), (3, 4), (6, 7)]


def test_depth_range():
    r = ssl.detect(profile(peak_z=40.0), DEPTH, min_depth=50.0)
    assert not r["found"][0]


def test_running_median_skips_missing_pings():
    x = np.array([10.0, np.nan, 12.0, 11.0, 50.0, 11.0, 12.0])
    m = ssl.running_median(x, 3)
    assert np.isnan(m[1]) and m[4] == 11.0


def test_demo_data_with_the_case_study_settings():
    for name, shape, n_found in (("tue_night", (2931, 996), 2931), ("wed_night", (5200, 996), 5199)):
        d = ssl.load_data(os.path.join(ROOT, "data", f"autonaut_{name}.npz"))
        assert d["sv"].shape == shape
        r = ssl.detect(d["sv"], d["depth"], **ssl.DEFAULTS)
        assert int(r["found"].sum()) == n_found


def two_layers(weak_z, strong_z):
    """A weak and a strong layer with a wide school (masked) between them."""
    sv = -90 + 10 * np.exp(-0.5 * ((DEPTH - weak_z) / 3) ** 2) + 25 * np.exp(-0.5 * ((DEPTH - strong_z) / 3) ** 2)
    sv[(DEPTH >= 40) & (DEPTH <= 55)] = -30.0
    return sv


def test_most_prominent_peak_over_all_stretches_wins():
    for weak, strong in ((20.0, 75.0), (75.0, 20.0)):
        r = ssl.detect(two_layers(weak, strong), DEPTH)
        assert r["found"][0] and abs(r["z_peak"][0] - strong) < 0.5


def test_min_prominence_and_min_width():
    sv = profile()
    assert not ssl.detect(sv, DEPTH, min_prominence=30.0)["found"][0]     # the layer stands about 20 dB out
    assert not ssl.detect(sv, DEPTH, min_width=50.0)["found"][0]          # and is narrower than 50 m
    assert ssl.detect(sv, DEPTH, min_prominence=5.0, min_width=2.0)["found"][0]


def test_floor_stops_the_bounds_at_sv_min():
    sv = profile()                                                        # background -90, peak -70
    a = ssl.detect(sv, DEPTH, sv_min=-80.0, rel_height=1.0)
    b = ssl.detect(sv, DEPTH, sv_min=-80.0, rel_height=1.0, floor=True)
    assert b["z_u"][0] > a["z_u"][0] and b["z_l"][0] < a["z_l"][0]
    s, z = ssl.smooth(sv, DEPTH, 2.5, 100.0, 3.1)
    above = z[s >= -80.0]
    assert abs(b["z_u"][0] - above.min()) < 0.2 and abs(b["z_l"][0] - above.max()) < 0.2


def test_tuned_settings_on_the_demo_data():
    d = ssl.load_data(os.path.join(ROOT, "data", "autonaut_tue_night.npz"))
    r = ssl.detect(d["sv"], d["depth"], **ssl.TUNED)
    assert r["found"].all()
