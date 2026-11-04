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


def test_strong_echo_ends_the_profile():
    # the layer lies below a strong echo at 30 m, so it is removed with it
    r = ssl.detect(profile(peak_z=50.0, seabed_z=30.0), DEPTH)
    assert not r["found"][0]
    r = ssl.detect(profile(peak_z=20.0, seabed_z=30.0), DEPTH)
    assert r["found"][0] and abs(r["z_peak"][0] - 20.0) < 0.5


def test_depth_range():
    r = ssl.detect(profile(peak_z=40.0), DEPTH, min_depth=50.0)
    assert not r["found"][0]


def test_running_median_skips_missing_pings():
    x = np.array([10.0, np.nan, 12.0, 11.0, 50.0, 11.0, 12.0])
    m = ssl.running_median(x, 3)
    assert np.isnan(m[1]) and m[4] == 11.0


def test_demo_data_with_the_paper_settings():
    d = ssl.load_data(os.path.join(ROOT, "data", "autonaut_wed_night.npz"))
    assert d["sv"].shape == (5200, 996)
    r = ssl.detect(d["sv"], d["depth"], **ssl.DEFAULTS)
    assert int(r["found"].sum()) == 4223
