"""Minimal per-ping detector of the main sound scattering layer (SSL).

This is the detector as described in the paper (section "Case-study implementation of edge
processing"), with three optional peak tests (min_prominence, min_width, floor) that are off in the
case-study settings and on in the tuned ones. It follows the principle of the Peaks method of Peña et al. (2021,
Progress in Oceanography 198: 102669, https://doi.org/10.1016/j.pocean.2021.102669): a layer is a peak in
the smoothed Sv profile of a single ping, and its edges mark the layer's vertical extent.

Input: Sv in dB on an ascending, evenly spaced depth grid, with no NaN values. Putting the profiles on
such a grid is the paper's resampling step; the demo data are already averaged into 0.1 m bins (see
data/README.md), so the code does not resample. A NaN spreads over the whole smoothing window, about
+-4 sigma, so replace missing values first (for example with np.nan_to_num(sv, nan=-150.0)).

For each ping:

  1. Keep the profile within the depth range [min_depth, max_depth].
  2. Smooth it along depth with a Gaussian filter (sigma in metres, on the dB values).
  3. Mask strong echoes: values >= sv_max (seafloor, fish school, near-transducer echo) are left out,
     and the layer is searched in each stretch of the profile between them.
  4. Find the layer peaks with scipy.signal.find_peaks: local maxima with Sv >= sv_min that stand at
     least min_prominence dB above their surroundings and are at least min_width m wide (measured at
     rel_height). With floor, the profile is first raised to sv_min, so the bounds stop there.
  5. Keep the main layer peak: the most prominent one over all stretches.
  6. Its edges, where the profile has dropped by rel_height of the peak's prominence (as
     scipy.signal.peak_widths measures them), give the upper bound z_u and the lower bound z_l.
  7. Intensity I = (Sv_peak - sv_min) / (sv_max - sv_min), clipped to [0, 1].

Depths are ranges from the transducer in metres.
"""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# The settings used in the paper's case study, chosen by a parameter sweep.
DEFAULTS = dict(
    min_depth=2.5,       # m   top of the depth range (below the near-transducer echo)
    max_depth=100.0,     # m   bottom of the depth range (operational depth)
    sigma=3.1,           # m   Gaussian smoothing along depth
    sv_min=-84.0,        # dB  peaks below this are background
    sv_max=-52.5,        # dB  echoes from this up are masked
    rel_height=0.95,     # -   where the bounds are read, as a fraction of the peak's prominence
    min_prominence=0.0,  # dB  how far a peak must stand above its surroundings
    min_width=0.0,       # m   minimum layer thickness, measured at rel_height
    floor=False,         # -   raise the profile to sv_min, so the bounds stop there
)

# Settings tuned by hand on other stretches of the same survey (among them the Thursday CW data, not included
# here): good on most stretches, but not on Wednesday night (see the notebook). Use them with a running median
# over 23 pings (running_median(x, TUNED_MEDIAN)).
TUNED = dict(min_depth=2.5, max_depth=81.5, sigma=2.0, sv_min=-80.0, sv_max=-60.5, rel_height=0.95,
             min_prominence=1.0, min_width=1.0, floor=True)
TUNED_MEDIAN = 23


def load_data(path="data/autonaut_wed_night.npz"):
    """The demo echogram: sv (n_ping, n_depth) in dB, depth (m), time (UTC), lat, lon."""
    with np.load(path) as f:
        return dict(sv=f["sv_cdb"] / 100.0, depth=f["depth"], time=f["time"],
                    lat=f["lat"], lon=f["lon"])


def smooth(sv, depth, min_depth, max_depth, sigma):
    """Steps 1-2 on one ping or a whole echogram. Returns (smoothed Sv, depths kept)."""
    keep = (depth >= min_depth) & (depth <= max_depth)
    z = depth[keep]
    s = np.asarray(sv, float)[..., keep]
    if sigma > 0 and z.size > 1:
        s = gaussian_filter1d(s, sigma / (z[1] - z[0]), axis=-1, mode="nearest")
    return s, z


def find_layer(s, z, sv_min, sv_max, rel_height, min_prominence=0.0, min_width=0.0, floor=False):
    """Steps 3-7 on one smoothed profile `s` on depths `z`. Returns a dict, with found=False and
    NaN values (intensity 0) when there is no layer."""
    out = dict(found=False, z_u=np.nan, z_l=np.nan, z_peak=np.nan, sv_peak=np.nan, intensity=0.0)
    dz = z[1] - z[0] if z.size > 1 else 1.0
    best = None                                                  # (prominence, start, left, right, peak)
    for a, b in stretches(s < sv_max):                           # step 3
        seg = np.maximum(s[a:b], sv_min) if floor else s[a:b]
        peaks, props = find_peaks(seg, height=sv_min, prominence=min_prominence,    # step 4
                                  width=min_width / dz, rel_height=rel_height)
        if peaks.size:
            j = np.argmax(props["prominences"])                  # step 5
            if best is None or props["prominences"][j] > best[0]:
                best = (props["prominences"][j], a, props["left_ips"][j], props["right_ips"][j], peaks[j])
    if best is None:
        return out
    _, a, left, right, k = best
    bins = np.arange(z.size)                                     # step 6
    sv_peak = s[a + k]
    out.update(found=True, z_u=float(np.interp(a + left, bins, z)), z_l=float(np.interp(a + right, bins, z)),
               z_peak=float(z[a + k]), sv_peak=float(sv_peak),
               intensity=float(np.clip((sv_peak - sv_min) / (sv_max - sv_min), 0, 1)))   # step 7
    return out


def stretches(keep):
    """Start and end (exclusive) of each run of True in the boolean array `keep`."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def detect(sv, depth, min_depth=2.5, max_depth=100.0, sigma=3.1, sv_min=-84.0, sv_max=-52.5,
           rel_height=0.95, min_prominence=0.0, min_width=0.0, floor=False):
    """Run the detector on every ping of an echogram sv (n_ping, n_depth) in dB.
    Returns a dict of per-ping arrays: found, z_u, z_l, z_peak, sv_peak, intensity."""
    s, z = smooth(np.atleast_2d(sv), depth, min_depth, max_depth, sigma)
    rows = [find_layer(p, z, sv_min, sv_max, rel_height, min_prominence, min_width, floor) for p in s]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}


def running_median(x, n):
    """Centred running median over n pings (n odd), ignoring pings without a layer (NaN).
    The paper notes that the per-ping bounds are noisy and need filtering over pings."""
    x = np.asarray(x, float)
    h = int(n) // 2
    if h < 1:
        return x.copy()
    pad = np.concatenate([np.full(h, np.nan), x, np.full(h, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # windows with no layer at all
        out = np.nanmedian(win, axis=1)
    out[np.isnan(x)] = np.nan
    return out
