"""Minimal per-ping detector of the main sound scattering layer (SSL).

This is the detector as described in the paper (section "Case-study implementation of edge
processing"), with nothing added. It follows the principle of the Peaks method of Peña et al. (2021,
Progress in Oceanography 198: 102669, https://doi.org/10.1016/j.pocean.2021.102669): a layer is a peak in
the smoothed Sv profile of a single ping, and its edges mark the layer's vertical extent.

Input: Sv in dB on an ascending, evenly spaced depth grid, with no NaN values. Putting the profiles on
such a grid is the paper's resampling step; the demo data are already averaged into 0.1 m bins (see
data/README.md), so the code does not resample. A NaN spreads over the whole smoothing window, about
+-4 sigma, so replace missing values first (for example with np.nan_to_num(sv, nan=-150.0)).

For each ping:

  1. Keep the profile within the depth range [min_depth, max_depth].
  2. Smooth it along depth with a Gaussian filter (sigma in metres, on the dB values).
  3. Remove strong echoes: the profile ends at the first value >= sv_max (seafloor, fish school),
     so that echo and everything below it are ignored.
  4. Find the layer peaks: local maxima with Sv >= sv_min (scipy.signal.find_peaks).
  5. Keep the main layer peak: the most prominent one.
  6. Its edges, where the profile has dropped by rel_height of the peak's prominence
     (scipy.signal.peak_widths), give the upper bound z_u and the lower bound z_l.
  7. Intensity I = (Sv_peak - sv_min) / (sv_max - sv_min), clipped to [0, 1].

Depths are ranges from the transducer in metres.
"""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# The settings used in the paper's case study, chosen by a parameter sweep.
DEFAULTS = dict(
    min_depth=2.5,     # m   top of the depth range (below the near-transducer echo)
    max_depth=100.0,   # m   bottom of the depth range (operational depth)
    sigma=3.1,         # m   Gaussian smoothing along depth
    sv_min=-84.0,      # dB  peaks below this are background
    sv_max=-52.5,      # dB  echoes from this up are removed
    rel_height=0.95,   # -   where the bounds are read, as a fraction of the peak's prominence
)


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


def find_layer(s, z, sv_min, sv_max, rel_height):
    """Steps 3-7 on one smoothed profile `s` on depths `z`. Returns a dict, with found=False and
    NaN values (intensity 0) when there is no layer."""
    out = dict(found=False, z_u=np.nan, z_l=np.nan, z_peak=np.nan, sv_peak=np.nan, intensity=0.0)
    strong = np.flatnonzero(s >= sv_max)
    s = s[:strong[0]] if strong.size else s                      # step 3
    peaks, props = find_peaks(s, height=sv_min, prominence=0)    # step 4
    if peaks.size == 0:
        return out
    k = peaks[np.argmax(props["prominences"])]                   # step 5
    _, _, left, right = peak_widths(s, [k], rel_height=rel_height)   # step 6
    bins = np.arange(z.size)
    out.update(found=True, z_u=float(np.interp(left[0], bins, z)), z_l=float(np.interp(right[0], bins, z)),
               z_peak=float(z[k]), sv_peak=float(s[k]),
               intensity=float(np.clip((s[k] - sv_min) / (sv_max - sv_min), 0, 1)))   # step 7
    return out


def detect(sv, depth, min_depth=2.5, max_depth=100.0, sigma=3.1, sv_min=-84.0, sv_max=-52.5,
           rel_height=0.95):
    """Run the detector on every ping of an echogram sv (n_ping, n_depth) in dB.
    Returns a dict of per-ping arrays: found, z_u, z_l, z_peak, sv_peak, intensity."""
    s, z = smooth(np.atleast_2d(sv), depth, min_depth, max_depth, sigma)
    rows = [find_layer(p, z, sv_min, sv_max, rel_height) for p in s]
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
