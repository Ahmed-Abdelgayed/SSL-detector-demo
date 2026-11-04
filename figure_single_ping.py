"""The paper's single-ping figure: the detector on one Thursday-morning ping, with the echogram around it.

    python figure_single_ping.py [output folder] [--case-study]

Panel A: half an hour of the Thursday CW echogram (6 June 2024, 09:00 to 09:30 UTC) with the per-ping bounds
(dots) and their running median over ssl.TUNED_MEDIAN pings (lines; no median with --case-study). Panel B:
one ping, with the raw and smoothed profiles, the thresholds, any masked strong echoes, the chosen layer peak,
the other candidates and the bounds. Uses the tuned settings (ssl.TUNED), or the case-study settings with
--case-study. Writes fig_ssl_single_ping.{png,pdf,tif} (PLOS: 7.5 in wide at most, Arial, 300+ dpi).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import ssl_detector as ssl

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "autonaut_thu_morning.npz")
START, END = np.datetime64("2024-06-06T09:00"), np.datetime64("2024-06-06T09:30")
PING = np.datetime64("2024-06-06T09:14:16")   # the ping in panel B (the one closest to this time)
ZMAX = 100              # m shown in both panels
VMIN, VMAX = -85, -55   # echogram colour range
PAD = 60                # pings run on each side of the window, so the running median has no edge effect
C_RAW, C_SM, C_PEAK, C_EDGE, C_MASK = "0.6", "#2a78d6", "#e34948", "#e34948", "#b23a48"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})


def candidates(s, z, p):
    """Every peak that passes the tests, as (depth, Sv, prominence, edge level), over all stretches."""
    out = []
    for a, b in ssl.stretches(s < p["sv_max"]):
        seg = np.maximum(s[a:b], p["sv_min"]) if p["floor"] else s[a:b]
        pk, pr = find_peaks(seg, height=p["sv_min"], prominence=p["min_prominence"],
                            width=p["min_width"] / (z[1] - z[0]), rel_height=p["rel_height"])
        out += [(z[a + k], s[a + k], pr["prominences"][j], pr["width_heights"][j]) for j, k in enumerate(pk)]
    return out


def main(out_dir=".", case_study=False):
    p, median = (ssl.DEFAULTS, 1) if case_study else (ssl.TUNED, ssl.TUNED_MEDIAN)
    d = ssl.load_data(DATA)
    w = np.flatnonzero((d["time"] >= START) & (d["time"] <= END))
    run = slice(max(0, w[0] - PAD), min(len(d["time"]), w[-1] + 1 + PAD))
    full = ssl.detect(d["sv"][run], d["depth"], **p)
    keep = slice(w[0] - run.start, w[-1] + 1 - run.start)
    res = {k: v[keep] for k, v in full.items()}
    filt = {k: ssl.running_median(full[k], median)[keep] for k in ("z_u", "z_l")}
    sv, t, depth = d["sv"][w], d["time"][w], d["depth"]
    i = int(np.argmin(np.abs(t - PING)))
    s, z = ssl.smooth(sv[i], depth, p["min_depth"], p["max_depth"], p["sigma"])
    r = ssl.find_layer(s, z, p["sv_min"], p["sv_max"], p["rel_height"], p["min_prominence"], p["min_width"],
                       p["floor"])
    cands = candidates(s, z, p)
    chosen = max(cands, key=lambda c: c[2])

    fig = plt.figure(figsize=(7.5, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[0.035, 1], wspace=0.08, hspace=0.04)

    # --- A: echogram with the per-ping bounds and their running median ---
    a = fig.add_subplot(gs[1, 0])
    tn = mdates.date2num(t)
    im = a.imshow(sv.T, aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX, interpolation="nearest",
                  extent=[tn[0], tn[-1], depth[-1], depth[0]])
    fd = res["found"]
    a.plot(tn[fd], res["z_u"][fd], ".", ms=0.8, color="white", alpha=0.5, lw=0)
    a.plot(tn[fd], res["z_l"][fd], ".", ms=0.8, color=C_EDGE, alpha=0.5, lw=0)
    a.plot(tn, filt["z_u"], color="white", lw=1.1, label=r"$z_{\mathrm{u}}$")
    a.plot(tn, filt["z_l"], color=C_EDGE, lw=1.1, label=r"$z_{\mathrm{l}}$")
    a.axvline(tn[i], color="white", lw=0.8, ls=":")
    a.annotate("ping in B", (tn[i], ZMAX - 2), xytext=(4, 0), textcoords="offset points", color="white",
               fontsize=7, va="bottom")
    a.set_ylim(ZMAX, 0)
    a.set_ylabel("Depth (m)")
    a.set_xlabel("Time (UTC), 6 June 2024")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    a.legend(loc="center left", framealpha=0.85, handlelength=1.5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 0]), orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(r"$S_v$ (dB re 1 m$^{-1}$, uncalibrated)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    a.text(0.01, 0.99, "A", transform=a.transAxes, fontweight="bold", fontsize=9, color="white", va="top")

    # --- B: the single ping ---
    b = fig.add_subplot(gs[1, 1], sharey=a)
    for z0, z1 in ((0, p["min_depth"]), (p["max_depth"], ZMAX)):
        b.axhspan(z0, z1, color="0.88", zorder=0)
    for dd in (p["min_depth"], p["max_depth"]):
        if dd < ZMAX:
            b.axhline(dd, color="0.35", lw=0.6, ls=":", zorder=1)
    b.tick_params(labelleft=False)
    b.axvspan(p["sv_min"], p["sv_max"], color="#eef4ec", zorder=0)
    for v in (p["sv_min"], p["sv_max"]):
        b.axvline(v, color="0.35", lw=0.8, ls="--", zorder=1)
    b.text(p["sv_min"], 1.0, r"$S_{v,\min}$", ha="center", va="bottom", fontsize=7, transform=b.get_xaxis_transform())
    b.text(p["sv_max"], 1.0, r"$S_{v,\max}$", ha="center", va="bottom", fontsize=7, transform=b.get_xaxis_transform())
    b.axhspan(r["z_u"], r["z_l"], color="#fde2e1", zorder=0)
    b.plot(sv[i], depth, color=C_RAW, lw=0.4, zorder=2)
    strong = s >= p["sv_max"]
    b.plot(np.where(strong, np.nan, s), z, color=C_SM, lw=1.4, zorder=3)
    b.plot(np.where(strong, s, np.nan), z, color=C_MASK, lw=2.6, alpha=0.6, zorder=3)
    for zc, svc, _, _ in cands:
        if zc != chosen[0]:
            b.plot(svc, zc, "o", ms=4.5, mfc="white", mec=C_SM, mew=1, zorder=4)
    b.plot([chosen[3]] * 2, [r["z_u"], r["z_l"]], color="0.15", lw=1.2, zorder=4)
    b.plot(r["sv_peak"], r["z_peak"], "o", ms=6, color=C_PEAK, mec="black", mew=0.6, zorder=5)
    xl = (-92, max(-50, 5 * np.ceil((np.nanmax(s) + 2) / 5)))   # room for the labels and any strong echo
    box = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8)
    for nm, zz, dy in ((r"$z_{\mathrm{u}}$", r["z_u"], -0.5), (r"$z_{\mathrm{l}}$", r["z_l"], 2.7)):
        b.axhline(zz, color=C_EDGE, lw=0.8, zorder=1)
        b.text(xl[1] - 0.8, zz + dy, f"{nm} = {r[('z_u' if 'u' in nm else 'z_l')]:.1f} m", ha="right", fontsize=7,
               bbox=box, zorder=6)
    b.annotate(r"$S_{v,\mathrm{peak}}$" + f" = {r['sv_peak']:.1f} dB", (r["sv_peak"], r["z_peak"]),
               xytext=(8, 0), textcoords="offset points", fontsize=7, va="center", bbox=box, zorder=6)
    b.text(xl[1] - 0.8, ZMAX - 2, f"$I$ = {r['intensity']:.2f}", fontsize=7.5, va="bottom", ha="right",
           bbox=box, zorder=6)
    b.set_xlim(*xl)
    b.set_xticks(np.arange(-90, xl[1] + 1, 10))
    b.set_xlabel(r"$S_v$ (dB re 1 m$^{-1}$, uncalibrated)")
    # direct labels instead of a legend (a legend box hides part of the profile)
    zs = np.arange(55.0, 78.0, 0.5)                               # a depth for the label away from the candidates
    zs = zs[np.argmax([min([abs(x - c[0]) for c in cands] + [99]) for x in zs])]
    k = np.searchsorted(z, zs)
    b.text(s[k] + 2, z[k], "smoothed", color=C_SM, fontsize=7, ha="left", va="center", bbox=box, zorder=6)
    if strong.any():
        zm = z[np.flatnonzero(strong)[-1]]
        b.text(p["sv_max"] + 1.5, zm + 2, "masked\n(strong echo)", fontsize=7, va="top", ha="left", color=C_MASK,
               bbox=box, zorder=6)
    others = [c for c in cands if c[0] != chosen[0]]
    if others:
        zc, svc = others[0][0], others[0][1]
        b.annotate("other\ncandidate" if len(others) == 1 else "other\ncandidates", (svc, zc), xytext=(8, 0),
                   textcoords="offset points", fontsize=7, va="center", color=C_SM, bbox=box, zorder=6)
    b.text(0.02, 0.99, "B", transform=b.transAxes, fontweight="bold", fontsize=9, va="top", zorder=7)
    b.grid(color="0.92", lw=0.4, zorder=0)

    for ext, kw in (("png", dict(dpi=300)), ("pdf", {}),
                    ("tif", dict(dpi=400, pil_kwargs={"compression": "tiff_lzw"}))):
        fig.savefig(os.path.join(out_dir, f"fig_ssl_single_ping.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print({k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()},
          f"{len(cands)} candidates, ping {t[i]}, window {t[0]} to {t[-1]} ({len(t)} pings)")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(args[0] if args else ".", "--case-study" in sys.argv)
