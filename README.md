# SSL detector demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Ahmed-Abdelgayed/SSL-detector-demo/HEAD?labpath=ssl_detector_demo.ipynb)

Supporting material for:

> Barstein K, Rogge R, Olaisen A, Abdelgayed A, Dunn M, Stevenson-Jones R, Davies E, Grimsmo L, Ludvigsen M.
> *A Remote Operations Center Framework for Heterogeneous Robotic Zooplankton Sampling: System Design and
> Field Case Study.*

A minimal version of the per-ping sound scattering layer (SSL) detector described in the paper, with two
small datasets (a simple case and a harder one) and a notebook that explains each setting and lets you tune
them.

## Contents

| File | |
|---|---|
| `ssl_detector_demo.ipynb` | The notebook: the data, what each setting does, and an interactive tuning panel |
| `ssl_detector.py` | The detector, in one short file |
| `data/autonaut_tue_night.npz` | 2.5 hours of AutoNaut echosounder data: one layer near 20 m, seabed in range (see `data/README.md`) |
| `data/autonaut_wed_night.npz` | 4.3 hours: a layer moving between 45 and 90 m, deeper water |
| `tests/` | Tests of the detector |

## The method

The detector follows the principle of the Peaks method of Peña et al. (2021): a sound scattering layer
shows up in the vertical Sv profile of a single ping as a peak that stands above weaker scattering on both
sides, and the edges of that peak mark the layer's vertical extent. The detector finds that peak and its
edges in every ping on its own, so it is light enough to run on board. The peaks and edges are found with
SciPy (Virtanen et al. 2020).

The input is an echogram of Sv in dB on a fixed vertical grid (here 0.1 m bins, ascending, with no NaN
values). Putting the profiles on such a grid is the paper's resampling step, so one set of settings in metres
works for any echosounder. For each ping:

1. **Depth range.** Keep the profile from `min_depth` to `max_depth`.
2. **Smoothing.** Smooth it along depth with a Gaussian filter of standard deviation `sigma` (in metres,
   applied to the dB values).
3. **Strong echoes.** Mask every smoothed value at or above `sv_max` (seafloor, dense fish school, the echo
   near the transducer). The masked values are left out, and the layer is searched in each stretch of the
   profile between them, so a layer above or below a school or the seabed is still found.
4. **Candidate peaks.** In each stretch, find the local maxima (`scipy.signal.find_peaks`) with Sv at or
   above `sv_min`, a prominence of at least `min_prominence` dB, and a width of at least `min_width` m
   (measured at `rel_height`). With `floor`, the profile is first raised to `sv_min`, so the bounds stop where
   it crosses `sv_min`.
5. **Main layer.** Keep the most prominent candidate over all stretches.
6. **Bounds.** The upper bound `z_u` and the lower bound `z_l` are where the profile has dropped below the
   peak by `rel_height` times its prominence (as `scipy.signal.peak_widths` measures them). The peak depth
   is `z_peak` and its smoothed Sv is `Sv_peak`.
7. **Intensity.** `I = (Sv_peak - sv_min) / (sv_max - sv_min)`, clipped to 0 to 1. It says how strong the
   layer is between the two thresholds; it is not a probability.

A ping with no candidate has no layer (`found` is False, the depths are NaN and `I` is 0). Single pings are
noisy, so the per-ping bounds are noisy as well; as the paper notes, they need filtering over pings. The
notebook uses a running median (`ssl.running_median`), which is not part of the detector.

### Settings

| Setting | Case study (`ssl.DEFAULTS`) | Tuned (`ssl.TUNED`) | |
|---|---|---|---|
| `min_depth`, `max_depth` | 2.5 m, 100 m | 2.5 m, 81.5 m | Depth range searched |
| `sigma` | 3.1 m | 2.0 m | Width of the Gaussian smoothing |
| `sv_min` | −84 dB | −80 dB | Weaker peaks are background; bottom of the intensity scale |
| `sv_max` | −52.5 dB | −60.5 dB | Stronger echoes are masked; top of the intensity scale |
| `rel_height` | 0.95 | 0.95 | Where the bounds are read, as a fraction of the peak's prominence |
| `min_prominence` | 0 dB (off) | 1 dB | How far a peak must stand above its surroundings |
| `min_width` | 0 m (off) | 1 m | Minimum layer thickness |
| `floor` | off | on | Bounds stop at `sv_min` |
| running median | | 23 pings (`ssl.TUNED_MEDIAN`) | Filter over pings, after the detector |

- **Case study:** the settings of the paper's case study, chosen by a parameter sweep.
- **Tuned:** tuned by hand on other stretches of the same survey (among them the Thursday CW data, which are
  not included here). They work well on most stretches, but not on Wednesday night: their 81.5 m bottom
  cuts off the layer there when it sinks (see section 9 of the notebook).

The echosounder was not calibrated, so these Sv thresholds apply to this instrument only.

```python
import ssl_detector as ssl

d = ssl.load_data("data/autonaut_tue_night.npz")   # sv (n_ping, n_depth) in dB, depth, time, lat, lon
r = ssl.detect(d["sv"], d["depth"], **ssl.TUNED)
z_u = ssl.running_median(r["z_u"], ssl.TUNED_MEDIAN)
r["z_u"], r["z_l"], r["z_peak"], r["sv_peak"], r["intensity"], r["found"]
```

## Running it

In the browser, with no installation: use the Binder badge above.

On your own computer, with Python 3.10 or newer:

```bash
pip install -r requirements.txt
jupyter lab ssl_detector_demo.ipynb
```

Then run all cells (Run > Run All Cells). The tuning panel at the end needs a running Python kernel, so it
does not show on GitHub's static view of the notebook.

Run the tests with `pip install pytest`, then `pytest`.

## References

- Peña M, Munuera-Fernández I, Nogueira E, González-Quirós R (2021). First recording of a bathypelagic
  deep scattering layer in the Bay of Biscay. *Progress in Oceanography* 198: 102669.
  https://doi.org/10.1016/j.pocean.2021.102669
- Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0: fundamental algorithms for scientific
  computing in Python. *Nature Methods* 17: 261–272. https://doi.org/10.1038/s41592-019-0686-2
- Lee W-J, Setiawan L, Tuguinay C, Mayorga E, Staneva V (2024). Interoperable and scalable echosounder data
  processing with Echopype. *ICES Journal of Marine Science* 81(10): 1941–1951.
  https://doi.org/10.1093/icesjms/fsae133 (used to compute Sv for the demo data)

## Licence

The code is under the MIT licence (`LICENSE`). The data are under CC BY 4.0 (`data/LICENSE`).
