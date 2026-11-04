# SSL detector demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Ahmed-Abdelgayed/SSL-detector-demo/HEAD?labpath=ssl_detector_demo.ipynb)

Supporting material for:

> Barstein K, Rogge R, Olaisen A, Abdelgayed A, Dunn M, Stevenson-Jones R, Davies E, Grimsmo L, Ludvigsen M.
> *A Remote Operations Center Framework for Heterogeneous Robotic Zooplankton Sampling: System Design and
> Field Case Study.*

A minimal version of the per-ping sound scattering layer (SSL) detector described in the paper, with a
small dataset and a notebook that explains each setting and lets you tune them.

## Contents

| File | |
|---|---|
| `ssl_detector_demo.ipynb` | The notebook: the data, what each setting does, and an interactive tuning panel |
| `ssl_detector.py` | The detector (about 60 lines) |
| `data/autonaut_wed_night.npz` | Four hours of AutoNaut echosounder data (see `data/README.md`) |
| `tests/` | Tests of the detector |

## The detector

For each ping:

1. Keep the profile within the depth range `min_depth` to `max_depth`.
2. Smooth it along depth with a Gaussian filter of width `sigma`.
3. Remove strong echoes: the profile ends at the first value at or above `sv_max` (seafloor, fish school).
4. Find the layer peaks with Sv at or above `sv_min` (`scipy.signal.find_peaks`).
5. Keep the main layer peak, the most prominent one.
6. Its edges at `rel_height` of the peak's prominence give the upper and lower bounds `z_u` and `z_l`
   (`scipy.signal.peak_widths`).
7. Intensity `I = (Sv_peak - sv_min) / (sv_max - sv_min)`, clipped to 0 to 1.

| Setting | Case-study value | |
|---|---|---|
| `min_depth`, `max_depth` | 2.5 m, 100 m | Depth range searched |
| `sigma` | 3.1 m | Width of the Gaussian smoothing |
| `sv_min` | −84 dB | Weaker peaks are background |
| `sv_max` | −52.5 dB | Stronger echoes are removed, with everything below them |
| `rel_height` | 0.95 | Where the bounds are read, as a fraction of the peak's prominence |

```python
import ssl_detector as ssl

d = ssl.load_data()                              # sv (n_ping, n_depth) in dB, depth, time, lat, lon
r = ssl.detect(d["sv"], d["depth"], **ssl.DEFAULTS)
r["z_u"], r["z_l"], r["z_peak"], r["sv_peak"], r["intensity"], r["found"]
```

The input is Sv in dB on an ascending, evenly spaced depth grid with no NaN values. Putting profiles on
such a grid is the paper's resampling step; the demo data are already on a 0.1 m grid.

The per-ping bounds are noisy. As the paper notes, they need filtering over pings; the notebook uses
`ssl.running_median` as an example. That filter is not part of the detector.

## Running it

In the browser, with no installation: use the Binder badge above.

On your own computer, with Python 3.10 or newer:

```bash
pip install -r requirements.txt
jupyter lab ssl_detector_demo.ipynb
```

Run the tests with `pip install pytest`, then `pytest`.

## Licence

The code is under the MIT licence (`LICENSE`). The data are under CC BY 4.0 (`data/LICENSE`).
