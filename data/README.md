# Demo data

Two stretches of echosounder data from the AutoNaut uncrewed surface vehicle (NTNU) off Mausund/Frøan,
Norway, in June 2024:

| File | Time (UTC) | Pings | |
|---|---|---|---|
| `autonaut_thu_morning.npz` | 6 June 08:40 to 09:40 | 3594 | A strong layer near 15 m and a weaker one near 55 m; one ping a second |
| `autonaut_wed_night.npz` | 5 June 21:26 to 6 June 01:46 | 5200 | A layer moving between about 45 and 90 m; one ping every 3 s |

- Simrad EK80 WBT Mini with a 200 kHz split-beam transducer (ES200-7CDK), CW pulses of 1.024 ms at 75 W.
  The echosounder was not calibrated. In both stretches the seabed is deeper than 100 m.
- Sv computed with Echopype (Lee et al. 2024, https://doi.org/10.1093/icesjms/fsae133) from the raw files
  listed in `raw_files` (18 and 26 files), then averaged into 0.1 m depth bins (mean of the linear values)
  from 0.45 to 99.95 m. Depth is the range from the transducer; the transducer depth is not added. The
  empty or blank bins in the first 0.4 m are left out.
- Position from the vehicle's navigation log.

| Array | Shape | |
|---|---|---|
| `sv_cdb` | (n_ping, 996) int16 | Sv in 0.01 dB (divide by 100 for dB re 1 m⁻¹) |
| `depth` | (996,) | Bin centres, m |
| `time` | (n_ping,) datetime64[ms] | Ping time, UTC |
| `lat`, `lon` | (n_ping,) | Position, decimal degrees |
| `raw_files` | (n_file,) str | The EK80 .raw files the pings come from |

```python
import numpy as np

with np.load("autonaut_thu_morning.npz") as f:
    sv = f["sv_cdb"] / 100.0
```

Licence: CC BY 4.0 (`LICENSE`).
