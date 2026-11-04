# Demo data

`autonaut_wed_night.npz`: echosounder data from the AutoNaut uncrewed surface vehicle (NTNU) off
Mausund/Frøan, Norway, 5 June 2024 21:26 to 6 June 2024 01:46 UTC.

- Simrad EK80 WBT Mini with a 200 kHz split-beam transducer (ES200-7CDK), CW pulses of 1.024 ms at 75 W,
  about one ping every 3 s. The echosounder was not calibrated.
- Sv computed with Echopype (Lee et al. 2024, https://doi.org/10.1093/icesjms/fsae133) from 26 raw files
  (`raw_files`), then averaged into 0.1 m depth bins (mean of the linear values) from 0.45 to 99.95 m. Depth is the range from the transducer; the transducer depth is
  not added. The empty or blank bins in the first 0.4 m are left out.
- Position from the vehicle's navigation log.

| Array | Shape | |
|---|---|---|
| `sv_cdb` | (5200, 996) int16 | Sv in 0.01 dB (divide by 100 for dB re 1 m⁻¹) |
| `depth` | (996,) | Bin centres, m |
| `time` | (5200,) datetime64[ms] | Ping time, UTC |
| `lat`, `lon` | (5200,) | Position, decimal degrees |
| `raw_files` | (26,) str | The EK80 .raw files the pings come from |

```python
import numpy as np

with np.load("autonaut_wed_night.npz") as f:
    sv = f["sv_cdb"] / 100.0
```

Licence: CC BY 4.0 (`LICENSE`).
