# NASA POWER climate record

**File:** `nasa_power_updated.csv` (300 rows, monthly, 1996–2020)

## Provenance

Downloaded from the [NASA POWER API](https://power.larc.nasa.gov/) for a single
location (one latitude/longitude point). No latitude/longitude column is stored
in the CSV — the point was kept implicit.

## Columns

| Column    | Unit  | Meaning                                |
|-----------|-------|----------------------------------------|
| Year      | —     | Calendar year (1996–2020)              |
| Month     | —     | Calendar month (1–12)                  |
| Rainfall  | mm    | Monthly total precipitation            |
| AvgTemp   | °C    | Monthly average temperature            |
| MaxTemp   | °C    | Monthly maximum temperature            |
| MinTemp   | °C    | Monthly minimum temperature            |

## Coverage

* **Raw CSV:** monthly, 1996–2020 (25 years × 12 months = 300 rows).
* **Pipeline overlap:** the crop dataset spans 1966–2017, so the inner merge
  on `Year` in `src.data.loader.merge_datasets` only uses the shared window
  **1996–2017**. The 2018–2020 rows of this record are not currently consumed
  by the pipeline.

## Honest limitations (from the Phase 0 audit)

* The record is a **single point** (time series), not a spatial grid — it does
  **not** provide spatial climate information and cannot by itself support
  spatially resolved "downscaling".
* In the original notebook this single point's yearly aggregates were merged
  onto every district in all 20 states, implying one weather signal for the
  whole country. This is a significant simplification.
* The download script is not yet part of the repository; the CSV is the raw
  input. A reproduction script is planned (see `src/data/download.py`).

## Fair use

NASA POWER data are U.S. Government work in the public domain. See the
[POWER Data Access Viewer](https://power.larc.nasa.gov/) for the canonical terms.