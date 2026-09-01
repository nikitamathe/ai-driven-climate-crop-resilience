# District-level crop yield dataset

**File:** `Custom_Crops_yield_Historical_Dataset.csv` (50,765 rows, 1966–2017)

## Columns

| Column                  | Meaning                                          |
|-------------------------|--------------------------------------------------|
| Dist Code               | District numeric code (row index)                |
| Year                    | Crop year (1966–2017)                            |
| State Code / State Name | Admin state (20 states)                          |
| Dist Name               | District (311 districts)                         |
| Crop                    | rice, maize, chickpea or cotton                  |
| Area_ha                 | Sown area (ha)                                   |
| Yield_kg_per_ha         | **Target** yield variable                        |
| N/P/K_req_kg_per_ha     | Nutrient requirement per hectare                 |
| Total_N/P/K_kg          | Nutrient requirement x Area                      |
| Temperature_C .. Solar_Radiation_MJ_m2_day | Climate/soil attributes             |

## Known data issues (from the Phase 0 audit) — READ THIS FIRST

### 1. The "climate" columns are static per-crop constants, not measurements

The columns `Temperature_C`, `Humidity_%`, `pH`, `Rainfall_mm`,
`Wind_Speed_m_s`, `Solar_Radiation_MJ_m2_day` contain only 4–6 unique values
that are identical across every year and district for a given crop:

| Crop     | Temp | Humidity | Rainfall | Wind | Solar | pH |
|----------|------|----------|----------|------|-------|----|
| rice     | 25   | 80       | 1200     | 2.0  | 18    | 6.5 |
| maize    | 22   | 70       | 800      | 2.5  | 20    | 6.0 |
| chickpea | 20   | 60       | 600      | 1.5  | 16    | 6.5 |
| cotton   | 28   | 65       | 700      | 3.0  | 22    | 6.5 |

They encode crop suitability, not real spatiotemporal weather.

### 2. The nutrient columns are derived from the target (data leakage)

The nutrient-requirement columns are deterministically computed from the
target (`Yield_kg_per_ha`) using **crop-specific constant coefficients**. The
verified ratio `nutrient / Yield` is constant for every row of a given crop:

| Crop     | `N_req / Yield` | `P_req / Yield` | `K_req / Yield` |
|----------|-----------------|-----------------|-----------------|
| chickpea | 0.018           | 0.010           | 0.018           |
| cotton   | 0.027           | 0.012           | 0.027           |
| maize    | 0.027           | 0.012           | 0.017           |
| rice     | 0.025           | 0.012           | 0.022           |

The total-nutrient fields are likewise deterministic products of those
requirement fields and sown area:

    Total_N_kg ≈ N_req_kg_per_ha × Area_ha
    Total_P_kg ≈ P_req_kg_per_ha × Area_ha
    Total_K_kg ≈ K_req_kg_per_ha × Area_ha

(verified with `np.allclose` over the complete dataset).

Because all six nutrient columns are derived from the variable being
predicted, **including them in a model circularly inflates performance** (this
is why the original notebook reported R²≈0.97). The improved pipeline excludes
them (see `src/features/engineering.py`), and the test suite
(`tests/test_data.py`) verifies these invariants so the leakage cannot be
silently re-introduced.

## Provenance

District-level yield statistics for Indian crops distributed freely by
government sources. The original source and download procedure are not
recorded in the project; treat the data as demonstration material pending
confirmation of its provenance.
