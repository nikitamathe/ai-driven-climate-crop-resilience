# Processed outputs

Current contents:

| File                                                    | Generator                       | Meaning                                    |
|---------------------------------------------------------|---------------------------------|--------------------------------------------|
| `final_crop_resilience_district_year.csv`               | `src.pipeline.run_pipeline`     | Corrected resilience CSV (leak-free features, temporal split) |
| `final_crop_resilience_district_year_original.csv`      | Original Colab notebook (leaky) | Historical result, preserved as audit evidence |
| `yield_vs_rainfall.png`                                 | `src.pipeline.run_pipeline`     | Diagnostic plot: yield vs rainfall by class |
| `feature_importance.png`                                | `src.pipeline.run_pipeline`     | Diagnostic plot: top yield predictors       |
| `resilience_distribution.png`                           | `src.pipeline.run_pipeline`     | Diagnostic plot: resilience class counts    |

Not yet present:

* `downscaled/` — generated only if `hpc/downscale_cnn.py` is run; the CNN
  demonstration has not been executed in this environment yet, so no directory
  exists and none is claimed.

## Benchmark outputs (E09 HPC harness)

`data/processed/benchmarks/` is written by `python -m hpc.benchmark` and is
git-ignored. Each run creates a subdirectory (`<run_id>/`) containing:

* `<run_id>.csv` — measured rows: workload (tile/cnn), backend/device, grid
  size, tiles, workers, wall time, throughput, speedup, status.
* `config.json` — the exact configuration used.
* `speedup_vs_workers.png`, `scaling_by_grid_size.png` — diagnostic plots.

All numbers are measured on the local CPU; GPU/absent-hardware rows are labelled
`status="not run"` and are never fabricated. These are engineering performance
measurements, not scientific climate results.

## Geospatial outputs (E10 — district resilience maps)

`data/processed/geojson/` is written by `python -m src.visualization.geospatial`
and is git-ignored. It contains:

* `resilience_district_<tag>.geojson` — merged district polygons with mean
  `Resilience_Index`, min/max, record count, and `% vulnerable` per district.
* `choropleth_index_<tag>.png` / `choropleth_vulnerability_<tag>.png` — static
  choropleth maps.
* `resilience_map_<tag>.html` — interactive Folium map.

The `<tag>` is the filter used (`all`, or a crop name / a year). Aggregations
are computed independently for each filter.

### E10 join coverage & honesty

The maps are a **geospatial visualization/aggregation layer** over the existing
model output — they do **not** perform climate downscaling. The NASA POWER
climate input remains a single spatial point (E06); the maps simply spatialize
district-level model-output resilience using administrative boundaries, joined
via a name-standardization map (`data/geo/name_mapping.json`).

Spatial validity depends on that district-name join, so unmatched districts are
always counted and reported. On the current boundary dataset (DataMeet Census
2011) the join covers 305/311 districts (~98%); the 6 unmatched model districts
(Champaran, Shahabad, Singhbhum, 24 Parganas, Midnapur, West Dinajpur) were
split into or merged with modern districts and have no single valid geometry, so
they are left unmatched rather than forced into a modern district. See
`data/geo/name_mapping.json` (or the builder's printed report) for the exact map
and provenance.