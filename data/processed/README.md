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