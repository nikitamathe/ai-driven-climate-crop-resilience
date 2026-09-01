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