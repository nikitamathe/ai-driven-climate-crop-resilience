# AI-Driven Climate Downscaling and Crop Resilience Mapping

Predicting crop resilience (yield shortfall vs. expected yield) across Indian
districts using a Random Forest trained on district-level crop records merged
with a NASA POWER climate record, plus a convolutional super-resolution
"downscaling" demonstration.

> **Honest status (post-audit).** This repository was rebuilt from an academic
> submission whose claims partially outran its implementation. Everything here
> is either (a) genuinely implemented and reproducible, or (b) explicitly
> labelled as demonstration/future work. Nothing is claimed that the code does
> not do. See [Implementation vs. original claims](#implementation-vs-original-claims).

---

## Table of contents

- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Data](#data)
- [Key audit findings](#key-audit-findings)
- [Implementation vs. original claims](#implementation-vs-original-claims)
- [Tests](#tests)
- [Reproducing the original notebook](#reproducing-the-original-notebook)
- [Roadmap](#roadmap)

## Repository layout

```
data/
  raw/
    nasa_power/         NASA POWER monthly climate record + provenance
    crop_yield/         District-level crop yield dataset + provenance
  processed/            Pipeline outputs (incl. original, audit-flagged one)
src/
  data/loader.py        Load, aggregate, merge datasets
  features/engineering.py  Feature sets, temporal split (leakage-aware)
  models/               RF training, metrics, resilience index
  pipeline/run_pipeline.py  End-to-end CLI
  visualization/plots.py    Diagnostic plots
hpc/
  downscale_cnn.py      SRCNN-style rainfall super-resolution demo
notebooks/ML_Code.ipynb Original Colab notebook (paths fixed to relative)
docs/                   Report PDF, presentation, Power BI file, screenshots
tests/                  pytest suite
```

## Quick start

Requires Python >= 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch needs manual install
python -m src.pipeline.run_pipeline      # full resilience pipeline
python -m pytest                         # test suite
```

The pipeline (re)generates:

* `data/processed/final_crop_resilience_district_year.csv`
* `data/processed/yield_vs_rainfall.png`, `feature_importance.png`,
  `resilience_distribution.png`

The CNN demo:

```bash
python hpc/downscale_cnn.py --epochs 30
```

## Pipeline

1. **Load** district crop records and the monthly NASA POWER record.
2. **Aggregate** the NASA record to yearly rainfall sum + temperature means.
3. **Merge** the two on `Year`.
4. **Features** — excludes the nutrient columns shown to be derived from the
   target (see [data leakage](#key-audit-findings)).
5. **Temporal split** — train on years before a cutoff, test on the rest
   (default cutoff = 2014). This replaces the original *random* split that
   leaked across the time axis.
6. **Train** a `RandomForestRegressor` (300 trees, depth 15) on yield.
7. **Score** with R², RMSE, MAE.
8. **Resilience** — `Resilience_Index = Actual_Yield / Predicted_Yield`;
   classes: Highly Resilient (>=0.9), Moderately Resilient (>=0.7),
   Vulnerable (<0.7).
9. **Aggregate** to `Year × State × District × Crop`, write CSV + plots.

## Data

See the provenance READMEs in `data/raw/nasa_power/` and `data/raw/crop_yield/`.
Both documents record the limitations found during the audit.

## Key audit findings

The Phase 0 forensic audit against the original project delivered these
findings (full report in the open-review conversation; summarized here):

1. **Data leakage** — the crop dataset's nutrient columns are
   deterministically derived from `Yield_kg_per_ha` using crop-specific
   constant coefficients (e.g. rice N/Yield = 0.025, maize N/Yield = 0.027).
   These are target-derived leakage features and must not be used as model
   inputs. The original model's R²≈0.97 is largely a circular artifact, not
   skill.
2. **Static climate columns** — the district-level "weather" columns are
   per-crop constants (e.g. rice is always 25 °C / 1200 mm), not real
   observations.
3. **Single-point NASA data applied nationally** — one lat/lon's climate was
   merged onto every district.
4. **Temporal leakage** — the original used a random 80/20 split, so the same
   year sat on both sides of the split.
5. **"Downscaling CNN" was a stub** — the original `HPC_Code.txt` truncated
   mid-training-loop and referenced a dataset that does not exist.
6. **Dashboard not code-based** — only a Power BI binary, nothing
   reproducible in the repository.

The improved pipeline fixes 1 and 4, and documents 2 and 3. Item 5 is now a
working, honest demonstration harness (see below).

## Implementation vs. original claims

| Original claim                         | Status in this repo |
|----------------------------------------|---------------------|
| Random Forest yield prediction          | **Implemented** (corrected split + features) |
| Resilience index / class mapping         | **Implemented**     |
| Diagnostic plots                        | **Implemented**     |
| "Climate downscaling" (CNN)             | **Demonstration harness** — SRCNN super-resolution on an illustrative field built from the real NASA record. Not a validated downscaled climate product. |
| "Parallel data processing / HPC"        | **Future work** — `DataParallel` path exists for multi-GPU, but no distributed compute or benchmarks yet. |
| Web dashboard                           | **Future work** — original deliverable was a closed Power BI file (`docs/Project V.pbix`). |
| District-level downscaled climate data  | **Future work** — requires gridded coarse/fine climate data, which this repo does not yet contain. |

## Tests

```bash
python -m pytest
```

Covers data loading, the leakage invariants, temporal split behaviour, and the
resilience index/classification boundaries.

## Model explainability (E04, SHAP)

`src/models/explain.py` explains the tuned Random Forest and XGBoost models
(from E03) with `shap.TreeExplainer`, and `src/visualization/shap_plots.py`
renders beeswarm summary, dependence (temperature & rainfall) and
mean-|SHAP|-vs-permutation-importance plots.

```bash
python -m src.models.explain --out-dir data/processed/shap
```

**SHAP is attribution, not causation.** SHAP values describe how much each
feature contributed to a specific model's predictions for a *specific* dataset;
they do **not** establish cause-and-effect relationships between climate
variables and yield. Treat these plots as attribution diagnostics, never as
causal claims.

Leakage note: attributions use `tree_path_dependent` SHAP (no interventional
background reference set) and are computed on the held-out temporal test set,
so no test-year information enters the explanation.

## Reproducing the original notebook

`notebooks/ML_Code.ipynb` preserves the original methodology with paths
converted from `/content/...` to repository-relative paths. Running it as-is
reproduces the audit-flagged results (including the leaky R²≈0.97), which are
kept for comparison in
`data/processed/final_crop_resilience_district_year_original.csv`.

## Roadmap

- [ ] NASA POWER download script (`src/data/download.py`) for full provenance
- [ ] True region-aware validation (e.g. Telangana subset)
- [ ] Gridded coarse→fine downscaling input data + CNN evaluation metrics
- [ ] Baseline models (median-yield, linear) for honest skill comparison
- [ ] Cross-validation with group-by-year folds
- [ ] Reproducible dashboard (e.g. Streamlit) replacing the raw `.pbix`
- [ ] CI, packaging, and a published PyPI-style release