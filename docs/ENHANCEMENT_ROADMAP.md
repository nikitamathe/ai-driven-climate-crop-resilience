# Project Enhancement Roadmap & Final Project Definition

**Phase 0 deliverable — planning/assessment only.**
No code or data was modified by this document. All enhancements below are
proposals; the recommended roadmap gates implementation behind approval.

Ground truth used throughout this document (from the completed forensic audit):

| Fact | Evidence |
|------|----------|
| Crop dataset: 50,765 rows, 1966–2017, 20 states, 311 districts, 4 crops | `data/raw/crop_yield/Custom_Crops_yield_Historical_Dataset.csv` |
| Nutrient columns are computed from the target (`N_req = Yield × 0.025`) | verified row-level arithmetic |
| District "climate" columns are static per-crop constants | 4–6 unique values per column |
| NASA POWER record is a single point, monthly, 1996–2020 | `data/raw/nasa_power/nasa_power_updated.csv` |
| Original RF used a random 80/20 split, R²≈0.9728 | notebook cells 19–23 |
| Original output: 12,121/12,131 rows "Highly Resilient" | `data/processed/final_..._original.csv` |
| No gridded coarse/fine climate data exists in the repo | full tree walk |
| No git history, no CI, no tests, no requirements before Phase 1 | audit |

---

## 1. Design principles for this roadmap

1. **Evidence-first.** Every claim must be traceable to code in the repo or a
   documented external source. Anything not supportable is explicitly labeled.
2. **Provenance over polish.** Download scripts, schema checks, and reproducible
   seeds outweigh one-off impressive notebooks.
3. **HPC must be earned.** We only parallelize workloads that measurably benefit
   from it, and we publish measured speedups — never estimated ones.
4. **No fake "downscaling".** Single-point NASA data cannot produce a validated
   downscaled product. The CNN remains a transparent demonstration until real
   gridded data is available (data-gated Tier 3).
5. **Scientific metrics only where the data supports them.** SPI (drought) is an
   established metric and can be computed from the monthly record. "Heat stress"
   from monthly means is a clearly-labeled heuristic, not an agronomic claim.
6. **The resilience index is a yield-gap ratio, not agronomic advice.** It is
   validated statistically (correlation with stress indicators), not claimed as
   decision support.

---

## 2. Candidate enhancement registry

Each enhancement is specified by the 15 attributes requested. Entries are kept
deliberately focused; features with overlapping value were merged so the set
stays small and high-value.

---

### E01 — Honest baselines, model comparison, and corrected methodology (CORE)

1. **Name:** Honest model baselines & comparison
2. **Problem it solves:** The original R²≈0.97 is a circular artifact of
   target-derived features and a random split. There is currently no honest
   statement of what any model can actually predict on unseen years.
3. **Relevance:** The entire project's credibility rests on knowing the true
   predictive skill. A leak-free RF vs. XGBoost vs. linear vs. "predict the
   district–crop historical mean" comparison is the scientific backbone.
4. **Technical approach:** One evaluation harness, four models. All use the
   corrected feature set (no `N_req_*`/`Total_*`), a temporal split (train
   `< 2014`, test `>= 2014`), identical scoring (R², RMSE, MAE, MAPE). The
   district–crop mean baseline is the honesty anchor (a model that beats it is
   earning its keep).
5. **Required data:** Already present.
6. **Libraries/tools:** scikit-learn, xgboost (add `xgboost`), pandas.
7. **Computational requirements:** Trivial (< 1 min, CPU).
8. **CPU/GPU/HPC:** CPU only. No parallelization justified at 40K rows.
9. **Difficulty:** Easy.
10. **Effort:** 1–2 days (includes a `compare_models.py` entry point + tests).
11. **Scientific validity:** High — this is standard best practice; the
    expectation is that honest R² drops to roughly 0.3–0.6, which is a *good*,
    defensible result that the README should celebrate, not hide.
12. **Evaluation:** Metric table across four models; residual plots; reported
    performance vs. the naive mean baseline.
13. **Output/visualization:** `data/processed/model_comparison.csv`, bar chart
    of R²/RMSE, residual scatter.
14. **Placement:** Core (Tier 1).
15. **Feasible now:** Yes, fully.

---

### E02 — Group-aware and temporal cross-validation

1. **Name:** Spatial/temporal cross-validation
2. **Problem it solves:** The single temporal split gives one noisy estimate.
    Random KFold would reintroduce temporal leakage. A grouped CV reports mean ±
    std of skill across years and districts.
3. **Relevance:** Directly attacks the audit's leakage finding and makes every
    metric in E01 statistically honest.
4. **Technical approach:** `GroupKFold`/custom splitter: (a) group by `Year`
    (temporal holdout), (b) group by `Dist Name` (geographic holdout, tests
    generalization to unseen districts). Report metric distributions.
5. **Required data:** Present.
6. **Libraries/tools:** scikit-learn.
7. **Computational requirements:** Low (CPU, minutes).
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Easy.
10. **Effort:** 1 day.
11. **Scientific validity:** High. This is the correct way to estimate
    generalization for tabular yield data.
12. **Evaluation:** CV mean±std table; per-fold metrics; comparison to the
    single-split numbers.
13. **Output/visualization:** `data/processed/cv_results.csv`, fold boxplots.
14. **Placement:** Core (Tier 1).
15. **Feasible now:** Yes.

---

### E03 — Hyperparameter optimization with time-aware CV

1. **Name:** Hyperparameter tuning (RF + XGBoost)
2. **Problem it solves:** `n_estimators=300, max_depth=15` were never tuned; the
   gap in model quality vs. a tuned model is unknown.
3. **Relevance:** Rounds out the ML methodology; demonstrates search hygiene
   (tuning *inside* the CV loop so we do not leak tuning signal).
4. **Technical approach:** `RandomizedSearchCV` (cheap, adequate) or Optuna
   `TPESampler` on RF (n_estimators, max_depth, min_samples_split) and XGBoost
   (eta, max_depth, subsample, colsample_bytree) with the year-grouped CV from
   E02 as the objective.
5. **Required data:** Present.
6. **Libraries/tools:** scikit-learn, optional `optuna`.
7. **Computational requirements:** Moderate (CPU, 10–30 min per model).
8. **CPU/GPU/HPC:** CPU; can use `n_jobs` for the search, but **do not** claim
   this as an HPC contribution.
9. **Difficulty:** Easy–Medium.
10. **Effort:** 2–3 days.
11. **Scientific validity:** High if nested/grouped CV is respected; otherwise it
    silently overfits the metric.
12. **Evaluation:** Optimized vs. default metric table under identical E02 CV.
13. **Output/visualization:** Tuning log, parallel-box plots of best params.
14. **Placement:** Core/High-value (Tier 2).
15. **Feasible now:** Yes.

---

### E04 — Model explainability with SHAP

1. **Name:** SHAP-based yield-model explanation
2. **Problem it solves:** Raw feature importance does not say *which way* a
   feature pushes yield (rainfall up → yield up?) nor is it stable across models.
3. **Relevance:** Turns the "top factors" chart into an actual causal-sign
   statement the report can use; also exposes whether the static per-crop
   constants dominate (expected — a finding to document, not hide).
4. **Technical approach:** `shap.TreeExplainer` on the best RF and XGBoost;
   summary (beeswarm) plot, dependence plots for temperature and rainfall,
   mean |SHAP| bar chart vs. scikit feature importance for honesty about
   attributions.
5. **Required data:** Present.
6. **Libraries/tools:** `shap` (~0.46), matplotlib.
7. **Computational requirements:** Low–Moderate (CPU, minutes on 40K rows).
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 2 days.
11. **Scientific validity:** High as *attribution*; SHAP is not causal — state
    this in the README.
12. **Evaluation:** Consistency of top features across RF/XGB; SHAP vs.
    permutation importance comparison.
13. **Output/visualization:** beeswarm, dependence plots, SHAP-by-crop panels.
14. **Placement:** High-value (Tier 2).
15. **Feasible now:** Yes.

---

### E05 — Uncertainty estimation (quantile regression forests / conformal)

1. **Name:** Yield prediction with prediction intervals
2. **Problem it solves:** A resilience class computed from a point prediction
   carries no confidence. Districts with wide intervals should not be labeled
   "Vulnerable" with the same certainty as narrow ones.
3. **Relevance:** The resilience signal is inherently uncertain; surfacing it is
   a senior-level methodological addition and a README differentiator.
4. **Technical approach:** `RandomForestQuantileRegressor`
   (`sklearn_quantile`, or implement quantile RF with `fit/predict` on
   weighted trees) or split-conformal prediction around the best point model
   (~5 lines of careful tuning-free code, distribution-free coverage).
5. **Required data:** Present.
6. **Libraries/tools:** `scikit-garden`/`sklearn_quantile` or pure-scikit
   conformal implementation.
7. **Computational requirements:** Moderate (CPU).
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 3–4 days (incl. coverage validation).
11. **Scientific validity:** Conformal gives a distribution-free coverage
    guarantee (e.g., 90% CI on test years); that is a publishable-quality
    claim. Quantile RF is approximate but cheap.
12. **Evaluation:** Empirical coverage on held-out years, interval width by
    district/crop; uncertainty overlaid on the resilience map.
13. **Output/visualization:** "Resilience with 90% CI" choropleth/bars.
14. **Placement:** High-value (Tier 2).
15. **Feasible now:** Yes, with conformal.

---

### E06 — Climate-stress indicators from the NASA record

1. **Name:** NASA climate-stress indicators (drought, heat, anomaly, trend)
2. **Problem it solves:** The project title says "climate resilience", but the
   current pipeline has *no climate analysis at all* — just two aggregated
   temperatures and a summed rainfall merged on by year.
3. **Relevance:** The single real observational asset (25 years of monthly
   climate) is currently underused. Indicators give the resilience analysis a
   scientifically established stress axis.
4. **Technical approach:**
   - **SPI-3 / SPI-6 / SPI-12** (standardized precipitation index; McKee et
     al. 1993) from rolling precipitation sums — established metric, computed
     with gamma fitting (`scipy.stats.gamma`) — label: *established*.
   - **Rainfall & temperature anomalies** vs. the 1996–2020 climatological
     baseline (z-scores) — label: *established (standard anomaly)*.
   - **Trend tests:** Mann-Kendall + Sen slope on monthly/yearly rainfall and
     temperature — label: *established statistical test* (scipy.implementation
     or `pymannkendall`).
   - **Crop-thermal stress heuristic:** months where AvgTemp deviates from the
     crop's thermal optimum (from the static constants) beyond ±Δ — label:
     *project heuristic, NOT an agronomic claim*; clearly flagged.
5. **Required data:** Present (`nasa_power_updated.csv`). No extra downloads.
6. **Libraries/tools:** scipy, `pymannkendall` (or re-implemented MK), pandas.
7. **Computational requirements:** Trivial (CPU, seconds).
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium (SPI/MK are subtle; validate against known library).
10. **Effort:** 3–4 days incl. unit tests pinning known-value checks.
11. **Scientific validity:** Strong for SPI/anomalies/MK. Heuristics explicitly
    separated from established metrics in output columns (`indicator_type`).
12. **Evaluation:** Unit tests against published SPI values; sanity: drought
    years known in Telangana (e.g., 2002/2015) should flicker in SPI.
13. **Output/visualization:** `data/processed/climate_indicators.csv`,
    SPI time series, anomaly heatmap (year × indicator), trend significance
    table.
14. **Placement:** Core (Tier 1) — this is where real climate science enters.
15. **Feasible now:** Yes.

---

### E07 — Data validation, provenance, and ingestion module

1. **Name:** Reproducible data ingestion + validation
2. **Problem it solves:** Datasets are undocumented binaries with known
   defects; nothing stops silent schema drift or re-introduction of leaked
   columns.
3. **Relevance:** Makes the "data quality problems we found" a *managed*
   property of the pipeline instead of a footnote.
4. **Technical approach:** `src/data/ingest.py`: (a) `download_nasa_power.py`
   (POWER API call, params in a reproducible YAML/JSON, writes
   `nasa_power_updated.csv`), (b) pandera schema for both CSVs
   (`DataFrameSchema` with column types, ranges, no-NaN checks), (c) invariant
   checks that *fail loudly* if nutrient columns are re-added to features (the
   leakage detector), (d) write canonical Parquet copies with a recorded
   SHA-256 checksum for cached/re-verified reads.
5. **Required data:** Present; download script needs internet at run time (CI
   integration-testable with `vcr`/mocked API response).
6. **Libraries/tools:** pandas, `pandera`, pyarrow, requests.
7. **Computational requirements:** Trivial.
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Easy–Medium.
10. **Effort:** 2–3 days.
11. **Scientific validity:** Engineering quality, not science per se; it is the
    foundation for every reproducible claim.
12. **Evaluation:** `pytest` coverage of all schema failures; checksums recorded
    and asserted in CI.
13. **Output/visualization:** validated Parquet files, `data/raw/*/manifest.json`.
14. **Placement:** Core (Tier 1).
15. **Feasible now:** Yes (download requires internet; validation does not).

---

### E08 — Experiment configuration, artifact logging, and CI

1. **Name:** Experiment tracking + model artifacts + CI
2. **Problem it solves:** Every metric so far exists only in notebook outputs
   and chat transcripts. No versioned record of "config X → metrics Y" exists.
3. **Relevance:** Standard MLOps hygiene that a portfolio reviewer checks for
   immediately; makes runs reproducible with one command.
4. **Technical approach:** (a) YAML experiment configs (`experiments/*.yaml`
   with seed, split cutoff, model, params), (b) `run_experiment.py` that writes
   config hash, git commit hash, metrics, and param artifacts to
   `data/processed/experiments/<hash>/`, (c) serialized model + `joblib`,
   (d) GitHub Actions workflow: `pytest`, a smoke pipeline run, a data checksum
   check. Optionally an MLflow local run — but **not required**; JSON artifacts
   are enough at this scale.
5. **Required data:** Present (CI will `pip install -r requirements.txt`).
6. **Libraries/tools:** PyYAML, joblib, pytest, GitHub Actions (runners have
   pip — the one capability this local machine lacks).
7. **Computational requirements:** CI minutes; cheap.
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 3 days.
11. **Scientific validity:** Reproducibility claim becomes verifiable — the
   strongest possible argument for the whole project.
12. **Evaluation:** CI green; any reviewer can reproduce metrics from config
   alone.
13. **Output/visualization:** `data/processed/experiments/*` + README table of
   runs.
14. **Placement:** Core (Tier 1).
15. **Feasible now:** Yes (given GitHub; local run once packages are installed).

---

### E09 — HPC made real: parallel tile preprocessing + benchmark harness

1. **Name:** Measured HPC / performance engineering
2. **Problem it solves:** "HPC" is currently a label on a file. There is no
   parallel workload, no benchmark, no speedup number. The grid-based CNN
   (spatial fields) is the rare workload where tiling genuinely scales.
3. **Relevance:** This is the single highest-value *engineering* addition for
   the "AI + HPC" claim: it converts a trivially-small task into a defensible
   parallelization study with measured results.
4. **Technical approach:**
   - **Workload A — parallel tile preprocessing:** partition synthetic/CNN
     rainfall grids into N×N tiles; process tiles with `multiprocessing.Pool`
     (and `joblib` variant); compare vs serial; scale N tiles × P workers.
   - **Workload B — GPU CNN training:** train `DownscaleCNN` on CPU, then GPU
     (when available); record wall time/epoch for several field sizes
     (64² → 512²) to show GPU crossover.
   - `hpc/benchmark.py` writes `data/processed/benchmarks/<run>.csv` with
     config, wall time, throughput, and speedup. Plot speedup-vs-workers and
     scale curves. **Rules: no fabricated numbers — empty/absent rows are
     labelled "not run".**
5. **Required data:** Synthetic fields from the real NASA record (as in the
   CNN demo) — no new data.
6. **Libraries/tools:** `multiprocessing`/`joblib`, torch (CPU path works
   without CUDA), matplotlib.
7. **Computational requirements:** Minutes on CPU; optionally a GPU machine.
8. **CPU/GPU/HPC:** **This is where CPU/GPU/HPC genuinely matters.** Tiling +
   pooling exploits multi-core; the CNN is GPU-elastic. MPI is not justified
   at this scale — state that explicitly.
9. **Difficulty:** Medium.
10. **Effort:** 4–5 days.
11. **Scientific validity:** High for engineering claims *if and only if*
    results are measured, not promised. Document contention, GIL effects on the
    tile loop (use `ProcessPoolExecutor` to sidestep it).
12. **Evaluation:** Speedup tables per workload; Amdahl's-law sanity lane;
    scaling exponent estimate.
13. **Output/visualization:** `benchmarks/` CSVs, `speedup_vs_workers.png`,
    `scaling_by_grid_size.png`, "HPC: what we parallelize and why" section in
    README.
14. **Placement:** Core/High-value (Tier 1 for the honest harness; the claim is
    Tier 2).
15. **Feasible now:** Yes — CPU-only mode runs here; GPU columns await a GPU
    host (honestly labeled).

---

### E10 — Geospatial resilience maps (GeoPandas)

1. **Name:** District-level choropleth resilience maps
2. **Problem it solves:** All outputs are tabular. Spatial patterns — where
   vulnerability concentrates — are invisible.
3. **Relevance:** The map is the highest-impact single visualization for the
   portfolio and is required to answer the practical question "which regions
   are most vulnerable?".
4. **Technical approach:** Join the resilience summary on an India district
   boundary shapefile (public source: DataMeet maps or ICRISAT district
   boundaries) via `Dist Name`/`State Name` (needs name-standardization
   mapping). Produce choropleths of mean Resilience_Index and %Vulnerable per
   district, per crop, and per key year. Export merged GeoJSON to
   `data/processed/geojson/`.
5. **Required data:** **New external data:** one district-boundary shapefile +
   name mapping. Public and free; must be fetched in user's environment (this
   machine lacks pip/internet tooling).
6. **Libraries/tools:** `geopandas` (~0.14x, needs GDAL wheels or conda),
   matplotlib, pyarrow.
7. **Computational requirements:** Low.
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 3–4 days (bulk is the name-matching mapping).
11. **Scientific validity:** Standard cartography; validity depends on the
    join (must count unmatched districts and report them).
12. **Evaluation:** Join coverage printout (X/311 districts matched); map sanity
    vs. state means.
13. **Output/visualization:** PNG choropleths + interactive HTML (folium) maps
    for crop/year filters.
14. **Placement:** High-value (Tier 2). Dependency: shapefile acquisition.
15. **Feasible now:** Yes once a public shapefile is fetched; the join logic
    can be developed and tested on a tiny sample now.

---

### E11 — Vulnerability atlas and "practical questions" report

1. **Name:** Vulnerability atlas (analysis layer)
2. **Problem it solves:** The README lists questions ("which regions are most
   vulnerable? which crops? which years?") that nothing currently answers with
   numbers.
3. **Relevance:** Turns model + indicators into the concrete analytical
   conclusions a research project should state.
4. **Technical approach:** Aggregations answering each question:
   - most vulnerable districts (bottom-decile Resilience_Index, count of crops
     in "Vulnerable");
   - most affected crops (class share by crop);
   - rainfall/yield and temperature/resilience relationship (E04 SHAP x
     E06 indicator correlation);
   - severe-stress years (years where SPI < −1.5 overlaps vulnerable yield
     gaps);
   - declining-resilience trend per district (MK test on residual year series).
   Each output row carries `evidence: model|indicator|heuristic` and a CI width
   from E05.
5. **Required data:** Present (uses E01–E06 outputs).
6. **Libraries/tools:** pandas, scipy.
7. **Computational requirements:** Trivial.
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 3 days.
11. **Scientific validity:** High if hedged: correlations reported with widths
    and marked non-causal; no agronomic advice claims.
12. **Evaluation:** Every claim in the README "Findings" section must cite a
    generated CSV row.
13. **Output/visualization:** `data/processed/atlas/*.csv` + a
    `docs/FINDINGS.md` auto-generated from them.
14. **Placement:** High-value (Tier 2).
15. **Feasible now:** Yes.

---

### E12 — Reproducible dashboard (replaces the `.pbix`)

1. **Name:** Interactive Streamlit dashboard
2. **Problem it solves:** The only dashboard is a closed Power BI binary
   (`docs/Project V.pbix`) that cannot be rebuilt or reviewed.
3. **Relevance:** Portfolio reviewers want a runnable product; Streamlit is the
   cheapest honest replacement and directly showcases E10's maps + E11's atlas.
4. **Technical approach:** Streamlit app with: crop/year/district filters;
   choropleth (folium); model-comparison table; SPI/temperature trend panels;
   vulnerability leaderboard. Reads only `data/processed/` artifacts — no
   recomputation in-app.
5. **Required data:** Present (post-E01–E11 artifacts); includes shapefile maps
   from E10.
6. **Libraries/tools:** streamlit, folium, geopandas, plotly (optional).
7. **Computational requirements:** Local web app, single process.
8. **CPU/GPU/HPC:** CPU.
9. **Difficulty:** Medium.
10. **Effort:** 4–5 days.
11. **Scientific validity:** n/a (presentation layer); must not let the
    dashboard imply claims the analysis doesn't support.
12. **Evaluation:** `streamlit run` from README works on clean checkout.
13. **Output/visualization:** runnable app; archived screenshots in
    `docs/images/dashboard/` (genuine outputs, per README visual policy).
14. **Placement:** High-value (Tier 2; can drop to optional if time-bound).
15. **Feasible now:** Yes, once E10 artifacts exist.

---

### E13 — Downscaling: demonstrated or real (data-gated)

1. **Name:** CNN downscaling — honest advanced step
2. **Problem it solves:** The project title claims "downscaling"; the current
   code is a super-resolution *demonstration* on an illustrative field. The gap
   must be either closed (with data) or made unmistakably transparent.
3. **Relevance:** Determines whether "downscaling" stays honest-only or becomes
   a defensible experiment (the senior technical centerpiece if data is
   obtained).
4. **Technical approach:**
   - **Track A (no new data, do now):** strengthen the demo into an SR
     benchmark — multiple architectures (SRCNN 3- vs. 5-layer, ESPCN-style
     subpixel), losses (MSE/MAE/SSIM), datasets of grid sizes; report
     **PSNR/SSIM on a held-out field**; state in README/code exactly what is
     demonstrated vs. not. *This is a methodology study, not validated
     downscaling.*
   - **Track B (requires data, future):** real coarse→fine experiment —
     coarse input (e.g., ERA5-Land 0.1°, or NASA POWER gridded) and fine
     reference (e.g., CHIRPS/IMD rainfall at ~0.05°) for a study domain, with
     the CNN/SR model trained and evaluated against the fine observations.
     Only then use the word "downscaling" in claims.
5. **Required data:** Track A: present. Track B: new gridded datasets
   (see §7).
6. **Libraries/tools:** torch, scikit-image (PSNR/SSIM), xarray (Track B).
7. **Computational requirements:** Track A: minutes on CPU, faster on GPU.
   Track B: moderate; GPU useful.
8. **CPU/GPU/HPC:** Track B legitimately uses GPU. This is the natural home of
   the HPC label beyond preprocessing.
9. **Difficulty:** Medium (A), Hard (B).
10. **Effort:** A: 3–4 days. B: 2–4 weeks incl. data ingestion + evaluation.
11. **Scientific validity:** A: valid as methodology demonstration. B: valid
    only with independent evaluation data + a baseline (bicubic/bi-linear
    interpolation must be beaten).
12. **Evaluation:** A: PSNR/SSIM table. B: + comparison vs. interpolation
    baseline, spatial error maps.
13. **Output/visualization:** field reconstruction panels, error maps.
14. **Placement:** Advanced (Tier 3).
15. **Feasible now:** Track A yes; Track B **no — blocked on data**, this is
    stated loudly in the roadmap.

---

### E14 — CI pipeline (lint + test + smoke-run)

1. **Name:** GitHub Actions CI
2. **Problem it solves:** Nothing currently runs automatically; the machine
   here even lacks pip, so correctness is only syntax-checked.
3. **Relevance:** Portable proof the project works; catches regressions as
   E01–E13 land.
4. **Technical approach:** `.github/workflows/ci.yml`: Ubuntu + Python 3.11;
   `pip install -r requirements.txt`; `pytest`; a 30-second
   `python -m src.pipeline.run_pipeline` smoke; checksum assertion on raw
   data. Optional: CPU benchmark job.
5. **Required data:** Present in repo.
6. **Libraries/tools:** GitHub Actions (runners have pip/sudo — solves the
   local tooling gap for verification).
7. **Computational requirements:** ~3–5 CI minutes.
8. **CPU/GPU/HPC:** CPU runner.
9. **Difficulty:** Easy.
10. **Effort:** 1 day.
11. **Scientific validity:** n/a; reproducibility amplifier.
12. **Evaluation:** Green badge + reproducible steps.
13. **Output/visualization:** status badge in README.
14. **Placement:** Core (Tier 1).
15. **Feasible now:** Yes on GitHub.

---

### Tier 4 / optional pool (considered, not recommended for core)

| # | Idea | Why it was NOT prioritized |
|---|------|----------------------------|
| T4-1 | xarray/Dask chunked NetCDF pipeline | No gridded/NetCDF data exists; would be building for a dataset we don't have. *Reserve until E13-B is funded.* |
| T4-2 | Distributed training (DDP / multi-node/Horovod) | Dataset is 40K rows / ≤512² fields — distributed training adds latency, not value. Explicitly unjustified. |
| T4-3 | MLflow full experiment server | Overkill; JSON artifacts in E08 give the same value at this scale. |
| T4-4 | Spatial clustering of districts by climate-yield profile | Interesting but adds little inference we can't read from the atlas directly. |
| T4-5 | Anomaly-detection isolation forest on yield residuals | Deprecated by E11's MK-trend + SPI framework which is more interpretable. |
| T4-6 | Data augmentation / GAN for fields | Demonstrable but low scientific value without real target field | data; fancy ≠ valuable. |

---

## 3. Tier classification & rationale

### TIER 1 — MUST HAVE (completeness / correctness)

**E01 honest baselines, E02 grouped CV, E06 climate indicators, E07 data
validation, E08 experiment tracking + CI, E09 HPC benchmark harness (the
benchmark harness itself), E14 CI.**

Why: These convert a leaky notebook into a defensible project. Without E01/E02
every metric is suspect; without E07/E08 claims aren't reproducible; without
E09 there is no HPC at all; without E06 there is no climate science in a
"climate" project. E06 is included in Tier 1 because the title promises
climate, and indicators are cheap to compute from the one real dataset.

### TIER 2 — HIGH VALUE (strengthens the contribution)

**E03 tuning, E04 SHAP, E05 uncertainty, E10 geospatial maps, E11 vulnerability
atlas, E12 dashboard.**

Why: These add analytical depth and portfolio impact without new data beyond a
public shapefile (E10). E05's conformal coverage is the most "senior" scientific
addition in this tier; E11 is where the project starts answering its stated
questions.

### TIER 3 — ADVANCED (data-gated sophistication)

**E13 CNN downscaling Track B (real experiment).**

Why: The only genuinely advanced remain-advanced item, and it is gated on
gridded input data we do not have. Track A (demonstration + SR metrics) can run
now and is folded into Tier 2 effort. Track B is the single most defensible
"AI + HPC" centerpiece *if* data is acquired.

### TIER 4 — OPTIONAL

As listed in §2. Not needed to make the project substantial.

---

## 4. Recommended roadmap (order of implementation)

Numbering = dependency order. P-range phases each end in a reviewable commit.

| Phase | Work | Depends on | Est. effort | Exit gate |
|-------|------|-----------|-------------|-----------|
| **P1** | E01 honest baselines + E02 grouped CV (single `evaluate_models.py`, full test coverage) | done Phase 1 | 3–4 d | metric table RF vs XGB vs linear vs mean-baseline, honest R² reported |
| **P2** | E06 climate indicators (SPI, anomalies, MK trends, heat heuristic) + tests | P1 (uses corrected merged data) | 3–4 d | SPI known-value tests pass; indicator CSV generated |
| **P3** | E07 data ingestion/validation (pandera schemas, leak detector, Parquet + checksums) + E08 config/artifact logging | P1 | 4–5 d | `pytest` green; schema violations fail loudly; experiments/ artifacts reproducible |
| **P4** | E09 HPC benchmark harness (serial vs parallel tile preprocessing; GPU CNN timing when available) | P1 | 4–5 d | `benchmarks/` CSVs + speedup plots; README "what we parallelize" honest section |
| **P5** | E03 tuning + E04 SHAP + E05 conformal/QR uncertainty | P1–P2 | 5–6 d | tuned-model table under grouped CV; SHAP plots; 90% CI coverage ≥ ~0.87 on test years |
| **P6** | E10 geospatial maps + E11 vulnerability atlas | P1–P2, P5 | 5–6 d | district matches ≥ 95% reported; every "Findings" claim cites a CSV row |
| **P7** | E12 Streamlit dashboard + E13-A downscaling SR benchmark | P5–P6 | 5–7 d | `streamlit run` works on clean checkout; PSNR/SSIM table for SR demo |
| **P8** | E14 CI (can start earlier; wire after P3) + README rewrite + release | P3 | 2 d | green CI badge; polished README/LIMITATIONS |
| **Future** | E13-B real downscaling — **requires gridded data acquisition first** | external | 2–4 wk | only after data |

**Why this order:** correctness (P1) before climate (P2), both before
reproducibility (P3), engineering/HPC (P4) before tuning/explainability (P5),
geography/analysis (P6) and product (P7) last, since they consume the earlier
artifacts. Tiers are satisfied in order: T1 = P1–P4(+P8), T2 = P5–P7, T3 =
future E13-B.

---

## 5. Final project definition

- **Title:** *Crop Resilience Mapping Across Indian Districts: Leakage-Free ML
  Yield Modeling, Climate-Stress Indicators, and CNN-Based Spatial
  Downscaling (Demonstration)*
- **Problem statement:** Prior work reported R²≈0.97 on crop-yield prediction,
  but the score rested on target-derived nutrient features, static per-crop
  "climate" columns, a random time-leaking split, and a truncated, never-run
  downscaling CNN. The project must be rebuilt so that every claim is honest,
  reproducible, and evidence-backed.
- **Core research/engineering question:** *Given historical district yield
  records and a single-point regional climate record, what is the real,
  leakage-free predictive skill for yield shortfalls, can climate-stress
  indicators explain resilience gaps, and can CNN super-resolution be a
  defensible (rather than decorative) method for spatial rainfall
  downscaling?*
- **Input data:** (1) district crop dataset — after excluding leaky nutrient
  columns and treating static climate columns as crop-suitability constants;
  (2) NASA POWER single-point monthly record (1996–2020) for SPI, anomalies,
  trends, and yearly climate features; (3) optional public India-district
  boundaries for mapping; (4) optional gridded coarse/fine climate for a real
  downscaling experiment (future).
- **Processing pipeline:** ingest+validate (schemas, checksums) → aggregate
  NASA to yearly → merge → engineer leak-free features + climate indicators →
  grouped/temporal CV → train/baseline comparison → tune → explain (SHAP) →
  quantify uncertainty (conformal) → resilience mapping → vulnerability atlas →
  dashboard.
- **ML models:** Random Forest, XGBoost, linear, and the district–crop mean
  *baseline*; quantile/conformal variants for intervals.
- **Downscaling methodology:** SRCNN-style super-resolution CNN with measured
  PSNR/SSIM, framed openly as a *methodology demonstration* on an illustrative
  field built from the real NASA record; a validated experiment only if/when
  real coarse/fine gridded data is added.
- **HPC contribution:** measured parallelization of genuine workloads —
  multiprocessing over spatial tiles, GPU CNN training timing, and a benchmark
  harness publishing speedup/scaling tables; explicit statement of what does
  *not* justify HPC (40K-row tabular model, RF tuning).
- **Resilience methodology:** yield-gap ratio (actual / model-expected) with
  conformal interval width, threshold classes (≥0.9 / ≥0.7 / else), validated
  statistically against NASA drought/heat indicators — not claimed as
  agronomic advice.
- **Spatial analysis:** district-level choropleths of mean resilience and share
  of vulnerable crop-years; GIS join with reported match rate; GeoJSON exports.
- **Outputs:** processed CSVs per stage, model-comparison tables, CV results,
  SPI/anomaly/trend tables, SHAP panels, CI-vs-metric plots, benchmark tables,
  choropleth maps, vulnerability atlas, runnable Streamlit dashboard, and an
  auto-linked FINDINGS document.
- **Evaluation methodology:** grouped-CV metrics (mean±std) across models vs.
  naive baseline; conformal coverage on held-out years; SPI sanity vs. known dry
  years; join-coverage for maps; measured (never estimated) HPC speedups.
- **Limitations (stated in README):** single-point climate record applied
  nationally; static crop-climate constants; no true downscaling product until
  gridded data; no agronomic/decision-support claims; dashboard is
  presentation, not advice.
- **Future work:** E13-B real downscaling (data-gated), gridded input data
  ingestion, xarray/Dask path, DDP if/when scale justifies it.

---

## 6. Before vs. After

| Aspect | Current project | Proposed enhanced project |
|---|---|---|
| Model skill claim | R²≈0.9728 (circular, leaky) | Honest R²/RMSE vs. naive baseline under grouped CV, mean±std |
| Validation | Single random split | Year-blocked + district-blocked CV |
| Features | Includes target-derived `N/P/K` columns | Excludes them; leakage detector enforces it |
| Climate input | 2 temps + summed rain | SPI-3/6/12, anomalies, Mann-Kendall trends, heat heuristic (labeled) |
| Climate columns honesty | "district weather" implied | Documented as crop-suitability constants |
| Resilience | 12,121/12,131 "Highly Resilient" (vacuous) | Yield-gap ratio + CI width, spread classes, stress-correlation check |
| Uncertainty | None | Conformal 90% intervals, coverage validated |
| Explanability | Raw Gini importance | SHAP + permutation importance, sign-aware |
| Downscaling | Truncated stub, never run | Working SRCNN demo with PSNR/SSIM; real experiment data-gated |
| HPC | Label only | Measured tile-parallel preprocessing + GPU timing + benchmark tables |
| Geospatial | None | District choropleths, GeoJSON export, maps in dashboard |
| Dashboard | Closed `.pbix` binary | Reproducible Streamlit app |
| Data provenance | None | Download script, schemas, checksums, manifests |
| Reproducibility | Unpinned deps, `/content/` paths | `requirements.txt`, experiment configs+wheathered artifacts, CI |
| Tests | None | pytest across data/features/models/pipeline |
| Practical questions | Unanswerable | Vulnerability atlas answers each with a cited CSV row |

**Status legend:** ✅ exists (Phase 1) — corrected pipeline, leakage-free
features, temporal split, tests, CNN demo, structure. 🔧 requires repair —
nothing additional beyond what's in the roadmap now. 🆕 new — everything in
rows 4–17 above.

---

## 7. Data/library prerequisites (external)

- **Public district boundaries:** DataMeet maps
  (`github.com/datameet/maps` district GeoJSON) or Survey-of-India-derived
  boundaries — free, ~5–50 MB. Needed for E10/E12.
- **Gridded climate (E13-B only):** ERA5-Land / CHIRPS / IMD — free after
  registration / public archives. Not needed for the core roadmap.
- **Python packages to add:** `xgboost`, `geopandas`, `shap`, `pandera`,
  `pyarrow`, `streamlit`, `folium`, `pymannkendall` (or local MK), optional
  `optuna`. **torch remains manual install.**
- **Local tooling gap:** this machine has no pip/sudo — installation and
  execution must happen on a dev machine or in GitHub CI (runners have pip).

---

*End of Phase 0 enhancement roadmap. Nothing in this document has been
implemented; the repository remains untouched by it.*