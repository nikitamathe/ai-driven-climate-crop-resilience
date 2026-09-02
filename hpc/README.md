# HPC Downscaling Code

This directory contains the E09 HPC module plus two related but **distinct**
CNN files.

## `HPC_Code_original.py` — recovered original prototype

The original `HPC_Code.txt` submitted with the project. It was recovered
verbatim during the Phase 0 audit (it had been truncated in the original
project at line 68 and never run).

- It references `crop_resilience_full_dataset.csv`, which does not exist in
  the project.
- It expects `Latitude` / `Longitude` columns that the data does not contain.
- Its training loop is incomplete (the `for epoch in range(epochs):` body is
  empty).
- **It is an incomplete, non-runnable prototype** and is preserved here for
  reference only.

## `downscale_cnn.py` — reconstructed implementation

A complete, runnable super-resolution CNN written during the rebuild. It
loads the real NASA POWER record from `data/raw/nasa_power/`, builds an
illustrative rainfall field, trains with a real training loop, and saves model
and field outputs.

## Important

These two CNN files must **not** be presented as the same implementation.
`downscale_cnn.py` shares the SRCNN-style model concept of the original but is
otherwise new code. The original prototype is archived as-is; it is not part
of the active pipeline. `benchmark.py` reuses `DownscaleCNN` (imported lazily)
for Workload B — it does not duplicate the model.

## Scope

`downscale_cnn.py` demonstrates super-resolution methodology **only**. It is
not a validated real spatial climate-downscaling system (there is no gridded
coarse/fine climate data in this repository), and any claims beyond that scope
are engineering benchmarks (E09), not scientific climate results.

## `tiling.py` + `benchmark.py` — HPC made real (E09)

**HPC: what we parallelize and why.** The only workload here that genuinely
benefits from parallelism is the grid (CNN-style) pipeline, where a 2-D rainfall
field is partitioned into `N x N` tiles and a CPU-bound op is applied to each
tile. That op is embarrassingly parallel — tiles do not communicate — so it is
the rare place where a process pool yields a real speedup.

* **Processes, not threads.** The tile op is CPU-bound Python/Numpy, so the
  Global Interpreter Lock (GIL) caps a thread pool at ~1 core. We therefore use
  `ProcessPoolExecutor` (and `joblib`'s `loky` backend) to sidestep the GIL.
* **Measured, never estimated.** `benchmark.py` runs both the serial baseline
  and the parallel backends on the same grids on this CPU, records real wall
  time, and computes throughput and speedup relative to the *measured* serial
  time at the same grid size. No "expected" or idealized speedups are shown as
  measured.
* **Overhead and contention.** Process pools pay IPC + serialization cost for
  each tile, and this 2-core host cannot exceed ~2x serial. At small grids the
  overhead dominates and measured speedup can be below 1.0 — that is the honest
  result (see `data/processed/benchmarks/`), not a bug.
* **Amdahl / scaling sanity check.** Speedup is bounded by 1 / (1 − f) where f
  is the parallelizable fraction. With the data-transfer + reduce overhead, a
  ~2x process count yields far less than 2x — consistent with Amdahl and with
  the measured sub-linear scaling (see `scaling_by_grid_size.png`).
* **Why NOT MPI.** The tiled grids are a handful of in-RAM arrays (≤ 512²), far
  below the size where message passing across nodes pays for itself. MPI is
  explicitly *not justified* at this scale; a process pool on one node is the
  right tool. (See roadmap T2/4 and E09 §8.)

**Workload B (CNN timing).** `benchmark.py` times the existing `DownscaleCNN`
(imported lazily, never duplicated) on CPU when PyTorch is present. GPU rows are
written only when CUDA is detected; otherwise they are labelled `status="not
run"` — never fabricated.

**Run it:**
```bash
python -m pytest tests/test_hpc.py -v            # correctness of tiling + backends
python -m hpc.benchmark --grid-sizes 64,128,256 --workers 1,2 --repeat 2
```
Outputs (CSV + `speedup_vs_workers.png` + `scaling_by_grid_size.png`) land in
`data/processed/benchmarks/<run>/` and are git-ignored (generated artifacts).

**Engineering vs. scientific claims.** These are *engineering* performance
measurements on synthetic fields. They do NOT make the climate data spatially
resolved: E06 remains a single-point NASA POWER series, and no spatial
downscaling of real data is claimed (see roadmap Ground Truth).