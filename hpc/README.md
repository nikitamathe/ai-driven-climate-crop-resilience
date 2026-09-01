# HPC Downscaling Code

This directory contains two related but **distinct** files.

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

These two files must **not** be presented as the same implementation.
`downscale_cnn.py` shares the SRCNN-style model concept of the original but is
otherwise new code. The original prototype is archived as-is; it is not part
of the active pipeline.

## Scope

`downscale_cnn.py` demonstrates super-resolution methodology **only**. It is
not a validated real spatial climate-downscaling system (there is no gridded
coarse/fine climate data in this repository), and no parallel/HPC speedup has
been measured or claimed. Any "HPC" or "downscaling" claims beyond this scope
are roadmap items (see `docs/ENHANCEMENT_ROADMAP.md`), not features of this
code.