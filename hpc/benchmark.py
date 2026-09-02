"""HPC benchmark harness (E09).

Measures, rather than fabricates, two parallel workloads and writes the results
to ``data/processed/benchmarks/<run>.csv`` together with two diagnostic plots
(``speedup_vs_workers.png``, ``scaling_by_grid_size.png``).

Workload A — parallel tile preprocessing
    A synthetic rainfall grid is partitioned into N x N tiles and each tile is
    processed by a CPU-bound op. Wall time is measured for the *serial*
    baseline and for ``multiprocessing``/``joblib`` process pools across a grid
    of tile counts and worker counts. Speedup is computed relative to the serial
    baseline at the same grid size.

Workload B — CNN training timing
    If PyTorch is installed, the SRCNN demo (``hpc.downscale_cnn.DownscaleCNN``)
    is trained for a few epochs on several field sizes (64^2 .. 512^2) on the
    available device, recording wall time per epoch. If PyTorch is NOT
    installed, these rows are written with ``status = "not run"`` — never a
    fabricated number.

Honesty rules
    * Numbers are always measured. Absent hardware results are labelled
      ``not run`` and excluded from speedup/scaling plots.
    * Speedup (Workload A) uses the *measured serial* time at the same grid
      size, so Amdahl-style contention is reported, not idealized.
    * MPI is *not* justified for these workloads at this scale: the dataset is a
      handful of in-RAM grids, not a distributed mesh. This is stated explicitly
      in the output and README rather than bolted on for ceremony.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

# Allow running as a script from the repo root (python hpc/benchmark.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hpc.tiling import (
    PROCESSORS,
    build_rainfall_field,
    partition_tiles,
    process_serial,
    reassemble,
    tile_op,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "data" / "processed" / "benchmarks"

WORKLOAD_COLUMNS = [
    "run_id", "workload", "backend", "grid_size", "tiles", "workers",
    "wall_time_s", "throughput_tiles_s", "speedup_vs_serial",
    "device", "status",
]


@dataclass
class BenchConfig:
    """Configuration for a single benchmark execution (Workload A grid)."""
    grid_sizes: tuple[int, ...] = (64, 128, 256)
    tile_size: int = 32
    worker_counts: tuple[int, ...] = (1, 2, 4)
    op_noise: float = 0.0
    repeat: int = 2
    seed: int = 42
    run_id: str = "local"


# ---------------------------------------------------------------------------
# Workload A: parallel tile preprocessing
# ---------------------------------------------------------------------------

def _bench_tile_a(config: BenchConfig) -> list[dict]:
    """Benchmark tile preprocessing across grid sizes x backends x workers.

    Returns rows (dicts), one per (grid_size, backend, workers) combination.
    ``speedup_vs_serial`` is the ratio of the serial wall time to the backend's
    wall time at the *same* grid size.
    """
    rows: list[dict] = []
    op = _op_for(config)

    for grid in config.grid_sizes:
        field = build_rainfall_field(size=grid, seed=config.seed)
        tiles = partition_tiles(field, config.tile_size)
        n_tiles = len(tiles)
        # Reference result: the (correct-by-construction) serial reassembly.
        expected = reassemble(process_serial(tiles, op), field.shape)

        # Serial baseline (workers meaningless; record serial timing).
        serial_time = _time_op(lambda: process_serial(tiles, op), config.repeat)
        ok = _verify(process_serial(tiles, op), field.shape, expected)
        rows.append(_row(config, "tile", "serial", grid, n_tiles, 1, serial_time, ok,
                         speedup=1.0))

        for backend in ("multiprocessing", "joblib"):
            for workers in config.worker_counts:
                proc = PROCESSORS[backend]
                t = _time_op(lambda: proc(tiles, workers, op), config.repeat)
                result_tiles = proc(tiles, workers, op)
                ok = _verify(result_tiles, field.shape, expected)
                speedup = serial_time / t if t > 0 else float("nan")
                rows.append(_row(config, "tile", backend, grid, n_tiles, workers, t, ok, speedup))
    return rows


def _op_for(config) -> partial:
    """Return a picklable tile operator honouring ``config.op_noise``.

    ``functools.partial(tile_op, noise=...)`` of a module-level function is
    picklable, so it can be shipped to process-pool workers.
    """
    return partial(tile_op, noise=config.op_noise) if config.op_noise else tile_op


def _time_op(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _verify(tiles, grid_shape, expected) -> bool:
    import numpy as np
    try:
        rebuilt = reassemble(tiles, grid_shape)
        return bool(np.allclose(rebuilt, expected, rtol=1e-6, atol=1e-6))
    except Exception:
        return False


def _row(config: BenchConfig, workload: str, backend: str, grid: int, n_tiles: int,
         workers: int, wall_time: float, ok: bool, speedup: float) -> dict:
    return {
        "run_id": config.run_id,
        "workload": workload,
        "backend": backend,
        "grid_size": grid,
        "tiles": n_tiles,
        "workers": workers,
        "wall_time_s": round(wall_time, 6),
        "throughput_tiles_s": round(n_tiles / wall_time, 3) if wall_time > 0 else 0.0,
        "speedup_vs_serial": round(speedup, 4),
        "device": "cpu",
        "status": "ok" if ok else "mismatch",
    }


# ---------------------------------------------------------------------------
# Workload B: CNN training timing (torch-optional)
# ---------------------------------------------------------------------------

def _bench_cnn_b(config: BenchConfig) -> list[dict]:
    """Time SRCNN training per epoch on several field sizes.

    Returns rows for each field size. If PyTorch is unavailable every row is
    written with ``status = "not run"`` and NaN timings — labelled, not faked.
    """
    try:
        import torch
    except ImportError:
        return [
            {
                "run_id": config.run_id, "workload": "cnn", "backend": "cpu",
                "grid_size": g, "tiles": 1, "workers": 1,
                "wall_time_s": float("nan"), "throughput_tiles_s": float("nan"),
                "speedup_vs_serial": float("nan"), "device": "torch-not-installed",
                "status": "not run",
            }
            for g in (64, 128, 256, 512)
        ]

    from hpc.downscale_cnn import DownscaleCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(config.seed)
    rows_out = []
    for g in (64, 128, 256, 512):
        field = build_rainfall_field(size=g, seed=config.seed)
        # Fine target == source field (identity super-resolution) so timing
        # reflects the compute cost of the architecture, not the demo's pair
        # generation. Trains the fixed SRCNN on the field.
        fine_t = torch.from_numpy(field)[None, None].float().to(device)
        model = DownscaleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        epochs = 2
        start = time.perf_counter()
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(fine_t)
            loss = loss_fn(out, fine_t)
            loss.backward()
            optimizer.step()
        per_epoch = (time.perf_counter() - start) / epochs
        rows_out.append({
            "run_id": config.run_id, "workload": "cnn", "backend": "device",
            "grid_size": g, "tiles": 1, "workers": 1,
            "wall_time_s": round(per_epoch, 6),
            "throughput_tiles_s": round(1.0 / per_epoch, 3) if per_epoch > 0 else 0.0,
            "speedup_vs_serial": float("nan"), "device": device_name, "status": "ok",
        })
    return rows_out


# ---------------------------------------------------------------------------
# Orchestration + outputs
# ---------------------------------------------------------------------------

def run_benchmarks(config: BenchConfig | None = None) -> pd.DataFrame:
    """Run Workload A and (where possible) Workload B, returning the results DF."""
    config = config or BenchConfig()
    rows = _bench_tile_a(config)
    rows += _bench_cnn_b(config)
    df = pd.DataFrame(rows, columns=WORKLOAD_COLUMNS)
    return df


def _plot_bound(df: pd.DataFrame, out_dir: Path, run_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # speedup_vs_workers for tile workload at the largest grid size
    tile = df[(df["workload"] == "tile") & (df["status"] == "ok")]
    if not tile.empty and tile["speedup_vs_serial"].notna().any():
        largest = tile["grid_size"].max()
        sub = tile[tile["grid_size"] == largest]
        fig, ax = plt.subplots(figsize=(7, 5))
        for backend in ("multiprocessing", "joblib"):
            b = sub[sub["backend"] == backend]
            if b.empty:
                continue
            b = b.sort_values("workers")
            ax.plot(b["workers"], b["speedup_vs_serial"], marker="o", label=backend)
        ax.plot([1, max(sub["workers"].max(), 1)], [1, max(sub["workers"].max(), 1)],
                ls="--", color="grey", label="ideal")
        ax.set_xlabel("workers"); ax.set_ylabel("speedup vs serial")
        ax.set_title(f"Speedup vs workers (grid {largest}) {run_id}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "speedup_vs_workers.png", dpi=120)
        plt.close(fig)

    # scaling_by_grid_size: serial + best-parallel wall time by grid size
    if not tile.empty:
        serial = tile[tile["backend"] == "serial"].set_index("grid_size")["wall_time_s"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(serial.index, serial.values, marker="o", label="serial")
        mp = tile[tile["backend"] == "multiprocessing"].groupby("grid_size")["wall_time_s"].min()
        ax.plot(mp.index, mp.values, marker="s", label="best multiprocessing")
        ax.set_xlabel("grid size"); ax.set_ylabel("best wall time (s)")
        ax.set_title(f"Scaling by grid size {run_id}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "scaling_by_grid_size.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=str, default="64,128,256",
                        help="Comma-separated grid sizes to benchmark")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--workers", type=str, default="1,2,4",
                        help="Comma-separated worker counts")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None,
                        help="Run id used for output dir; defaults to timestamp")
    parser.add_argument("--cnn", action="store_true",
                        help="Run Workload B (CNN timing) even if torch is present")
    args = parser.parse_args(argv)

    grid_sizes = tuple(int(x) for x in args.grid_sizes.split(","))
    worker_counts = tuple(int(x) for x in args.workers.split(","))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = BenchConfig(
        grid_sizes=grid_sizes,
        tile_size=args.tile_size,
        worker_counts=worker_counts,
        repeat=args.repeat,
        seed=args.seed,
        run_id=run_id,
    )

    out_dir = BENCH_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_benchmarks(config)
    csv_path = out_dir / f"{run_id}.csv"
    df.to_csv(csv_path, index=False)

    _plot_bound(df, out_dir, run_id)

    # Explicit MPI rationale
    print("\n[HPC] MPI note: not justified at this scale — the tiled grids are a")
    print("      handful of in-RAM arrays (<=512^2). ProcessPool/joblib suffice;")
    print("      no multi-node message-passing setup is claimed (E09).")

    print(f"\n[HPC] wrote benchmarks to {out_dir}")
    print(f"  - {csv_path}")
    print(f"  - {out_dir / 'speedup_vs_workers.png'}")
    print(f"  - {out_dir / 'scaling_by_grid_size.png'}")

    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2, sort_keys=True)

    print(f"\nWorkload A summary (median of runs by backend):")
    summary = (df[df["workload"] == "tile"]
               .groupby("backend")["wall_time_s"]
               .median())
    print(summary.to_string())

    # Show which rows are labelled not-run / mismatch so nothing is silently dropped
    non_ok = df[df["status"] != "ok"]
    if not non_ok.empty:
        print(f"\nRows NOT 'ok' (labelled honestly, excluded from plots):")
        print(non_ok[["workload", "grid_size", "backend", "status"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
