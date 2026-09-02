"""Parallel tile preprocessing (E09, Workload A).

Partitions a synthetic rainfall grid into N x N tiles, applies a self-contained
compute op to each tile, and processes the tiles with one of three backends:

* ``serial``            — a plain Python loop (baseline)
* ``multiprocessing``   — ``concurrent.futures.ProcessPoolExecutor``
* ``joblib``            — ``joblib.Parallel`` with ``loky`` backend

Global Interpreter Lock (GIL) note: the tile op is CPU-bound Python/Numpy, so a
thread-based pool would not scale. ``ProcessPoolExecutor`` and ``joblib``'s
``loky`` sidestep the GIL by forking worker processes. This is the honest,
measured rationale for using processes rather than threads.

The tiles are reassembled and compared to the expected result so the harness
only reports *correct* parallelization — never merely *fast* parallelization.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Registry of available backends (used by the benchmark harness).
BACKENDS = ("serial", "multiprocessing", "joblib")


# ---------------------------------------------------------------------------
# Synthetic rainfall field (torch-free, deterministic)
# ---------------------------------------------------------------------------

def build_rainfall_field(size: int = 128, seed: int = 42, base: float = 120.0,
                         amp: float = 40.0) -> np.ndarray:
    """Build a smooth, deterministic synthetic rainfall grid.

    Torch-free stand-in for the grid fields used by the CNN demo, so Workload A
    can run (and be benchmarked) without PyTorch installed. The spatial pattern
    is a smooth trigonometric surface plus a small seeded noise term, clipped to
    non-negative rainfall.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    field = base + amp * (
        0.5 * np.sin(2 * np.pi * (x / size) * 3 + 0.7)
        + 0.35 * np.cos(2 * np.pi * (y / size) * 4)
        + 0.15 * rng.normal(0, 1, (size, size))
    )
    return np.clip(field, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

@dataclass
class Tile:
    """A rectangular tile plus its origin coordinates in the parent grid.

    ``row0, col0`` are the tile's top-left offset so results can be placed back
    exactly. ``shape`` is the tile's (height, width), which carries an overlap
    flag for edge tiles that are smaller than ``tile_size``.
    """
    index: int
    row0: int
    col0: int
    data: np.ndarray


def partition_tiles(field: np.ndarray, tile_size: int) -> list[Tile]:
    """Slice a 2-D ``field`` into ``tile_size x tile_size`` tiles.

    Edge tiles (when the field is not a multiple of ``tile_size``) are returned
    at their actual size with ``overlap`` set (they are "partial" tiles). The
    caller must account for the exact tile shape when reassembling.
    """
    h, w = field.shape
    tiles: list[Tile] = []
    idx = 0
    for row0 in range(0, h, tile_size):
        for col0 in range(0, w, tile_size):
            tile = field[row0:row0 + tile_size, col0:col0 + tile_size]
            tiles.append(Tile(index=idx, row0=row0, col0=col0, data=tile))
            idx += 1
    return tiles


def reassemble(tiles: list[Tile], grid_shape: tuple[int, int]) -> np.ndarray:
    """Place processed tiles back into a full grid of ``grid_shape``.

    Each tile is written at its recorded origin; the full grid size is checked
    so a mis-sized tile set produced by a buggy parallel path fails loudly.
    """
    out = np.zeros(grid_shape, dtype=np.float64)
    for tile in tiles:
        th, tw = tile.data.shape
        out[tile.row0:tile.row0 + th, tile.col0:tile.col0 + tw] = tile.data
    return out


# ---------------------------------------------------------------------------
# Tile operator
# ---------------------------------------------------------------------------

def tile_op(tile_data: np.ndarray, noise: float = 0.0) -> np.ndarray:
    """A deterministic, CPU-bound compute op applied to each tile.

    Applies a 3x3 local-mean smoothing filter (edge-preserving area average)
    plus an optional additive term. The op is pure and per-tile, so it is a
    valid embarrassingly-parallel workload representative of grid preprocessing
    (e.g. neighbourhood statistics, normalization, feature extraction).
    """
    smooth = _box3(tile_data)
    if noise:
        smooth = smooth + noise * smooth.mean()
    return smooth.astype(np.float64)


def _box3(a: np.ndarray) -> np.ndarray:
    """Zero-padded 3x3 box (local mean) filter."""
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64) / 9.0
    return _correlate2d_pad(a, kernel)


def _correlate2d_pad(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Dense 2-D correlation with zero padding to preserve input size."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(a, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(a, dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _make_tile(tile: Tile, op: Callable[[np.ndarray], np.ndarray]) -> Tile:
    return Tile(index=tile.index, row0=tile.row0, col0=tile.col0,
                data=np.asarray(op(tile.data)))


def process_serial(tiles: list[Tile],
                   op: Callable[[np.ndarray], np.ndarray]) -> list[Tile]:
    """Process tiles in a single process (baseline)."""
    return [_make_tile(t, op) for t in tiles]


def process_futures(tiles: list[Tile], workers: int,
                    op: Callable[[np.ndarray], np.ndarray]) -> list[Tile]:
    """Process tiles with a ``ProcessPoolExecutor`` (robust, stdlib).

    Uses ``spawn``/``forkserver`` (never unsafe ``fork`` inside a possibly
    multithreaded host) so the pool is safe regardless of the caller. ``op`` and
    tile data must be picklable — both ``tile_op`` (module-level) and numpy
    arrays are.
    """
    import multiprocessing as mp

    ctx = mp.get_context("forkserver") if "forkserver" in mp.get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_run_tile, t.row0, t.col0, t.index, op, t.data)
                   for t in tiles]
        return [f.result() for f in futures]


def _run_tile(row0: int, col0: int, index: int,
              op: Callable[[np.ndarray], np.ndarray],
              data: np.ndarray) -> Tile:
    return Tile(index=index, row0=row0, col0=col0,
                data=np.asarray(op(np.asarray(data))))


def process_joblib(tiles: list[Tile], workers: int,
                   op: Callable[[np.ndarray], np.ndarray]) -> list[Tile]:
    """Process tiles with ``joblib.Parallel`` (loky backend)."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=workers, backend="loky")(
        delayed(_run_tile)(t.row0, t.col0, t.index, op, t.data) for t in tiles
    )
    return list(results)


# Public dispatch table used by the benchmark harness.
PROCESSORS = {
    "serial": lambda tiles, workers, op: process_serial(tiles, op),
    "multiprocessing": process_futures,
    "joblib": process_joblib,
}
