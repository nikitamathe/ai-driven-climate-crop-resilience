"""Tests for E09 HPC tiling + benchmark harness (no PyTorch required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpc.benchmark import BenchConfig, WORKLOAD_COLUMNS, run_benchmarks
from hpc.tiling import (
    PROCESSORS,
    build_rainfall_field,
    partition_tiles,
    reassemble,
    process_futures,
    process_joblib,
    process_serial,
    tile_op,
)


class TestBuildRainfallField:
    def test_shape_and_dtype(self):
        f = build_rainfall_field(size=64, seed=42)
        assert f.shape == (64, 64)
        assert f.dtype == np.float32

    def test_deterministic_given_seed(self):
        a = build_rainfall_field(size=64, seed=42)
        b = build_rainfall_field(size=64, seed=42)
        assert np.array_equal(a, b)

    def test_non_negative(self):
        f = build_rainfall_field(size=128, seed=7)
        assert (f >= 0).all()


class TestPartition:
    def test_exact_grid_tile_count(self):
        f = build_rainfall_field(size=64, seed=42)
        tiles = partition_tiles(f, tile_size=16)
        assert len(tiles) == (64 // 16) ** 2  # 16

    def test_edge_partial_tiles(self):
        f = build_rainfall_field(size=50, seed=42)  # not a multiple of 16
        tiles = partition_tiles(f, tile_size=16)
        # 4 rows x 4 cols = 16 tiles, last row/col are partial
        assert len(tiles) == 4 * 4
        # last tile's height should be < 16
        assert tiles[-1].data.shape[0] == 50 - 48

    def test_reassemble_roundtrip(self):
        f = build_rainfall_field(size=64, seed=42)
        tiles = partition_tiles(f, tile_size=16)
        rebuilt = reassemble(tiles, f.shape)
        assert np.array_equal(rebuilt, f.astype(np.float64))


class TestBackendsMatch:
    """All processing backends must reproduce the serial (reference) result."""

    def _reference(self, size=64, tile=16):
        f = build_rainfall_field(size=size, seed=42)
        tiles = partition_tiles(f, tile_size=tile)
        return tiles, reassemble(process_serial(tiles, tile_op), f.shape)

    def test_multiprocessing_matches_serial(self):
        tiles, ref = self._reference()
        res = process_futures(tiles, workers=2, op=tile_op)
        assert np.allclose(reassemble(res, (64, 64)), ref, rtol=1e-6, atol=1e-6)

    def test_joblib_matches_serial(self):
        tiles, ref = self._reference()
        res = process_joblib(tiles, workers=2, op=tile_op)
        assert np.allclose(reassemble(res, (64, 64)), ref, rtol=1e-6, atol=1e-6)

    def test_all_dispatch_backends_match(self):
        tiles, ref = self._reference(size=32, tile=16)
        for name, proc in PROCESSORS.items():
            res = proc(tiles, workers=2, op=tile_op)
            rebuilt = reassemble(res, (32, 32))
            assert np.allclose(rebuilt, ref, rtol=1e-6, atol=1e-6), f"{name} mismatch"


class TestTileOp:
    def test_preserves_shape(self):
        d = np.ones((8, 8), dtype=np.float32)
        out = tile_op(d)
        assert out.shape == (8, 8)

    def test_box_local_mean_interior(self):
        d = np.full((8, 8), 6.0, dtype=np.float32)
        out = tile_op(d)
        # Interior pixels (not touching the zero-padded border) keep the value;
        # border pixels are pulled down by zero padding. Assert the interior.
        assert np.allclose(out[2:-2, 2:-2], 6.0)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

class TestRunBenchmarks:
    def test_schema_and_rows(self, tmp_path, monkeypatch):
        cfg = BenchConfig(grid_sizes=(32,), tile_size=16, worker_counts=(1,),
                          repeat=1, seed=42, run_id="unit-test")
        monkeypatch.chdir(tmp_path)  # keep any writes out of the repo
        df = run_benchmarks(cfg)
        assert list(df.columns) == WORKLOAD_COLUMNS

    def test_tile_rows_are_ok_and_serial_speedup_1(self, tmp_path, monkeypatch):
        cfg = BenchConfig(grid_sizes=(32,), tile_size=16, worker_counts=(1,),
                          repeat=1, seed=42, run_id="unit-test")
        monkeypatch.chdir(tmp_path)
        df = run_benchmarks(cfg)

        tile = df[df["workload"] == "tile"]
        assert not tile.empty
        assert (tile["status"] == "ok").all()
        serial = tile[tile["backend"] == "serial"]
        assert (serial["speedup_vs_serial"] == 1.0).all()
        assert (serial["wall_time_s"] > 0).all()

    def test_cnn_labelled_not_run_when_no_torch(self, tmp_path, monkeypatch):
        try:
            import torch  # noqa: F401
            pytest.skip("torch present; not-run path not exercised")
        except ImportError:
            pass
        cfg = BenchConfig(grid_sizes=(32,), tile_size=16, worker_counts=(1,),
                          repeat=1, seed=42, run_id="unit-test")
        monkeypatch.chdir(tmp_path)
        df = run_benchmarks(cfg)

        cnn = df[df["workload"] == "cnn"]
        assert not cnn.empty
        assert (cnn["status"] == "not run").all()
        # never fabricate numbers: NaN timing under the not-run label
        assert cnn["wall_time_s"].isna().all()
