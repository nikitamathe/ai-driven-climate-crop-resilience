"""Tests for E10 geospatial resilience maps module.

All geometry tests use small synthetic GeoDataFrames so the external district
boundary dataset is never required during unit testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization import geospatial as gs

NAME_MAP_PATH = Path(gs.NAME_MAP_PATH)


def _box_geom(x=0.0, y=0.0):
    from shapely.geometry import box

    return box(x, y, x + 1, y + 1)


@pytest.fixture
def synthetic_boundaries():
    """A tiny boundary GeoDataFrame using boundary-file names."""
    import geopandas as gpd

    rows = [
        {"ST_NM": "Punjab", "DISTRICT": "Bathinda", "geometry": _box_geom()},
        {"ST_NM": "Karnataka", "DISTRICT": "Bijapur", "geometry": _box_geom()},
        {"ST_NM": "Odisha", "DISTRICT": "Baleshwar", "geometry": _box_geom()},
        {"ST_NM": "Kerala", "DISTRICT": "Ernakulam", "geometry": _box_geom()},
        {"ST_NM": "Bihar", "DISTRICT": "Bhojpur", "geometry": _box_geom()},
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


@pytest.fixture
def synthetic_resilience():
    """Resilience rows using model (legacy) names, including filters."""
    return pd.DataFrame([
        {"State Name": "Punjab", "Dist Name": "Bhatinda", "Crop": "rice",
         "Year": 2000, "Resilience_Index": 1.2},
        {"State Name": "Punjab", "Dist Name": "Bhatinda", "Crop": "maize",
         "Year": 2001, "Resilience_Index": 0.5},
        {"State Name": "Karnataka", "Dist Name": "Bijapur / Vijayapura",
         "Crop": "cotton", "Year": 2000, "Resilience_Index": 0.8},
        {"State Name": "Orissa", "Dist Name": "Balasore", "Crop": "rice",
         "Year": 2000, "Resilience_Index": 1.0},
        {"State Name": "Kerala", "Dist Name": "Eranakulam", "Crop": "rice",
         "Year": 2001, "Resilience_Index": 0.3},
        {"State Name": "Bihar", "Dist Name": "Shahabad (now part of Bhojpur district)",
         "Crop": "rice", "Year": 2000, "Resilience_Index": 0.9},
    ])


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("Ananthapur", "Anantapur"),
        ("Kadapa YSR", "Y.s.r."),
        ("Bijapur / Vijayapura", "Bijapur"),
        ("Tiruchirapalli / Trichy", "Tiruchirappalli"),
        ("S.P.S. Nellore", "Sri Potti Sriramulu Nellore"),
        ("Mungair", "Munger"),
        ("Buland Shahar", "Bulandshahr"),
        ("24 Parganas", "24 Parganas"),      # no map -> passthrough (unmatched later)
    ])
    def test_legacy_map(self, raw, expected):
        assert gs.normalize_district_name(raw) == expected

    def test_whitespace_collapse(self):
        assert gs.normalize_district_name("  Mirzpur  ") == "Mirzapur"

    @pytest.mark.parametrize("raw,expected", [
        ("Orissa", "Odisha"),
        ("Telangana", "Andhra Pradesh"),
        ("Punjab", "Punjab"),
    ])
    def test_state_map(self, raw, expected):
        assert gs.normalize_state_name(raw) == expected

    def test_none_safe(self):
        assert gs.normalize_district_name(None) == ""
        assert gs.normalize_state_name(None) == ""


# --------------------------------------------------------------------------
# Merge + coverage
# --------------------------------------------------------------------------

class TestMergeCoverage:
    def test_merge_joins_spelled_variants(self, synthetic_boundaries,
                                          synthetic_resilience):
        merged, cov = gs.merge_resilience_with_geometry(
            synthetic_resilience, synthetic_boundaries)
        assert isinstance(cov, gs.JoinCoverage)
        # Bhattinda->Bathinda and Orissa->Odisha, Bijapur, Eranakulam join;
        # Shahabad residential district has no geometry -> unmatched.
        assert cov.matched_districts == 4
        assert cov.resilience_districts == 5
        assert "Shahabad (now part of Bhojpur district)" in cov.unmatched_resilience
        # merged frame carries only matched rows
        assert "Shahabad" not in set(merged["Dist Name"])
        assert len(merged) == 5

    def test_coverage_numbers(self, synthetic_boundaries, synthetic_resilience):
        _, cov = gs.merge_resilience_with_geometry(
            synthetic_resilience, synthetic_boundaries)
        assert cov.boundary_districts == 5
        assert cov.matched_districts == 4
        assert cov.coverage == pytest.approx(80.0)
        assert cov.match_percentage == pytest.approx(80.0)

    def test_unmatched_district_reported(self, synthetic_boundaries,
                                         synthetic_resilience):
        _, cov = gs.merge_resilience_with_geometry(
            synthetic_resilience, synthetic_boundaries)
        assert "Shahabad (now part of Bhojpur district)" in cov.unmatched_resilience
        # It should NOT appear in the matched set or the merged frame
        assert "Shahabad" not in [d for d in cov.unmatched_geometry]
        # covered districts are 4 distinct names


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

class TestAggregate:
    @pytest.fixture
    def merged_agg(self, synthetic_boundaries, synthetic_resilience):
        merged, _ = gs.merge_resilience_with_geometry(
            synthetic_resilience, synthetic_boundaries)
        return merged

    def test_mean_aggregation(self, merged_agg):
        agg = gs.aggregate_resilience(merged_agg)
        bhat = agg[agg["boundary_district"] == "Bathinda"].iloc[0]
        assert bhat["resilience_index_mean"] == pytest.approx((1.2 + 0.5) / 2)
        assert bhat["resilience_records"] == 2
        # Bhutan 1.2 & 0.5 -> 1 vulnerable -> 50 %
        assert bhat["pct_vulnerable"] == pytest.approx(50.0)

    def test_pct_vulnerable_threshold(self, merged_agg):
        agg = gs.aggregate_resilience(merged_agg)
        # Bhatinda 1.2 & 0.5 (only 0.5 < 0.7) -> 50%
        bhat = agg[agg["boundary_district"] == "Bathinda"].iloc[0]
        assert bhat["pct_vulnerable"] == pytest.approx(50.0)
        # Eranakulam 0.3 (< 0.7) -> 100%
        era = agg[agg["boundary_district"] == "Ernakulam"].iloc[0]
        assert era["pct_vulnerable"] == pytest.approx(100.0)

    def test_crop_filtering(self, merged_agg):
        agg = gs.aggregate_resilience(merged_agg, crop="rice")
        # only rice rows: Bhatinda rice(1.2) + Balasore rice(1.0) + Ernakulam rice
        bhat = agg[agg["boundary_district"] == "Bathinda"].iloc[0]
        assert bhat["resilience_records"] == 1
        assert bhat["resilience_index_mean"] == pytest.approx(1.2)

    def test_year_filtering(self, merged_agg):
        agg = gs.aggregate_resilience(merged_agg, year=2000)
        bhat = agg[agg["boundary_district"] == "Bathinda"].iloc[0]
        assert bhat["resilience_records"] == 1  # only year-2000 rice row remains
        assert bhat["resilience_index_mean"] == pytest.approx(1.2)

    def test_crop_and_year_together(self, merged_agg):
        agg = gs.aggregate_resilience(merged_agg, crop="maize", year=2001)
        bhat = agg[agg["boundary_district"] == "Bathinda"]
        assert bhat["resilience_index_mean"].iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

class TestExports:
    @pytest.fixture
    def agg(self, synthetic_boundaries, synthetic_resilience, tmp_path):
        merged, _ = gs.merge_resilience_with_geometry(
            synthetic_resilience, synthetic_boundaries)
        return gs.aggregate_resilience(merged)

    def test_geojson_export(self, agg, tmp_path):
        import geopandas as gpd
        out = gs.export_geojson(agg, tmp_path / "test.geojson")
        assert out.exists() and out.suffix == ".geojson"
        reloaded = gpd.read_file(out)
        assert len(reloaded) == len(agg)
        assert "resilience_index_mean" in reloaded.columns
        assert "geometry" in reloaded.geometry.name

    def test_choropleth_png(self, agg, tmp_path):
        import matplotlib.image as mpimg
        out = tmp_path / "idx.png"
        gs.plot_choropleth_index(agg, out)
        assert out.exists()
        img = mpimg.imread(out)
        assert img.ndim == 3

    def test_choropleth_vulnerability_png(self, agg, tmp_path):
        import matplotlib.image as mpimg
        out = tmp_path / "vuln.png"
        gs.plot_choropleth_vulnerability(agg, out)
        assert out.exists()
        img = mpimg.imread(out)
        assert img.ndim == 3

    def test_folium_html(self, agg, tmp_path):
        out = gs.plot_interactive_folium(agg, tmp_path / "map.html")
        assert out.exists() and out.suffix == ".html"
        text = out.read_text()
        assert "folium" in text or "leaflet" in text


# --------------------------------------------------------------------------
# State-level sanity
# --------------------------------------------------------------------------

def test_state_level_sanity(synthetic_boundaries, synthetic_resilience):
    """Aggregation at district level should carry the correct state names."""
    merged, _ = gs.merge_resilience_with_geometry(
        synthetic_resilience, synthetic_boundaries)
    agg = gs.aggregate_resilience(merged)
    # Orissa was remapped to Odisha; its aggregate should carry boundary_state Odisha
    odisha = agg[agg["boundary_district"] == "Baleshwar"].iloc[0]
    assert odisha["boundary_state"] == "Odisha"
