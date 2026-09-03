"""E10 geospatial resilience maps (GeoPandas) — district-level choropleths.

This module is a **visualization/aggregation layer** over the existing model
output (`data/processed/final_crop_resilience_district_year.csv`). It joins the
district-level resilience summary to India district boundary geometry and
produces choropleth maps, GeoJSON, and interactive HTML.

Scientific scope / honesty:
    * This is NOT climate downscaling. It does not create new spatial climate
      observations.
    * The NASA POWER climate input remains a single spatial point (see E06);
      the maps spatialize *model-output* resilience by administrative boundary.
    * Spatial validity depends entirely on the district-name join; unmatched
      districts are always counted and reported (see ``calculate_join_coverage``).

Geospatial dependencies (geopandas / folium / pyarrow) are imported **lazily**,
so importing this module or running the core pipeline never requires them.

Run as a standalone command from the project root::

    python -m src.visualization.geospatial \
        --boundaries data/geodata/2011_Dist.shp \
        --out-dir data/processed/geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAME_MAP_PATH = PROJECT_ROOT / "data" / "geo" / "name_mapping.json"
DEFAULT_BOUNDARIES = PROJECT_ROOT / "data" / "geodata" / "2011_Dist.shp"
DEFAULT_RESILIENCE = (
    PROJECT_ROOT / "data" / "processed" / "final_crop_resilience_district_year.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "geojson"

_STATE_COL = "State Name"
_DISTRICT_COL = "Dist Name"
_BOUNDARY_STATE_COL = "ST_NM"
_BOUNDARY_DISTRICT_COL = "DISTRICT"

# Matches src.models.evaluate.resilience_class's "Vulnerable" threshold (0.7),
# so % vulnerable here is consistent with the Resilience_Class column.
_VULNERABLE_THRESHOLD = 0.7


def _load_name_map() -> dict:
    with open(NAME_MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_state_name(name: str) -> str:
    """Return the boundary-file state name for a model state name.

    Applies the explicit state remap (``Orissa -> Odisha``,
    ``Telangana -> Andhra Pradesh``) then a casefold/whitespace normalization.
    """
    if name is None:
        return ""
    raw = " ".join(str(name).split()).strip()
    state_map = _load_name_map()["states"]
    mapped = state_map.get(raw, state_map.get(raw.casefold()))
    if mapped:
        return mapped
    return raw


def normalize_district_name(name: str) -> str:
    """Return the boundary-file district name for a model district name.

    1. Looks up the explicit, reviewable map in ``data/geo/name_mapping.json``
       (covers legacy spellings and slash-separated dual names).
    2. Falls back to a casefold/whitespace-normalized version of the input.
    """
    if name is None:
        return ""
    raw = " ".join(str(name).split()).strip()
    district_map = _load_name_map()["districts"]
    mapped = district_map.get(raw, district_map.get(raw.casefold()))
    if mapped:
        return mapped
    return raw


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_district_boundaries(path=DEFAULT_BOUNDARIES):
    """Load the district boundary shapefile/GeoJSON as a GeoDataFrame.

    Returns a GeoDataFrame with columns ``DISTRICT`` (boundary district name),
    ``ST_NM`` (boundary state name), and ``geometry``. Requires geopandas.
    """
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[[_BOUNDARY_STATE_COL, _BOUNDARY_DISTRICT_COL, "geometry"]].copy()
    return gdf


# ---------------------------------------------------------------------------
# Merge + coverage
# ---------------------------------------------------------------------------

@dataclass
class JoinCoverage:
    """Counts describing how many districts merged successfully."""

    boundary_districts: int
    resilience_districts: int
    matched_districts: int
    unmatched_resilience: list
    unmatched_geometry: list
    match_percentage: float

    @property
    def matched(self) -> int:
        return self.matched_districts

    @property
    def total(self) -> int:
        return self.resilience_districts

    @property
    def coverage(self) -> float:
        return self.match_percentage


def _canonical_state(s: str) -> str:
    return " ".join(str(s).split()).strip().upper()


def merge_resilience_with_geometry(resilience_df, boundaries, coverage=True):
    """Join model resilience rows to boundary geometry by normalized names.

    Adds ``boundary_district`` and ``boundary_state`` columns carrying the
    matched geometry names, then performs an inner merge (left-join on the
    *resilience* side restricted to matched rows) with the boundary geometry.
    Returns ``(merged_gdf, coverage_record)``.
    """
    import geopandas as gpd

    resilience = resilience_df.copy()
    resilience["_norm_state"] = resilience[_STATE_COL].map(normalize_state_name)
    resilience["_norm_district"] = resilience[_DISTRICT_COL].map(
        normalize_district_name
    )

    boundaries = boundaries.copy()
    boundaries["_norm_state"] = boundaries[_BOUNDARY_STATE_COL].map(
        lambda s: " ".join(str(s).split()).strip()
    )
    boundaries["_norm_district"] = boundaries[_BOUNDARY_DISTRICT_COL].map(
        lambda s: " ".join(str(s).split()).strip()
    )

    boundary_lookup = {}
    for _, row in boundaries.iterrows():
        key = (_canonical_state(row["_norm_state"]), _canonical_state(row["_norm_district"]))
        if key not in boundary_lookup:
            boundary_lookup[key] = row["geometry"]

    matched_keys = set(resilience[["_norm_state", "_norm_district"]].apply(
        lambda r: (_canonical_state(r["_norm_state"]),
                   _canonical_state(r["_norm_district"])), axis=1
    ))
    matched_keys &= set(boundary_lookup.keys())

    merged = resilience[resilience[["_norm_state", "_norm_district"]].apply(
        lambda r: (_canonical_state(r["_norm_state"]),
                   _canonical_state(r["_norm_district"])) in matched_keys, axis=1
    )].copy()
    merged["geometry"] = merged[["_norm_state", "_norm_district"]].apply(
        lambda r: boundary_lookup[
            (_canonical_state(r["_norm_state"]),
             _canonical_state(r["_norm_district"]))
        ], axis=1
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["boundary_district"] = merged["_norm_district"]
    merged["boundary_state"] = merged["_norm_state"]

    cov = None
    if coverage:
        # Map canonical (state, district) keys back to original display names so
        # the unmatched report lists readable model names.
        key_to_orig = {}
        for _, r in resilience.iterrows():
            key = (_canonical_state(r["_norm_state"]),
                   _canonical_state(r["_norm_district"]))
            key_to_orig.setdefault(key, r[_DISTRICT_COL])
        cov = calculate_join_coverage(
            matched_keys,
            set(boundary_lookup.keys()),
            set(matched_keys) | (set(resilience[["_norm_state", "_norm_district"]]
                                    .apply(lambda r: (_canonical_state(r["_norm_state"]),
                                                     _canonical_state(r["_norm_district"])),
                                           axis=1)) - set(boundary_lookup.keys())),
            original_names=key_to_orig,
        )
    return merged, cov


def calculate_join_coverage(resilience_keys, boundary_keys, row_keys=None,
                            original_names: dict | None = None):
    """Compute join-coverage counts from normalized key sets.

    ``resilience_keys`` — the matched (state, district) key set.
    ``boundary_keys`` — the set of (state, district) keys in the boundary file.
    ``row_keys`` — the full set of (state, district) keys in the resilience
    frame (matched + unmatched). ``original_names`` optionally maps canonical
    keys back to the original display names for readable reporting.
    """
    row_keys = resilience_keys if row_keys is None else row_keys
    matched = resilience_keys & boundary_keys
    unmatched_res_keys = row_keys - boundary_keys

    def _names(keys):
        names = []
        for key in sorted(keys, key=lambda k: (k[0], k[1])):
            if original_names:
                names.append(original_names.get(key, key[1]))
            else:
                names.append(key[1])
        return names

    unmatched_res = _names(unmatched_res_keys)
    unmatched_geo = sorted(
        {str(name) for _, name in (boundary_keys - row_keys) if name}
    )
    pct = 100.0 * len(matched) / max(1, len(row_keys))
    return JoinCoverage(
        boundary_districts=len(boundary_keys),
        resilience_districts=len(row_keys),
        matched_districts=len(matched),
        unmatched_resilience=unmatched_res,
        unmatched_geometry=unmatched_geo,
        match_percentage=round(pct, 2),
    )


def report_coverage(cov: JoinCoverage) -> None:
    """Print a human-readable join-coverage report (E10 honesty requirement)."""
    print("\nE10 district join coverage:")
    print(f"  Boundary districts: {cov.boundary_districts}")
    print(f"  Resilience districts: {cov.resilience_districts}")
    print(f"  Matched: {cov.matched_districts}")
    print(f"  Coverage: {cov.coverage:.2f}%")
    print(f"  Unmatched resilience districts ({len(cov.unmatched_resilience)}):")
    for name in cov.unmatched_resilience:
        print(f"    - {name}")
    print(f"  Unmatched geometry-only districts ({len(cov.unmatched_geometry)}):")
    for name in cov.unmatched_geometry[:40]:
        print(f"    - {name}")
    if len(cov.unmatched_geometry) > 40:
        print(f"    ... and {len(cov.unmatched_geometry) - 40} more")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_resilience(merged_gdf, crop: str | None = None,
                         year: int | None = None) -> "gpd.GeoDataFrame":
    """Aggregate resilience to the district level from a merged GeoDataFrame.

    ``merged_gdf`` must carry ``boundary_district`` / ``boundary_state`` keys.
    Returns a GeoDataFrame with one row per district containing:
      ``resilience_index_mean``, ``resilience_index_min``,
      ``resilience_index_max``, ``resilience_records``, ``pct_vulnerable``.
    Optional ``crop`` and ``year`` filters are applied before aggregating.
    """
    import geopandas as gpd

    df = merged_gdf.drop(columns=["geometry"])
    if crop is not None:
        df = df[df["Crop"].astype(str).str.lower() == str(crop).lower()]
    if year is not None:
        df = df[df["Year"] == int(year)]

    grouped = df.groupby(["boundary_state", "boundary_district"]).agg(
        resilience_index_mean=("Resilience_Index", "mean"),
        resilience_index_min=("Resilience_Index", "min"),
        resilience_index_max=("Resilience_Index", "max"),
        resilience_records=("Resilience_Index", "size"),
        pct_vulnerable=("Resilience_Index",
                        lambda s: 100.0 * (s < _VULNERABLE_THRESHOLD).mean()),
    ).reset_index()

    # Reattach one geometry per district for spatial output.
    unique_geom = (
        merged_gdf[["boundary_state", "boundary_district", "geometry"]]
        .drop_duplicates(["boundary_state", "boundary_district"])
    )
    out = grouped.merge(unique_geom, on=["boundary_state", "boundary_district"])
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Choropleths
# ---------------------------------------------------------------------------

def _setup_choropleth_axes(gdf, value_col, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=value_col, ax=ax, legend=True,
             cmap="RdYlGn", edgecolor="white", linewidth=0.2,
             legend_kwds={"shrink": 0.6, "label": value_col})
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_choropleth_index(agg_gdf, out_path, title="Mean Resilience Index by District") -> None:
    """Choropleth of mean Resilience_Index by district (PNG)."""
    _setup_choropleth_axes(agg_gdf, "resilience_index_mean", title, out_path)


def plot_choropleth_vulnerability(agg_gdf, out_path,
                                  title="% Vulnerable Districts by Crop") -> None:
    """Choropleth of percentage vulnerable by district (PNG)."""
    _setup_choropleth_axes(agg_gdf, "pct_vulnerable", title, out_path)


# ---------------------------------------------------------------------------
# GeoJSON + Folium
# ---------------------------------------------------------------------------

def export_geojson(agg_gdf, out_path) -> Path:
    """Write an aggregated GeoDataFrame to GeoJSON and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in agg_gdf.columns if c != "geometry"]
    agg_gdf[cols + ["geometry"]].to_file(out_path, driver="GeoJSON")
    return out_path


def plot_interactive_folium(agg_gdf, out_path, value_col="resilience_index_mean",
                            title="Resilience Index by District") -> Path:
    """Build an interactive Folium choropleth (HTML) and return the path."""
    import folium

    centroid_lat, centroid_lon = 22.0, 79.0  # India centroid fallback
    try:
        center = agg_gdf.geometry.union_all().centroid
        centroid_lat, centroid_lon = center.y, center.x
    except Exception:
        pass

    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=5,
                   tiles="CartoDB positron")
    folium.Choropleth(
        geo_data=agg_gdf,
        data=agg_gdf,
        columns=["boundary_district", value_col],
        key_on="feature.properties.boundary_district",
        fill_color="RdYlGn",
        legend_name=title,
        highlight=True,
        line_opacity=0.3,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------

def build_maps(boundaries=DEFAULT_BOUNDARIES, resilience_csv=DEFAULT_RESILIENCE,
               out_dir=DEFAULT_OUT_DIR, crop: str | None = None, year: int | None = None,
               name_tag: str = "all"):
    """Run the full E10 map-generation pipeline, returning paths + coverage."""
    import pandas as pd
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resilience_df = pd.read_csv(resilience_csv)
    bounds = load_district_boundaries(boundaries)
    merged, cov = merge_resilience_with_geometry(resilience_df, bounds)
    report_coverage(cov)
    print(f"Matched rows in merged frame: {len(merged)}")

    agg = aggregate_resilience(merged, crop=crop, year=year)

    index_png = out_dir / f"choropleth_index_{name_tag}.png"
    vuln_png = out_dir / f"choropleth_vulnerability_{name_tag}.png"
    geojson_path = out_dir / f"resilience_district_{name_tag}.geojson"
    html_path = out_dir / f"resilience_map_{name_tag}.html"

    plot_choropleth_index(agg, index_png,
                          title=f"Mean Resilience Index by District ({name_tag})")
    plot_choropleth_vulnerability(agg, vuln_png,
                                  title=f"% Vulnerable by District ({name_tag})")
    export_geojson(agg, geojson_path)
    plot_interactive_folium(agg, html_path,
                            title=f"Resilience Index by District ({name_tag})")

    return {
        "coverage": cov,
        "index_png": index_png,
        "vuln_png": vuln_png,
        "geojson": geojson_path,
        "html": html_path,
        "agg": agg,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=str, default=str(DEFAULT_BOUNDARIES))
    parser.add_argument("--resilience-csv", type=str, default=str(DEFAULT_RESILIENCE))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--crop", type=str, default=None)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args(argv)

    if args.crop:
        name_tag = args.crop
    elif args.year is not None:
        name_tag = str(args.year)
    else:
        name_tag = "all"

    build_maps(boundaries=args.boundaries, resilience_csv=args.resilience_csv,
               out_dir=args.out_dir, crop=args.crop, year=args.year,
               name_tag=name_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
