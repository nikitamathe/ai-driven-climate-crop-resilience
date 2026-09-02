"""Climate-stress indicators derived from the NASA POWER monthly record.

.. important::

   The NASA POWER data covers a **single spatial point** — all indicators
   below characterise *one location* and must not be interpreted as
   district-level climate observations.

   The anomaly baseline is the full 1996–2020 record (25 years), shorter
   than the standard WMO 30-year climatological reference period.

   Thermal-stress thresholds are **project heuristics** for visualisation
   and analysis only; they are not validated agronomic standards.

Indicators
----------
* **SPI** — Standardized Precipitation Index (McKee et al. 1993) at scales
  3, 6 and 12 months, computed from monthly precipitation via gamma-fitting
  and normal-score transformation.
* **Anomalies** — z-score departure of yearly rainfall and mean temperature
  from the 1996–2020 climatological mean.
* **Mann-Kendall trend** — non-parametric trend test returning S, variance,
  Z, p-value and direction for each yearly climate variable.
* **Sen's slope** — Theil-Sen robust trend estimator (median of pairwise
  slopes) for each yearly climate variable.
* **Thermal stress** — monthly count of months where mean temperature deviates
  beyond a configurable threshold (default ±5 °C) from each crop's thermal
  optimum (project heuristic).

API
---
``compute_all_indicators(nasa_monthly, crop_constants=None, ...)``
    Top-level orchestrator that returns a DataFrame with one row per year
    and columns for every indicator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONTH_ABBR_TO_NUM = {
    m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}

DEFAULT_CROP_OPTIMA: dict[str, float] = {
    "rice": 25.0,
    "maize": 22.0,
    "chickpea": 20.0,
    "cotton": 28.0,
}


def _month_abbr_to_num(abbr: str) -> int:
    """Convert a three-letter month abbreviation to its calendar number."""
    return MONTH_ABBR_TO_NUM[abbr.upper()]


# ---------------------------------------------------------------------------
# SPI (Standardized Precipitation Index)
# ---------------------------------------------------------------------------

def _gamma_cdf_fit(precip: np.ndarray) -> np.ndarray:
    """CDF of precipitation via L-moment gamma fit (McKee et al. 1993).

    Parameters
    ----------
    precip : 1-D array of non-negative precipitation totals.

    Returns
    -------
    1-D array of CDF values in (0, 1).

    Notes
    -----
    * Zero-precipitation months are handled by accumulating a separate
      probability mass *q0* and blending it with the gamma CDF of the
      positive values.
    * CDF values are clipped to [0.001, 0.999] to avoid numerical issues
      in the normal-score transform.
    """
    precip = np.asarray(precip, dtype=float)
    n = len(precip)
    n_zero = int(np.sum(precip == 0))
    q0 = (n_zero + 0.1) / (n + 0.1) if n > 0 else 0.0

    positive = precip[precip > 0]
    if len(positive) < 2:
        return np.clip(np.where(precip == 0, q0 * 0.5, q0 + 0.5), 0.001, 0.999)

    mean_p = float(np.mean(positive))
    if mean_p <= 0:
        return np.clip(np.where(precip == 0, q0 * 0.5, q0 + 0.5), 0.001, 0.999)

    # L-moment estimators for the gamma distribution
    log_var = np.log(mean_p) - np.mean(np.log(positive))
    s = np.sqrt(max(log_var, 0.0))
    if s < 1e-6:
        # Constant precipitation: degenerate distribution, return CDF near 0.5
        return np.clip(np.where(precip == 0, q0 * 0.5, q0 + 0.5), 0.001, 0.999)
    alpha = (1.0 + np.sqrt(1.0 + 4.0 * s / 3.0)) / (4.0 * s)
    beta = mean_p / alpha

    cdf = np.zeros(n, dtype=float)
    pos_mask = precip > 0
    cdf[pos_mask] = q0 + (1.0 - q0) * sp_stats.gamma.cdf(precip[pos_mask], alpha, scale=beta)
    cdf[precip == 0] = q0 * np.linspace(0.5 / max(n_zero, 1), 0.5, n_zero) if n_zero else 0.0
    return np.clip(cdf, 0.001, 0.999)


def spi(precip_monthly: pd.Series, scale: int) -> pd.Series:
    """Compute the Standardized Precipitation Index.

    Parameters
    ----------
    precip_monthly : pd.Series
        Monthly precipitation values, ideally with a ``DatetimeIndex`` or
        any ``PeriodIndex`` that ``rolling()`` can interpret.
    scale : int
        Accumulation window in months (e.g. 3 for SPI-3).

    Returns
    -------
    pd.Series
        SPI values clipped to [-3, 3].  The first ``scale - 1`` entries are
        ``NaN`` (warm-up period).
    """
    roll = precip_monthly.rolling(scale, min_periods=scale).sum()
    spi_vals = pd.Series(np.nan, index=roll.index, dtype=float)
    valid = roll.dropna()
    if len(valid) < 2:
        return spi_vals
    cdf = _gamma_cdf_fit(valid.values)
    cdf = np.clip(cdf, 0.001, 0.999)
    spi_vals.loc[valid.index] = np.clip(sp_stats.norm.ppf(cdf), -3.0, 3.0)
    return spi_vals


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

def compute_anomalies(
    nasa_monthly: pd.DataFrame,
    baseline_range: tuple[int, int] = (1996, 2020),
) -> pd.DataFrame:
    """Yearly z-score anomalies for rainfall and mean temperature.

    The baseline mean and standard deviation are computed over
    ``baseline_range``.  A z-score of 0 indicates the yearly value equals
    the climatological mean.

    .. note::

       The default 1996–2020 baseline is 25 years, shorter than the
       WMO-standard 30-year reference period.

    Parameters
    ----------
    nasa_monthly : DataFrame with columns ``Year``, ``Month``, ``Rainfall``,
        ``AvgTemp``.
    baseline_range : (start_year, end_year) inclusive.

    Returns
    -------
    DataFrame with columns ``Year``, ``Rainfall_Anomaly``, ``Temp_Anomaly``.
    """
    yearly = (
        nasa_monthly.groupby("Year")
        .agg(Rainfall=("Rainfall", "sum"), AvgTemp=("AvgTemp", "mean"))
        .reset_index()
    )
    bl = yearly[(yearly["Year"] >= baseline_range[0]) & (yearly["Year"] <= baseline_range[1])]
    rain_std = bl["Rainfall"].std(ddof=0)
    temp_std = bl["AvgTemp"].std(ddof=0)
    rain_anom = ((yearly["Rainfall"] - bl["Rainfall"].mean()) / rain_std).values
    temp_anom = ((yearly["AvgTemp"] - bl["AvgTemp"].mean()) / temp_std).values
    return pd.DataFrame({
        "Year": yearly["Year"].values,
        "Rainfall_Anomaly": rain_anom,
        "Temp_Anomaly": temp_anom,
    })


# ---------------------------------------------------------------------------
# Mann-Kendall trend test
# ---------------------------------------------------------------------------

def mann_kendall(
    series: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Non-parametric Mann-Kendall trend test.

    Parameters
    ----------
    series : 1-D array-like of numeric values.
    alpha : Significance level for the two-sided test (default 0.05).

    Returns
    -------
    dict with keys:
        ``S``       — rank-based statistic
        ``var_s``   — variance of S (with tie correction)
        ``Z``       — standardized test statistic
        ``p``       — two-sided p-value (NaN if variance is zero)
        ``trend``   — ``'increasing'``, ``'decreasing'`` or ``'no trend'``
        ``alpha``   — significance level used
    """
    series = np.asarray(series, dtype=float)
    valid = series[~np.isnan(series)]
    n = len(valid)
    if n < 2:
        return {"S": 0.0, "var_s": 0.0, "Z": 0.0, "p": float("nan"),
                "trend": "no trend", "alpha": alpha}

    # Count concordant/discordant pairs
    s = 0.0
    for k in range(n - 1):
        s += np.sum(np.sign(valid[k + 1:] - valid[k]))

    # Count ties for variance correction
    _, counts = np.unique(valid, return_counts=True)
    tp = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18.0

    if var_s == 0:
        return {"S": s, "var_s": 0.0, "Z": 0.0, "p": float("nan"),
                "trend": "no trend", "alpha": alpha}

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * sp_stats.norm.sf(abs(z))

    if s > 0 and p < alpha:
        trend = "increasing"
    elif s < 0 and p < alpha:
        trend = "decreasing"
    else:
        trend = "no trend"

    return {"S": s, "var_s": var_s, "Z": z, "p": p, "trend": trend, "alpha": alpha}


# ---------------------------------------------------------------------------
# Sen's slope (Theil-Sen estimator)
# ---------------------------------------------------------------------------

def sen_slope(series: np.ndarray, time_axis: np.ndarray | None = None) -> float:
    """Theil-Sen robust slope estimator (median of pairwise slopes).

    Parameters
    ----------
    series : 1-D array of values.
    time_axis : Optional 1-D array of time positions (default ``0, 1, ..., n-1``).

    Returns
    -------
    Median slope value.  Returns ``NaN`` if fewer than two valid points.
    """
    series = np.asarray(series, dtype=float)
    valid = series[~np.isnan(series)]
    if len(valid) < 2:
        return float("nan")
    n = len(valid)
    if time_axis is None:
        time_axis = np.arange(n)
    else:
        time_axis = np.asarray(time_axis, dtype=float)[~np.isnan(series)]
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = time_axis[j] - time_axis[i]
            if dt != 0:
                slopes.append((valid[j] - valid[i]) / dt)
    return float(np.median(slopes)) if slopes else float("nan")


# ---------------------------------------------------------------------------
# Thermal-stress heuristic (project heuristic, NOT agronomic)
# ---------------------------------------------------------------------------

def thermal_stress(
    monthly_temp: np.ndarray,
    crop_optimum: float,
    threshold: float = 5.0,
) -> int:
    """Count months where |temp − optimum| > threshold (project heuristic).

    .. warning::

       This is a project heuristic for exploratory analysis and
       visualisation.  It is **not** a validated agronomic stress metric.

    Parameters
    ----------
    monthly_temp : 1-D array of monthly mean temperatures (°C).
    crop_optimum : Optimal growing temperature for the crop (°C).
    threshold : Allowable deviation in °C (default 5).

    Returns
    -------
    Number of months exceeding the threshold.
    """
    t = np.asarray(monthly_temp, dtype=float)
    t = t[~np.isnan(t)]
    if len(t) == 0:
        return 0
    return int(np.sum(np.abs(t - crop_optimum) > threshold))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _yearly_mann_kendall_slopes(
    nasa_monthly: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compute MK trend + Sen's slope for yearly aggregates of *columns*."""
    yearly = (
        nasa_monthly.groupby("Year")
        .agg({c: "sum" if c == "Rainfall" else "mean" for c in columns})
        .sort_index()
    )
    rows = []
    for col in columns:
        vals = yearly[col].values.astype(float)
        mk = mann_kendall(vals)
        sl = sen_slope(vals, yearly.index.values.astype(float))
        rows.append({
            "variable": col,
            "sen_slope": sl,
            "mk_Z": mk["Z"],
            "mk_p": mk["p"],
            "mk_direction": mk["trend"],
        })
    return pd.DataFrame(rows)


def compute_all_indicators(
    nasa_monthly: pd.DataFrame,
    crop_constants: dict[str, float] | None = None,
    spi_scales: tuple[int, ...] = (3, 6, 12),
    baseline_range: tuple[int, int] = (1996, 2020),
    thermal_threshold: float = 5.0,
) -> pd.DataFrame:
    """Compute every E06 climate indicator and return a single DataFrame.

    Parameters
    ----------
    nasa_monthly : Raw monthly NASA POWER DataFrame.
    crop_constants : ``{crop_name: optimum_temp_C}``.  Defaults to the
        values established by the audited dataset.
    spi_scales : SPI accumulation windows.
    baseline_range : Years used to compute anomaly baselines.
    thermal_threshold : Deviation in °C for the thermal-stress heuristic.

    Returns
    -------
    DataFrame with one row per year and columns for every indicator.

    Notes
    -----
    * All indicators are computed from a **single spatial point** (the NASA
      POWER record).  They are not district-specific observations.
    * The default anomaly baseline (1996–2020, 25 years) is shorter than the
      WMO-standard 30-year climatological reference period.
    * Thermal-stress thresholds are **project heuristics**, not agronomic
      standards.
    """
    if crop_constants is None:
        crop_constants = DEFAULT_CROP_OPTIMA

    # Ensure numeric month column
    df = nasa_monthly.copy()
    if df["Month"].dtype == object:
        df["Month_num"] = df["Month"].map(_month_abbr_to_num)
    else:
        df["Month_num"] = df["Month"].astype(int)
    df = df.sort_values(["Year", "Month_num"]).reset_index(drop=True)

    # Create a monthly PeriodIndex for rolling SPI
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month_num"].astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("date").sort_index()

    # --- SPI ---
    spi_series: dict[str, pd.Series] = {}
    for s in spi_scales:
        spi_series[f"SPI_{s}"] = spi(df["Rainfall"], s)

    # Yearly SPI: last valid value per calendar year
    spi_yearly: dict[str, list] = {"Year": sorted(df["Year"].unique())}
    for key, ser in spi_series.items():
        spi_yearly[key] = [
            ser.loc[ser.index.year == y].dropna().iloc[-1]
            if not ser.loc[ser.index.year == y].dropna().empty
            else np.nan
            for y in spi_yearly["Year"]
        ]

    # --- Anomalies ---
    anomalies = compute_anomalies(nasa_monthly, baseline_range)

    # --- Mann-Kendall + Sen slope ---
    mk_df = _yearly_mann_kendall_slopes(
        nasa_monthly, ["Rainfall", "AvgTemp", "MaxTemp", "MinTemp"]
    )

    # --- Thermal stress (per crop, using monthly AvgTemp) ---
    monthly_temp = df["AvgTemp"]
    stress: dict[str, list] = {"Year": spi_yearly["Year"]}
    for crop, opt in crop_constants.items():
        col_name = f"Thermal_Stress_{crop}"
        stress[col_name] = [
            int(thermal_stress(
                monthly_temp.loc[monthly_temp.index.year == y].values,
                opt, thermal_threshold,
            ))
            for y in stress["Year"]
        ]
    stress["Thermal_Stress_Mean"] = [
        np.mean([stress[f"Thermal_Stress_{c}"][i] for c in crop_constants])
        for i in range(len(stress["Year"]))
    ]

    # --- Assemble output ---
    out = pd.DataFrame(spi_yearly)
    out = out.merge(anomalies, on="Year", how="left")

    for _, row in mk_df.iterrows():
        v = row["variable"]
        out[f"{v}_SenSlope"] = row["sen_slope"]
        out[f"{v}_MK_Z"] = row["mk_Z"]
        out[f"{v}_MK_p"] = row["mk_p"]
        out[f"{v}_MK_direction"] = row["mk_direction"]

    stress_df = pd.DataFrame(stress)
    out = out.merge(stress_df, on="Year", how="left")

    return out


def write_climate_indicators(
    nasa_monthly: pd.DataFrame,
    out_path: Path | str,
    **kwargs,
) -> pd.DataFrame:
    """Compute indicators and write ``climate_indicators.csv``.

    Parameters
    ----------
    nasa_monthly : Raw monthly NASA POWER DataFrame.
    out_path : Destination CSV path.
    **kwargs : Forwarded to :func:`compute_all_indicators`.

    Returns
    -------
    The indicators DataFrame that was written.
    """
    indicators = compute_all_indicators(nasa_monthly, **kwargs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(out_path, index=False)
    return indicators
