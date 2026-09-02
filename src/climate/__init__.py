"""Climate-stress indicator computation from the NASA POWER record.

This module computes SPI (Standardized Precipitation Index), rainfall and
temperature anomalies, Mann-Kendall trend statistics, Sen's slope, and crop
thermal-stress heuristics from the single-point NASA POWER monthly record.

.. important::

   The NASA POWER data is a **single spatial point**, not a grid.  All
   indicators computed here represent *one location's* climate history and
   are **not** district-specific observations.  The anomaly baseline spans
   only 1996–2020 (25 years), which is shorter than the WMO-standard 30-year
   climatological reference period.  Thermal-stress thresholds are project
   heuristics, not agronomic standards.
"""
