# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Engineering

Complete the `solve()` function. Read problem.md for the full specification.
The raw strings are deliberately messy — read the data before you parse it.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Engineer the 10-column feature table from raw HDB resale data.

    See problem.md for the exact columns, the storey_range OCR fix, the
    dual-format remaining_lease parser + null imputation rule, the room
    ordinal map, and the derived features.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")
    lo = pl.col("storey_range").str.split(" TO ").list.get(0).str.replace_all("O", "0")
    hi = pl.col("storey_range").str.split(" TO ").list.get(1).str.replace_all("O", "0")
    years = pl.col("remaining_lease").str.extract(r"^(\d+)").cast(pl.Float64)
    months = (
        pl.col("remaining_lease")
        .str.extract(r"years (\d+) months$")
        .cast(pl.Float64)
        .fill_null(0)
    )

    return (
        df.with_columns(pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year"))
        .with_columns(
            ((lo.cast(pl.Float64) + hi.cast(pl.Float64)) / 2).alias("storey_midpoint"),
            (pl.col("sale_year") - pl.col("lease_commence_date")).alias("flat_age_years"),
            (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
            pl.col("flat_type")
            .replace(
                {
                    "2 ROOM": 2,
                    "3 ROOM": 3,
                    "4 ROOM": 4,
                    "5 ROOM": 5,
                    "EXECUTIVE": 6,
                    "MULTI-GENERATION": 7,
                }
            )
            .cast(pl.Int64)
            .alias("flat_type_rooms"),
        )
        .with_columns(
            (years + months / 12)
            .fill_null(99 - pl.col("flat_age_years"))
            .alias("remaining_lease_years")
        )
        .select(
            "town",
            "flat_type",
            "flat_type_rooms",
            "sale_year",
            "storey_midpoint",
            "floor_area_sqm",
            "flat_age_years",
            "remaining_lease_years",
            "resale_price",
            "price_per_sqm",
        )
        .sort(["sale_year", "town"])
    )


if __name__ == "__main__":
    print(solve().head())
