"""
Sensor Data Processor
======================

Handles cleaning, normalization and feature engineering of raw IoT
telemetry before it enters the ML pipeline. Designed to run as part
of an Apache Airflow DAG (see airflow_dags/sensor_ingestion_dag.py).

Author : David Bravo · https://bravoaidatastudio.com
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_BOUNDS = {
    "soil_moisture_pct": (0.0,  100.0),
    "air_temp_c":         (-5.0, 50.0),
    "soil_temp_c":        (-2.0, 45.0),
    "humidity_pct":       (0.0,  100.0),
    "rainfall_mm":        (0.0,  200.0),
    "solar_rad_wm2":      (0.0,  1200.0),
    "ndvi_proxy":         (0.0,  1.0),
    "soil_ph":            (3.5,  9.5),
}


def validate_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """Clip sensor readings to physically plausible bounds."""
    df = df.copy()
    for col, (lo, hi) in SENSOR_BOUNDS.items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                print(f"    [{col}] clipped {out_of_range} out-of-range readings")
            df[col] = df[col].clip(lo, hi)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill within each lot×season group."""
    df = df.copy()
    group_cols = ["lot_id", "season_year"]
    numeric_cols = [c for c in SENSOR_BOUNDS if c in df.columns]
    df[numeric_cols] = (
        df.groupby(group_cols)[numeric_cols]
        .transform(lambda x: x.fillna(method="ffill").fillna(method="bfill"))
    )
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [4, 24, 96]) -> pd.DataFrame:
    """Add rolling mean features for key sensors within each lot×season.

    Windows are in number of 15-min intervals:
      4  = 1 hour
      24 = 6 hours
      96 = 24 hours
    """
    df = df.copy()
    target_cols = ["soil_moisture_pct", "air_temp_c", "solar_rad_wm2"]
    for col in target_cols:
        if col not in df.columns:
            continue
        for w in windows:
            hours = w * 15 // 60
            df[f"{col}_roll{hours}h"] = (
                df.groupby(["lot_id", "season_year"])[col]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
    return df


def add_stress_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean stress indicator columns for agronomic alerts."""
    df = df.copy()
    # Drought stress: moisture below 70% of field capacity proxy
    fc_proxy = {"sandy_loam": 28.0, "clay_loam": 38.0, "silty_clay": 42.0}
    df["drought_stress"] = 0
    for soil_type, fc in fc_proxy.items():
        mask = (df.get("soil_type") == soil_type) & (df["soil_moisture_pct"] < fc * 0.70)
        df.loc[mask, "drought_stress"] = 1

    # Heat stress: air temperature above 32°C
    df["heat_stress"] = (df["air_temp_c"] > 32).astype(int)

    # pH stress
    df["ph_stress"] = ((df["soil_ph"] < 6.0) | (df["soil_ph"] > 7.2)).astype(int)

    return df


def run_processing_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full processing pipeline: validate → impute → rolling features → stress flags."""
    print("  Running sensor processing pipeline...")
    df = validate_and_clip(raw_df)
    df = impute_missing(df)
    df = add_rolling_features(df)
    df = add_stress_flags(df)
    print(f"  ✔  Processed {len(df):,} readings | {df.shape[1]} columns")
    return df
