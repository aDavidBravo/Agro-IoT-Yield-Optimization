"""
Agro-IoT Sensor Telemetry Simulator
=====================================

Generates realistic IoT sensor readings for a multi-lot precision agriculture
operation across 3 growing seasons. The simulation incorporates:

  - Diurnal cycles for temperature, humidity and solar radiation
  - Soil moisture dynamics driven by rainfall events and evapotranspiration
  - Crop growth stages (sowing → emergence → vegetative → reproductive → harvest)
  - Agronomic stress events (drought, heat, waterlogging)
  - Sensor noise and occasional missing readings (~2%) simulating real telemetry
  - Per-lot variability in soil type, elevation and crop variety

The resulting dataset is the foundation for EDA and ML model training.

Author : David Bravo · https://bravoaidatastudio.com
Version: 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Output path ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LotConfig:
    """Per-lot agronomic configuration."""
    lot_id      : str
    area_ha     : float
    soil_type   : str           # sandy_loam | clay_loam | silty_clay
    elevation_m : float
    crop_variety: str
    sowing_doy  : int           # day-of-year sowing occurs each season
    base_yield_qt_ha: float     # expected yield under optimal conditions


LOTS: List[LotConfig] = [
    LotConfig("LOT-001", 45.0,  "clay_loam",  320, "SY-Mattis",   45, 68.0),
    LotConfig("LOT-002", 62.5,  "sandy_loam", 280, "DK-7220",     50, 61.0),
    LotConfig("LOT-003", 38.0,  "silty_clay", 410, "SY-Mattis",   42, 72.0),
    LotConfig("LOT-004", 55.0,  "clay_loam",  295, "P-1023",      48, 65.5),
    LotConfig("LOT-005", 29.0,  "sandy_loam", 850, "AW-Tornado",  55, 58.0),
    LotConfig("LOT-006", 71.0,  "clay_loam",  310, "DK-7220",     46, 70.0),
]

SOIL_FC = {"clay_loam": 38.0, "sandy_loam": 28.0, "silty_clay": 42.0}   # field capacity %vol
SOIL_WP = {"clay_loam": 18.0, "sandy_loam": 10.0, "silty_clay": 22.0}   # wilting point %vol

SEASONS = [
    {"year": 2022, "start": datetime(2022, 10, 1), "end": datetime(2023, 3, 31)},
    {"year": 2023, "start": datetime(2023, 10, 1), "end": datetime(2024, 3, 31)},
    {"year": 2024, "start": datetime(2024, 10, 1), "end": datetime(2025, 3, 31)},
]

READING_INTERVAL_MIN = 15


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def simulate_weather(dates: pd.DatetimeIndex, elevation_m: float) -> pd.DataFrame:
    """Generate hourly-equivalent weather variables for a location.

    Temperature, humidity and radiation follow diurnal patterns with
    realistic noise and occasional extreme events.
    """
    n      = len(dates)
    hour   = dates.hour + dates.minute / 60
    doy    = dates.day_of_year

    # Base temperature: seasonal + diurnal + elevation lapse rate
    elev_lapse   = (elevation_m - 300) * 0.006      # -0.6°C per 100m above 300m
    seasonal_amp = 6 * np.sin(2 * np.pi * (doy - 15) / 365)    # warmer in summer
    diurnal      = 8 * np.sin(np.pi * (hour - 6) / 12) * (hour > 6) * (hour < 20)
    air_temp     = 22 - elev_lapse + seasonal_amp + diurnal + np.random.normal(0, 1.2, n)
    air_temp     = np.clip(air_temp, -2, 42)

    # Humidity: inverse of temperature + noise
    humidity = 75 - 0.8 * (air_temp - 20) + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 98)

    # Solar radiation: bell curve during daylight
    rad_peak  = 800 + 150 * np.sin(2 * np.pi * (doy - 80) / 365)
    radiation = rad_peak * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1)
    radiation += np.random.normal(0, 30, n)
    radiation  = np.clip(radiation, 0, 1100)

    # Rainfall: Poisson events, heavier in LATAM wet season (Oct-Mar)
    wet_season   = ((doy >= 274) | (doy <= 90)).astype(float)  # Oct–Mar
    rain_prob    = 0.02 + 0.05 * wet_season
    rainfall     = np.where(np.random.random(n) < rain_prob,
                            np.random.exponential(2.5, n), 0.0)

    return pd.DataFrame({
        "air_temp_c":      air_temp,
        "humidity_pct":    humidity,
        "solar_rad_wm2":   radiation,
        "rainfall_mm":     rainfall,
    }, index=dates)


def simulate_soil(weather: pd.DataFrame, lot: LotConfig) -> pd.DataFrame:
    """Simulate soil moisture and temperature dynamics.

    Uses a simple water balance model:
        SM(t) = SM(t-1) + rainfall - ETc
    where ETc is estimated from temperature and radiation (simplified Hargreaves).
    """
    n          = len(weather)
    fc         = SOIL_FC[lot.soil_type]
    wp         = SOIL_WP[lot.soil_type]
    dt_hours   = READING_INTERVAL_MIN / 60

    soil_moisture = np.zeros(n)
    soil_moisture[0] = fc * 0.85                    # start near field capacity

    for i in range(1, n):
        rain  = weather.iloc[i]["rainfall_mm"]
        temp  = weather.iloc[i]["air_temp_c"]
        rad   = weather.iloc[i]["solar_rad_wm2"]

        # Simplified evapotranspiration (mm/interval)
        etc = max(0, (0.0023 * (temp + 17.8) * (rad / 24) * dt_hours))

        sm_new = soil_moisture[i-1] + rain * 0.7 - etc
        sm_new = np.clip(sm_new, wp * 0.8, fc * 1.05)  # physical bounds
        soil_moisture[i] = sm_new

    # Soil temperature: lags air temp by ~2 hours, dampened
    soil_temp = pd.Series(weather["air_temp_c"]).shift(8).fillna(method="bfill").values * 0.85
    soil_temp += np.random.normal(0, 0.5, n)

    # pH: slow drift around variety-specific optimum, with measurement noise
    ph_base  = 6.5 + lot.elevation_m / 2000        # slightly higher at elevation
    ph_drift = np.cumsum(np.random.normal(0, 0.002, n))
    ph_drift = ph_drift - ph_drift.mean()           # zero-mean drift
    ph       = np.clip(ph_base + ph_drift + np.random.normal(0, 0.08, n), 5.0, 8.5)

    return pd.DataFrame({
        "soil_moisture_pct": soil_moisture,
        "soil_temp_c":       soil_temp,
        "soil_ph":           ph,
    }, index=weather.index)


def compute_ndvi_proxy(soil: pd.DataFrame, weather: pd.DataFrame,
                       das: np.ndarray) -> np.ndarray:
    """Estimate canopy NDVI proxy from soil and growth stage.

    NDVI increases from emergence to peak vegetative (~DAS 60),
    then declines through grain fill and senescence.
    """
    # Logistic growth to peak, then linear decline
    peak_das = 65
    ndvi_max = 0.82 + np.random.uniform(-0.05, 0.05)
    ndvi_growth = ndvi_max / (1 + np.exp(-0.12 * (das - 35)))
    ndvi_decline = np.where(das > peak_das, ndvi_max - 0.008 * (das - peak_das), 0)
    ndvi = np.clip(ndvi_growth - ndvi_decline, 0.05, 0.95)

    # Penalize for moisture stress
    moisture_stress = np.clip((soil["soil_moisture_pct"] - 20) / 15, 0, 1)
    ndvi *= (0.7 + 0.3 * moisture_stress.values)

    return np.clip(ndvi + np.random.normal(0, 0.015, len(das)), 0.05, 0.95)


def compute_final_yield(lot: LotConfig, season_df: pd.DataFrame) -> float:
    """Compute final yield (qt/ha) based on season-long stress indicators."""
    # Drought stress: hours below 75% field capacity during reproductive stage
    fc = SOIL_FC[lot.soil_type]
    repro = season_df[season_df["growth_stage"] == "reproductive"]
    drought_hours = (repro["soil_moisture_pct"] < fc * 0.75).sum() * (READING_INTERVAL_MIN / 60)
    drought_penalty = min(0.25, drought_hours / 400 * 0.20)

    # Heat stress: hours above 32°C during grain fill
    heat_hours = (repro["air_temp_c"] > 32).sum() * (READING_INTERVAL_MIN / 60)
    heat_penalty = min(0.15, heat_hours / 200 * 0.12)

    # pH penalty
    mean_ph = season_df["soil_ph"].mean()
    ph_penalty = 0.0 if 6.0 <= mean_ph <= 7.2 else min(0.22, abs(mean_ph - 6.6) * 0.15)

    # Elevation bonus (cooler nights → better grain fill) — capped
    elev_bonus = min(0.05, (lot.elevation_m - 300) / 10000)

    final_yield = lot.base_yield_qt_ha * (1 - drought_penalty - heat_penalty - ph_penalty + elev_bonus)
    final_yield += np.random.normal(0, 1.5)          # residual unexplained variance
    return round(max(10.0, final_yield), 2)


GROWTH_STAGES = [
    (0,   7,   "pre-sowing"),
    (1,   15,  "emergence"),
    (16,  45,  "vegetative"),
    (46,  85,  "reproductive"),
    (86,  110, "grain_fill"),
    (111, 999, "harvest_ready"),
]


def assign_growth_stage(das: int) -> str:
    for lo, hi, stage in GROWTH_STAGES:
        if lo <= das <= hi:
            return stage
    return "harvest_ready"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_seasons() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sensor telemetry and yield labels for all lots × seasons.

    Returns
    -------
    readings_df : pd.DataFrame
        15-minute sensor telemetry (all lots, all seasons).
    yields_df : pd.DataFrame
        One row per lot × season with final yield label.
    """
    all_readings: list[pd.DataFrame] = []
    all_yields:   list[dict]         = []

    total = len(LOTS) * len(SEASONS)
    done  = 0

    for lot in LOTS:
        for season in SEASONS:
            done += 1
            print(f"  Simulating {lot.lot_id} · Season {season['year']} ({done}/{total})...")

            dates = pd.date_range(
                start=season["start"],
                end=season["end"],
                freq=f"{READING_INTERVAL_MIN}min",
            )

            # Compute DAS (days after sowing) for each timestamp
            sowing_date = datetime(season["year"], 1, 1) + timedelta(days=lot.sowing_doy - 1)
            sowing_date = max(sowing_date, season["start"])
            das_arr = np.array([(d - sowing_date).days for d in dates], dtype=float)
            das_arr = np.clip(das_arr, -30, 150)

            weather = simulate_weather(dates, lot.elevation_m)
            soil    = simulate_soil(weather, lot)
            ndvi    = compute_ndvi_proxy(soil, weather, das_arr)

            # Assemble reading frame
            df = pd.concat([weather, soil], axis=1)
            df["ndvi_proxy"] = ndvi
            df["das"]        = das_arr.astype(int)
            df["growth_stage"] = [assign_growth_stage(int(d)) for d in das_arr]

            # Cumulative GDD (base 8°C)
            gdd_daily  = df.groupby(df.index.date)["air_temp_c"].mean().apply(lambda t: max(0, t - 8))
            gdd_cumsum = gdd_daily.cumsum()
            df["gdd_cumulative"] = df.index.date
            df["gdd_cumulative"] = df["gdd_cumulative"].map(
                {d: float(gdd_cumsum.get(d, 0)) for d in df.index.date}
            )

            # Lot metadata
            df["lot_id"]       = lot.lot_id
            df["area_ha"]      = lot.area_ha
            df["soil_type"]    = lot.soil_type
            df["elevation_m"]  = lot.elevation_m
            df["crop_variety"] = lot.crop_variety
            df["season_year"]  = season["year"]
            df["timestamp"]    = df.index
            df = df.reset_index(drop=True)

            # Inject ~2% missing readings (realistic sensor dropouts)
            for col in ["soil_moisture_pct", "ndvi_proxy", "humidity_pct"]:
                mask = np.random.random(len(df)) < 0.02
                df.loc[mask, col] = np.nan

            all_readings.append(df)

            # Compute yield label for this lot-season
            final_yield = compute_final_yield(lot, df)
            all_yields.append({
                "lot_id":       lot.lot_id,
                "season_year":  season["year"],
                "area_ha":      lot.area_ha,
                "soil_type":    lot.soil_type,
                "elevation_m":  lot.elevation_m,
                "crop_variety": lot.crop_variety,
                "yield_qt_ha":  final_yield,
            })

    readings_df = pd.concat(all_readings, ignore_index=True)
    yields_df   = pd.DataFrame(all_yields)
    return readings_df, yields_df


def save_data(readings_df: pd.DataFrame, yields_df: pd.DataFrame) -> None:
    readings_path = DATA_DIR / "sensor_readings.parquet"
    yields_path   = DATA_DIR / "lot_yields.csv"

    readings_df.to_parquet(readings_path, index=False, engine="pyarrow")
    yields_df.to_csv(yields_path, index=False)

    print(f"\n  ✔  Sensor readings → {readings_path}  ({len(readings_df):,} rows)")
    print(f"  ✔  Yield labels    → {yields_path}  ({len(yields_df)} rows)")
    print(f"  ✔  File size       : {readings_path.stat().st_size / 1024**2:.1f} MB (Parquet)")


def print_summary(readings_df: pd.DataFrame, yields_df: pd.DataFrame) -> None:
    print("\n═" * 60)
    print("  SIMULATION SUMMARY")
    print("═" * 60)
    print(f"  Total readings   : {len(readings_df):,}")
    print(f"  Lots simulated   : {readings_df['lot_id'].nunique()}")
    print(f"  Seasons          : {readings_df['season_year'].nunique()}")
    print(f"  Missing rate     : {readings_df.isnull().mean().mean()*100:.2f}% (simulated dropouts)")
    print(f"\n  Yield range      : {yields_df['yield_qt_ha'].min():.1f} – {yields_df['yield_qt_ha'].max():.1f} qt/ha")
    print(f"  Mean yield       : {yields_df['yield_qt_ha'].mean():.1f} qt/ha")
    print()
    print(yields_df.to_string(index=False))


if __name__ == "__main__":
    print("═" * 60)
    print("  AGRO-IOT SENSOR TELEMETRY SIMULATOR")
    print("  David Bravo · https://bravoaidatastudio.com")
    print("═" * 60 + "\n")

    readings_df, yields_df = generate_all_seasons()
    save_data(readings_df, yields_df)
    print_summary(readings_df, yields_df)

    print("\n  ✅  Done. Run pipeline_showcase.py next.\n")
