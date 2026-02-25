"""
Yield Prediction Model — Random Forest Regressor
=================================================

Trains a Random Forest model to predict final crop yield (qt/ha) per lot
from aggregated IoT sensor features. Uses season-level leave-one-out
cross-validation to ensure the model generalises across growing seasons.

Author : David Bravo · https://bravoaidatastudio.com
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

SOIL_TYPE_MAP = {"sandy_loam": 0, "clay_loam": 1, "silty_clay": 2}
CROP_MAP      = {"SY-Mattis": 0, "DK-7220": 1, "P-1023": 2, "AW-Tornado": 3}


def build_season_features(readings: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-min sensor readings into season-level ML features per lot.

    Args
    ----
    readings : raw telemetry DataFrame from sensor_simulator
    yields   : yield labels DataFrame

    Returns
    -------
    feature_df : one row per lot × season with features + yield target
    """
    readings = readings.copy()
    readings["soil_moisture_pct"] = readings["soil_moisture_pct"].fillna(
        readings.groupby(["lot_id","season_year"])["soil_moisture_pct"].transform("median")
    )

    rows: List[Dict] = []

    for (lot_id, season_year), grp in readings.groupby(["lot_id", "season_year"]):
        yield_row = yields[
            (yields["lot_id"] == lot_id) & (yields["season_year"] == season_year)
        ]
        if yield_row.empty:
            continue

        repro = grp[grp["growth_stage"] == "reproductive"]
        grain = grp[grp["growth_stage"] == "grain_fill"]

        fc_map = {"clay_loam": 38.0, "sandy_loam": 28.0, "silty_clay": 42.0}
        fc = fc_map.get(yield_row.iloc[0]["soil_type"], 35.0)

        row = {
            # Season-mean sensor readings
            "mean_soil_moisture":   grp["soil_moisture_pct"].mean(),
            "min_soil_moisture":    grp["soil_moisture_pct"].min(),
            "mean_air_temp":        grp["air_temp_c"].mean(),
            "max_air_temp":         grp["air_temp_c"].max(),
            "mean_humidity":        grp["humidity_pct"].mean(),
            "total_rainfall":       grp["rainfall_mm"].sum(),
            "mean_radiation":       grp["solar_rad_wm2"].mean(),
            "mean_ndvi":            grp["ndvi_proxy"].mean(),
            "max_ndvi":             grp["ndvi_proxy"].max(),
            "mean_soil_ph":         grp["soil_ph"].mean(),
            "final_gdd":            grp["gdd_cumulative"].max(),

            # Reproductive-stage stress indicators
            "repro_drought_hours":  float((repro["soil_moisture_pct"] < fc * 0.75).sum()
                                         * (15/60)) if len(repro) else 0.0,
            "repro_heat_hours":     float((repro["air_temp_c"] > 32).sum()
                                         * (15/60)) if len(repro) else 0.0,
            "grain_mean_moisture":  grain["soil_moisture_pct"].mean() if len(grain) else np.nan,

            # Lot static attributes
            "area_ha":              yield_row.iloc[0]["area_ha"],
            "elevation_m":          yield_row.iloc[0]["elevation_m"],
            "soil_type_code":       SOIL_TYPE_MAP.get(yield_row.iloc[0]["soil_type"], -1),
            "crop_variety_code":    CROP_MAP.get(yield_row.iloc[0]["crop_variety"], -1),

            # Target
            "yield_qt_ha":          yield_row.iloc[0]["yield_qt_ha"],

            # Group key for CV
            "season_year":          int(season_year),
            "lot_id":               lot_id,
        }
        rows.append(row)

    return pd.DataFrame(rows).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "mean_soil_moisture", "min_soil_moisture",
    "mean_air_temp", "max_air_temp",
    "mean_humidity", "total_rainfall",
    "mean_radiation", "mean_ndvi", "max_ndvi",
    "mean_soil_ph", "final_gdd",
    "repro_drought_hours", "repro_heat_hours", "grain_mean_moisture",
    "area_ha", "elevation_m", "soil_type_code", "crop_variety_code",
]


def train_and_evaluate(feature_df: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    """Train RF model with Leave-One-Season-Out CV, return model and metrics."""
    X = feature_df[FEATURE_COLS].values
    y = feature_df["yield_qt_ha"].values
    groups = feature_df["season_year"].values

    logo  = LeaveOneGroupOut()
    maes, rmses, r2s = [], [], []

    print("  Leave-One-Season-Out cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups), 1):
        season_out = feature_df.iloc[val_idx]["season_year"].iloc[0]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ))
        ])
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[val_idx])

        mae  = mean_absolute_error(y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        r2   = r2_score(y[val_idx], preds)
        maes.append(mae); rmses.append(rmse); r2s.append(r2)
        print(f"    Fold {fold} (held-out season {season_out}) — MAE: {mae:.2f} qt/ha | RMSE: {rmse:.2f} | R²: {r2:.3f}")

    print(f"\n  CV Summary — MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f} qt/ha")
    print(f"              RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"              R²  : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    # Final model on all data
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(
            n_estimators=300, max_depth=8,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    final_pipe.fit(X, y)

    metrics = {
        "cv_mae":  float(np.mean(maes)),
        "cv_rmse": float(np.mean(rmses)),
        "cv_r2":   float(np.mean(r2s)),
    }

    return final_pipe, metrics


def save_model(pipe: Pipeline, feature_df: pd.DataFrame, metrics: Dict) -> None:
    out = MODEL_DIR / "yield_predictor.joblib"
    joblib.dump({"pipeline": pipe, "feature_cols": FEATURE_COLS, "metrics": metrics}, out)
    print(f"\n  ✔  Model saved → {out}")


if __name__ == "__main__":
    print("═" * 60)
    print("  YIELD PREDICTION MODEL TRAINING")
    print("  David Bravo · https://bravoaidatastudio.com")
    print("═" * 60 + "\n")

    readings = pd.read_parquet(DATA_DIR / "sensor_readings.parquet")
    yields   = pd.read_csv(DATA_DIR / "lot_yields.csv")

    print("  Building season-level features...")
    feature_df = build_season_features(readings, yields)
    print(f"  Feature matrix: {feature_df.shape[0]} rows × {len(FEATURE_COLS)} features")

    pipe, metrics = train_and_evaluate(feature_df)
    save_model(pipe, feature_df, metrics)

    print("\n  ✅  Training complete.\n")
