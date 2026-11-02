"""Automated Gwalior air-quality update and forecasting pipeline.

Run with: python auto_update.py
Set OPENAQ_API_KEY as a Windows environment variable before running.
"""
from pathlib import Path
import os
import sys
import traceback
import requests
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "gwalior_air_quality_output"
OUTPUT_DIR.mkdir(exist_ok=True)

GB_MODEL_PATH = BASE_DIR / "03_data_cleaning" / "gradient_boosting_model.pkl"
LR_MODEL_PATH = BASE_DIR / "03_data_cleaning" / "linear_regression_model.pkl"
# This should be the wide historical CSV containing timestamp + pollutant columns.
HISTORY_CANDIDATES = [
    Path(r"C:\Users\LOQ\Gwalior AQI Project\01_Data_Collection\gwalior_air_quality_historical.csv"),
    BASE_DIR / "processed_data" / "gwalior_air_quality_historical.csv",
    BASE_DIR / "gwalior_air_quality_historical.csv",
    BASE_DIR / "gwalior_air_quality_historical(1).xls",
]
HISTORY_PATH = OUTPUT_DIR / "gwalior_air_quality_history.csv"
LATEST_PATH = OUTPUT_DIR / "gwalior_latest_readings.csv"
FORECAST_PATH = OUTPUT_DIR / "gwalior_pollution_forecast.csv"
LOG_PATH = OUTPUT_DIR / "auto_update_log.txt"

LOCATION_ID = 10907
API_URL = f"https://api.openaq.org/v3/locations/{LOCATION_ID}/latest"
SENSOR_TO_POLLUTANT = {
    12236153: "pm25",
    12236152: "pm10",
    12236150: "no2",
    12236148: "co",
    12236155: "so2",
    12236151: "o3",
    12236156: "temperature",
    12236154: "humidity",
}
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3", "temperature", "humidity"]
FEATURES = [
    "pm25", "pm10", "no2", "co", "so2", "o3", "temperature", "humidity",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm25_lag_1", "pm25_lag_4", "pm25_lag_96",
    "pm10_lag_1", "pm10_lag_4", "pm10_lag_96",
    "no2_lag_1", "no2_lag_4", "no2_lag_96",
]
TARGETS = [
    "pm25_next_15min", "pm25_next_1hour",
    "pm10_next_15min", "pm10_next_1hour",
    "no2_next_15min", "no2_next_1hour",
]


def log(message):
    line = f"{pd.Timestamp.now(tz='Asia/Kolkata').isoformat()} | {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_history_path():
    if HISTORY_PATH.exists():
        return HISTORY_PATH
    for candidate in HISTORY_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(p) for p in [HISTORY_PATH, *HISTORY_CANDIDATES])
    raise FileNotFoundError(
        "Could not find historical wide CSV. Put the file at one of these paths:\n" + searched
    )


def load_history(path):
    # The older file may have a .xls extension while actually containing CSV text.
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)
    if "timestamp" not in df.columns:
        # If timestamp is stored as the first unnamed/index column, try that.
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce", utc=True)
        if parsed.notna().sum() > 0:
            df = df.rename(columns={first: "timestamp"})
        else:
            raise ValueError(f"No timestamp column found in historical file: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()
    for col in POLLUTANTS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp", *POLLUTANTS]].groupby("timestamp", as_index=False).mean(numeric_only=True)
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_latest():
    api_key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY environment variable is not set. Set it once in Windows, then rerun."
        )
    response = requests.get(
        API_URL,
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    rows = []
    for item in results:
        sensor_id = item.get("sensorsId") or item.get("sensors_id")
        # OpenAQ v3 often returns sensor metadata nested in a sensor object.
        if sensor_id is None and isinstance(item.get("sensors"), dict):
            sensor_id = item["sensors"].get("id")
        pollutant = SENSOR_TO_POLLUTANT.get(sensor_id)
        if not pollutant:
            # Some responses expose parameter info alongside a sensor ID.
            sensor = item.get("sensors") if isinstance(item.get("sensors"), dict) else {}
            parameter = (sensor.get("parameter") or {}).get("name") if isinstance(sensor.get("parameter"), dict) else None
            if parameter:
                parameter = str(parameter).lower().replace("2.5", "25").replace("2.5", "25")
                pollutant = {"pm25": "pm25", "pm10": "pm10", "no2": "no2", "co": "co", "so2": "so2", "o3": "o3", "temperature": "temperature", "relativehumidity": "humidity", "humidity": "humidity"}.get(parameter.replace("_", "").replace("-", ""))
        if not pollutant:
            continue
        value = item.get("value")
        dt = item.get("datetime") or item.get("period", {}).get("datetimeTo") or item.get("period", {}).get("datetimeFrom")
        if isinstance(dt, dict):
            dt = dt.get("utc") or dt.get("local")
        if value is None or dt is None:
            continue
        rows.append({"pollutant": pollutant, "value": float(value), "timestamp": pd.to_datetime(dt, utc=True, errors="coerce")})
    fresh = pd.DataFrame(rows).dropna(subset=["timestamp"])
    if fresh.empty:
        raise ValueError("OpenAQ returned no recognized pollutant readings. Check API response format/sensor IDs.")
    # Keep the newest available measurement for each pollutant, even if sensors report at different times.
    fresh = fresh.sort_values("timestamp").drop_duplicates("pollutant", keep="last")
    latest_wide = fresh.pivot_table(index="timestamp", columns="pollutant", values="value", aggfunc="last")
    latest_wide = latest_wide.reindex(columns=POLLUTANTS)
    # Use the newest actual sensor timestamp as this batch's timestamp; values retain their actual pollutant readings.
    batch_timestamp = fresh["timestamp"].max()
    batch = latest_wide.reindex([batch_timestamp])
    # Fill pollutant columns from each pollutant's latest value, irrespective of its own timestamp.
    for _, row in fresh.iterrows():
        batch.loc[batch_timestamp, row["pollutant"]] = row["value"]
    batch.index.name = "timestamp"
    return fresh, batch


def main():
    log("Starting update")
    for path in [GB_MODEL_PATH, LR_MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

    history_path = get_history_path()
    history = load_history(history_path)
    log(f"Loaded historical data from {history_path} ({history.shape[0]} timestamps)")

    fresh_long, fresh_batch = fetch_latest()
    # Save raw latest sensor values with each pollutant's actual observation time.
    fresh_long["timestamp"] = fresh_long["timestamp"].dt.tz_convert("Asia/Kolkata")
    fresh_long.to_csv(LATEST_PATH, index=False)
    log(f"Fetched {len(fresh_long)} latest pollutant readings from OpenAQ")

    # Merge readings into history. New timestamp rows may be sparse; regularize below.
    combined = pd.concat([history, fresh_batch], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.resample("15min").mean()
    combined = combined.interpolate(method="time", limit_direction="both")
    combined = combined.ffill().bfill()
    if combined[POLLUTANTS].isna().any().any():
        missing = combined[POLLUTANTS].isna().sum()
        raise ValueError(f"Still missing data after interpolation; cannot forecast safely:\n{missing[missing > 0]}")

    # Preserve updated history in a stable wide format.
    combined.index.name = "timestamp"
    history_export = combined.reset_index()
    history_export["timestamp"] = history_export["timestamp"].dt.tz_convert("Asia/Kolkata")
    history_export.to_csv(HISTORY_PATH, index=False)
    log(f"Saved updated history ({len(combined)} rows) to {HISTORY_PATH}")

    # Create model features on the regular 15-minute grid.
    feature_frame = combined.copy()
    local_index = feature_frame.index.tz_convert("Asia/Kolkata")
    feature_frame["hour"] = local_index.hour
    feature_frame["day"] = local_index.day
    feature_frame["month"] = local_index.month
    feature_frame["day_of_week"] = local_index.dayofweek
    feature_frame["is_weekend"] = (feature_frame["day_of_week"] >= 5).astype(int)
    for pollutant in ["pm25", "pm10", "no2"]:
        for lag in [1, 4, 96]:
            feature_frame[f"{pollutant}_lag_{lag}"] = feature_frame[pollutant].shift(lag)

    latest_input = feature_frame[FEATURES].iloc[[-1]]
    if latest_input.isna().any().any():
        raise ValueError("Latest model input has missing values; check history length and gaps.")

    gb_model = joblib.load(GB_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    gb_predictions = np.asarray(gb_model.predict(latest_input)).reshape(-1)
    lr_predictions = np.asarray(lr_model.predict(latest_input)).reshape(-1)
    if len(gb_predictions) != 6 or len(lr_predictions) != 6:
        raise ValueError("Expected both models to return exactly six predictions.")

    forecast = pd.DataFrame({
        "Target": TARGETS,
        "Gradient_Boosting": gb_predictions,
        "Linear_Regression": lr_predictions,
    })
    forecast["Model_Average"] = forecast[["Gradient_Boosting", "Linear_Regression"]].mean(axis=1)
    forecast["prediction_timestamp_ist"] = combined.index[-1].tz_convert("Asia/Kolkata").isoformat()
    forecast.to_csv(FORECAST_PATH, index=False)
    log("Forecast table updated")
    print("\nLATEST FORECASTS")
    print(forecast.to_string(index=False))
    print(f"\nUpdated files:\n- {LATEST_PATH}\n- {HISTORY_PATH}\n- {FORECAST_PATH}")
    log("Update completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
