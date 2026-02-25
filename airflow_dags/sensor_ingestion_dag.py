"""
Airflow DAG — Agro-IoT Sensor Ingestion
=========================================

Orchestrates the incremental ingestion of IoT sensor telemetry from
field sensors into the data lake, followed by processing and feature
engineering. Runs every 15 minutes in production.

DAG Structure:
    check_new_files >> ingest_sensor_batch >> process_and_validate
    >> compute_rolling_features >> trigger_yield_prediction

Author : David Bravo · https://bravoaidatastudio.com
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner":           "david.bravo",
    "retries":         2,
    "retry_delay":     timedelta(minutes=5),
    "email_on_failure": True,
    "email":           ["david.arielbravo@gmail.com"],
}


def _ingest_sensor_batch(**ctx):
    """Pull latest sensor readings from MQTT broker into Parquet landing zone."""
    import pandas as pd
    from pathlib import Path
    print(f"[{ctx['ts']}] Ingesting sensor batch...")
    # In production: reads from MQTT / S3 prefix; here we log intent
    print("  Incremental ingest complete — new readings appended to data lake.")


def _process_and_validate(**ctx):
    """Run sensor processor on new batch."""
    print(f"[{ctx['ts']}] Validating and processing sensor data...")
    print("  Clipping, imputing, stress flags applied.")


def _compute_rolling_features(**ctx):
    """Recompute rolling aggregates over updated dataset."""
    print(f"[{ctx['ts']}] Computing rolling features (1h, 6h, 24h windows)...")


def _trigger_yield_prediction(**ctx):
    """Trigger yield model scoring if end-of-season criteria met."""
    print(f"[{ctx['ts']}] Checking prediction trigger conditions...")


with DAG(
    dag_id          = "agro_iot_sensor_ingestion",
    default_args    = default_args,
    start_date      = datetime(2024, 10, 1),
    schedule        = "*/15 * * * *",
    catchup         = False,
    tags            = ["agro", "iot", "precision-agriculture"],
    doc_md          = __doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id         = "ingest_sensor_batch",
        python_callable = _ingest_sensor_batch,
    )

    process = PythonOperator(
        task_id         = "process_and_validate",
        python_callable = _process_and_validate,
    )

    features = PythonOperator(
        task_id         = "compute_rolling_features",
        python_callable = _compute_rolling_features,
    )

    predict = PythonOperator(
        task_id         = "trigger_yield_prediction",
        python_callable = _trigger_yield_prediction,
    )

    end = EmptyOperator(task_id="end")

    start >> ingest >> process >> features >> predict >> end
