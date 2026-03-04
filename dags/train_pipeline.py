from datetime import datetime, timedelta
import os
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

def on_failure_callback(context):
    ti = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = ti.task_id if ti else "unknown_task"
    run_id = context.get("run_id", "unknown_run")
    print(f"[ALERT] Failure in DAG={dag_id}, task={task_id}, run_id={run_id}")

def run_cmd(cmd: list[str]):
    print("[cmd]", " ".join(cmd))
    subprocess.check_call(cmd)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ids568", "milestone3"],
) as dag:

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=lambda: run_cmd(["python", "preprocess.py"]),
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=lambda: run_cmd(["python", "train.py", "--C", "1.0", "--max_iter", "200"]),
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=lambda: run_cmd(["python", "register_best.py"]),
    )

    preprocess_data >> train_model >> register_model