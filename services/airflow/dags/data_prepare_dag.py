from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import subprocess
from pendulum import datetime


def load_and_execute_data_prepare(**kwargs):
    """
    Load and execute the data preparation pipeline.
    """
    # FIXME:
    subprocess.run(["python", "services/airflow/dags/data_prepare.py"], check=True)


# Define DAG2
with DAG(
    "data_prepare_dag",
    description="DAG for data preparation",
    schedule="0/5 * * * *",
    start_date=datetime(2022, 1, 1, tz="UTC"),
    tags=["data_preparation", "zenml"],
    catchup=False,
    max_active_runs=1
) as dag:

    # External task sensor to wait for data extraction pipeline completion
    wait_for_extraction = ExternalTaskSensor(
        task_id="wait_for_extraction",
        external_dag_id="data_extract_dag",
        external_task_id="load_sample_to_data_store_task",
        timeout=timedelta(minutes=2),  # Adjust timeout as needed
    )

    # Bash task to run the ZenML pipeline
    run_zenml_pipeline = PythonOperator(
        task_id="run_zenml_pipeline",
        python_callable=load_and_execute_data_prepare,
        provide_context=True,
        retries=1,
        retry_delay=timedelta(seconds=3),  # Adjust retry delay as needed
    )

    # Define dependencies
    wait_for_extraction >> run_zenml_pipeline
