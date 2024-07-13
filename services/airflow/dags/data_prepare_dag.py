from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

# Define DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Define DAG
with DAG(
    'data_prepare_dag',
    default_args=default_args,
    description='DAG for data preparation',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    tags=['data_preparation', 'zenml'],
) as dag:

    # External task sensor to wait for data extraction pipeline completion
    wait_for_extraction = ExternalTaskSensor(
        task_id='wait_for_extraction',
        external_dag_id='data_extract_dag',
        external_task_id='data_extraction',
        timeout=timedelta(minutes=1),  # Adjust timeout as needed
    )

    # Bash task to run the ZenML pipeline
    run_zenml_pipeline = BashOperator(
        task_id='run_zenml_pipeline',
        bash_command='zenml pipeline run pipelines/data_prepare_pipeline.py',  # Replace with your ZenML pipeline path
        retries=1,
        retry_delay=timedelta(seconds=3),  # Adjust retry delay as needed
    )

    # Define dependencies
    wait_for_extraction >> run_zenml_pipeline

    # Optional: Load features to feature store
    # load_features = PythonOperator(
    #     task_id='load_features',
    #     python_callable=load_features_to_feature_store,  # Replace with your feature loading function
    # )
    # run_zenml_pipeline >> load_features
