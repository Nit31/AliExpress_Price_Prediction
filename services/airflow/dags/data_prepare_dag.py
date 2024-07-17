from datetime import timedelta

from airflow import DAG
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

def load_and_execute_data_prepare(**kwargs):
    """
    Load and execute the data preparation pipeline.
    """
    # FIXME:
    import subprocess
    subprocess.run(['python', 'services/airflow/dags/data_prepare.py'], check=True)

# Define DAG
with DAG(
    'data_prepare_dag',
    default_args=default_args,
    description='DAG for data preparation',
    schedule='5 * * * *',
    start_date=days_ago(1),
    tags=['data_preparation', 'zenml'],
) as dag:

    # External task sensor to wait for data extraction pipeline completion
    wait_for_extraction = ExternalTaskSensor(
        task_id='wait_for_extraction',
        external_dag_id='data_extract_dag',
        external_task_id='load_sample_to_data_store_task',
        timeout=timedelta(minutes=2),  # Adjust timeout as needed
    )

    # Bash task to run the ZenML pipeline
    run_zenml_pipeline = PythonOperator(
        task_id='run_zenml_pipeline',
        # FIXME:
        # bash_command='python $AIRFLOW_HOME/dags/data_prepare.py -prepare_data_pipeline ',  # Replace with your ZenML pipeline path
        python_callable=load_and_execute_data_prepare,
        provide_context=True,
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
