from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime
import subprocess
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import warnings
warnings.filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Append the current working directory to the system path
sys.path.append(os.getcwd())
from src.data import sample_data, handle_initial_data, validate_initial_data

# Define the default arguments
default_args = {
    'start_date': datetime(2022, 1, 1, tz="UTC"),
    'catchup': False
}

def load_and_execute_sample_data():
    # Call the sample_data function with the loaded config
    sample_data(cfg)

def load_and_execute_validate_data():
    # Call the validate_initial_data function with the loaded config
    try:
        # if the validation failed, then try to handle the initial data
        assert validate_initial_data(cfg)
    except Exception as e:
        handle_initial_data(cfg)
    assert validate_initial_data(cfg)

# Define the DAG
with DAG(
    dag_id="data_extract_dag",
    schedule='5 * * * *',
    default_args=default_args,
    description='An automated workflow for data extraction, validation, versioning, and loading',
) as dag:
    # Initialize Hydra and load the config
    hydra.initialize(config_path="../configs", job_name="data_extract_dag")
    cfg = hydra.compose(config_name="main")

    # Define a PythonOperator to execute the load_and_execute_sample_data function
    extract_sample_task = PythonOperator(
        task_id='extract_sample_task',
        python_callable=load_and_execute_sample_data,
    )

    # Define a PythonOperator to execute the load_and_execute_validate_data function
    validate_initial_data_task = PythonOperator(
        task_id='validate_initial_data_task',
        python_callable=load_and_execute_validate_data,
    )

    # Set the task dependencies
    extract_sample_task >> validate_initial_data_task
