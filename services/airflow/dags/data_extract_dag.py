from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime
import yaml
import io
import hydra
from omegaconf import open_dict
from data import sample_data, handle_initial_data, validate_initial_data
import warnings
import subprocess
from git import Repo
warnings.filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Define the default arguments
default_args = {
    'start_date': datetime(2022, 1, 1, tz="UTC"),
    'catchup': False
}

def version_data(cfg, sample):
    """
    Version the validated sample using DVC.
    """
    # Write the sample data to the file
    sample.to_csv(cfg.db.sample_path)

    # Save data version
    try:
        with io.open(cfg.dvc.data_version_yaml_path, 'w', encoding='utf8') as outfile:
            yaml.dump({"version" : cfg.db.sample_part}, outfile, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(e)
        raise

    # Use subprocess to run DVC commands
    try:
        subprocess.run(["dvc", "add", cfg.db.sample_path], check=True)
        subprocess.run(["git", "add", cfg.dvc.sample_info_path], check=True)
        subprocess.run(["git", "add", cfg.dvc.data_version_yaml_path], check=True)
        # Check for changes before committing
        result = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE)
        changed_files = result.stdout.decode().splitlines()
        files_to_check = {cfg.dvc.sample_info_path, cfg.dvc.data_version_yaml_path}
        # Determine if the specific files were changed
        files_changed = any(any(file in line for file in files_to_check) for line in changed_files)
        if files_changed:
            # Commit the changes to Git repository.
            subprocess.run(["git", "commit", "-m", "Versioned sample data"], check=True)
            # Push the commit to Github.
            subprocess.run(["git", "push"], check=True)
            # Check if the tag already exists
            repo = Repo('.')
            tag_name = f"v{cfg.db.sample_part}"
            if tag_name in repo.tags:
                # Delete the existing tag locally
                subprocess.run(["git", "tag", "-d", tag_name], check=True)
                # Delete the existing tag from the remote repository
                subprocess.run(["git", "push", "origin", "--delete", tag_name], check=True)
            # Tag the commit and push the tag to Github
            subprocess.run(["git", "tag", "-a", f"v{cfg.db.sample_part}", "-m", f"add data version v{cfg.db.sample_part}"], check=True)
            subprocess.run(["git", "push", "--tags"], check=True)
        else:
            print("No changes to commit.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running shell commands: {e}")
        raise


def load_and_execute_sample_data(**kwargs):
    """
    Extract a new sample of the data
    """
    # Parse data_version
    with open(cfg.dvc.data_version_yaml_path) as stream:
        try:
            data_version = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    with open_dict(cfg):
        cfg.db.sample_part = data_version['version']

    # Increment version by 1. Assume that there are only 5 samples
    cfg.db.sample_part = cfg.db.sample_part % 5 + 1
    sample = sample_data(cfg)
    # Push the sample to XCom
    kwargs['ti'].xcom_push(key='sample', value=sample)


def load_and_execute_validate_data(**kwargs):
    """
    Validate the sample using Great Expectations
    """
    # Pull the sample from XCom
    sample = kwargs['ti'].xcom_pull(key='sample', task_ids='extract_sample_task')
    # Call the validate_initial_data function with the loaded config
    try:
        # If the validation failed, then try to handle the initial data
        assert validate_initial_data(cfg, sample)
    except Exception:
        sample = handle_initial_data(cfg, sample)
    assert validate_initial_data(cfg, sample)

    # Push the sample to XCom
    kwargs['ti'].xcom_push(key='sample', value=sample)


def version_validated_sample(**kwargs):
    # Pull the sample from XCom
    sample = kwargs['ti'].xcom_pull(key='sample', task_ids='validate_initial_data_task')
    # Version the validated sample using DVC
    version_data(cfg, sample)


def load_sample_to_data_store(**kwargs):
    # Use subprocess to run DVC commands
    try:
        # Push the data to the DVC remote registry.
        subprocess.run(["dvc", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running DVC commands: {e}")
        raise

# Define the DAG
with DAG(
    dag_id="data_extract_dag",
    schedule='5 * * * *',
    default_args=default_args,
    description='An automated workflow for data extraction, validation, versioning, and loading',
) as dag:
    # Initialize Hydra and load the config
    hydra.initialize(config_path="../../../configs", job_name="data_extract_dag")
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

    # Define a PythonOperator to execute the version_validated_sample function
    version_validated_sample_task = PythonOperator(
        task_id='version_validated_sample_task',
        python_callable=version_validated_sample,
        provide_context=True,
    )

    load_sample_to_data_store_task = PythonOperator(
        task_id='load_sample_to_data_store_task',
        python_callable=load_sample_to_data_store,
        provide_context=True,
    )

    # Set the task dependencies
    extract_sample_task >> validate_initial_data_task >> version_validated_sample_task >> load_sample_to_data_store_task
