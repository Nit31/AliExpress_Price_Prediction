# AliExpress_Price_Prediction
 Innopolis University MLOps capstone project
### Requirements
- Python 3.11
### Configuration
- The dataset is taken from kaggle. Create a file `confidential.yaml` in the `configs` directory, following the template:
```yaml
kaggle: 
    kaggle_username: your-kaggle-username
    kaggle_key: your-kaggle-api-key
```
- Create a virtual environment. Enter it. Install requirements by running `./scripts/install_requirements.sh`
- Export paths in your working terminal
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export ZENML_CONFIG_PATH=$PWD/services/zenml
export AIRFLOW_HOME=$PWD/services/airflow
export PYTHONPATH=$PWD/src
source venv/bin/activate
```
- Configure the airflow

### MLflow entry points
It is recomended to start the mlflow server before running any entry point. Start the server with the command `mlflow server`
- main: runs model training
- validate: validates the model with alias 'champion'
- extract: extracts the data sample to `data/samples/sample.csv`. It will give you the next version of the sample. You can find the current version in `configs/data_version.yaml`
- transform: performs transfrom of the data sample. Loads the preprocessed artifact, saves the preprocessors 
- evaluate: evaluates the model with alias 'champion'
- predict: takes a random feature sample and sends it to the deployed server in Docker
- deploy: build, run and push the Docker image

To run the entry point, use
```bash
mlflow run . --env-manager=local -e entry_point
```

### Before training models
- Be sure that you have run extract & transform for each data sample starting from 1

### Run as a local hosted service
- Install requirements by running `./scripts/install_requirements.sh`
- Load model files(from your model artifact) to the folde api/model_dir:
    - conda.yaml
    - imput_example.json
    - MLmodel
    - python_env.yaml
    - requirements.txt
    - data
        - model.pth
        - pickle_module_info.txt
- Run scripts/run_service.sh




