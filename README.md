# AliExpress_Price_Prediction
 Innopolis University MLOps capstone project. The aim of this project is to apply the CRISP-ML process on a machine learning project which begins with business and data understanding till model deployment and monitoring. The phases of the process are as follows:

1. Business and data understanding
    - Elicit the project requirements and formulate business problem
    - Specify business goals and machine learning goals
    - Determine the success criteria for business and machine learning modeling
    - Analyse the risks and set mitigation approaches
2. Data engineering/Preparation
    - Create ETL pipelines using Apache Airflow
    - Perform data transformation
    - Check the quality of the data and perform data cleaning
    - Create ML-ready features and store them in feature stores such as feast
3. Model engineering
    - Select and build ML models
    - Perform and track experiments using MLflow
    - Optimize models and select best models
    model versioning in model registry of MLflow
4. Model validation
    - Prepare one model for production
    - Check the success criteria of machine learning
    - Check the business and machine learning modeling objectives
    - The business stakeholders take part in this phase
    - Check if deploying the model is feasible
    - Check the quality of the model for production
    - Select one model to be deployed
5. Model deployment
    - Search for options available to serve the model
    - Deploy the model
    - Create a REST endpoint for your model prediction using Flask or FastAPI
    - Create a UI for your model using streamlit or pure HTML and JS.
    - Create a CI/CD pipeline for your model using Github Actions and Docker

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




<<<<<<< HEAD
=======
- Now you can run `./scripts/test_data` to get a data sample
>>>>>>> aca200e1e6d3c792bb115d930ae677bd49eb462c
