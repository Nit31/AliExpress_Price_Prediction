# AliExpress_Price_Prediction
 Innopolis University MLOps capstone project

### How to run (for developers)
- Create a file `confidential.yaml` in the `configs` directory, following the template:
```yaml
kaggle: 
    kaggle_username: your-kaggle-username
    kaggle_key: your-kaggle-api-key
```
- Install requirements by running `./scripts/install_requirements.sh`
- Now you can run `./scripts/test_data` to get a data sample

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




