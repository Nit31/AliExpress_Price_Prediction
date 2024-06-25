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