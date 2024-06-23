import os
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):

    with open("configs/confidential.yaml", "r") as stream:
        config_confidential_data = yaml.safe_load(stream)

    kaggle_data = config_confidential_data['kaggle']

    os.environ['KAGGLE_USERNAME'] = kaggle_data['kaggle_username']
    os.environ['KAGGLE_KEY'] = kaggle_data['kaggle_key']

    # We need to import API library after setting up environmental variables 
    from kaggle.api.kaggle_api_extended import KaggleApi

    def download_kaggle_dataset(dataset_name, output_dir):

        api = KaggleApi()
        api.authenticate()  # Make sure to set up your Kaggle API authentication as mentioned in the previous messages

        try:
            api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
            print("Dataset downloaded successfully to:", output_dir)
        except Exception as e:
            print("Error downloading dataset:", str(e))

    dataset_name = f'{cfg.db.db_creator}/{cfg.db.db_name}'  # Replace with the actual Kaggle dataset name
    output_directory = "."  # Specify where you want to download the dataset__

    download_kaggle_dataset(dataset_name, output_directory)

    # Convert downloaded CSV to Pandas DataFrame
    df = pd.read_csv(f'{cfg.db.kaggle_filename}_csv.csv')

    # Remove temporary file
    os.remove(f'{cfg.db.kaggle_filename}_csv.csv')
    os.remove(f'{cfg.db.kaggle_filename}_json.json')
    
    # Number of sample part that we need(can be changed in config/main.yaml)
    number_of_sample = cfg.db.sample_part
    df_sortes = df.sort_values(by=['lunchTime'])
    if number_of_sample not in [1, 2, 3, 4, 5]:
        print('Number of the sample should be < that 6 and > 0')
        exit(0)
    df_sample = df_sortes[(len(df_sortes) // 5) * (number_of_sample - 1):(len(df_sortes) // 5) * (number_of_sample)]
    # df_sample.to_csv(cfg.db.sample_path)
    pass
sample_data()