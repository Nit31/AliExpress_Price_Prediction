import io
import os.path
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg : DictConfig = None) -> None:

    # Credentials to connect to google drive
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    SERVICE_ACCOUNT_FILE = cfg.db.google_credentials_path # Path to your service account credentials JSON file

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    # Note: File ID need to be changed in configs/main.yanl, if new version of data loaded to the cloud storage
    file_id = cfg.db.file_id

    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = io.BytesIO()
    downloader.write(request.execute())
    downloaded.seek(0)

    # Temporary save downloaded data to the file
    with open('temp.csv', 'wb') as f:
        f.write(downloader.getvalue())

    # Convert downloaded CSV to Pandas DataFrame
    df = pd.read_csv('temp.csv')

    # Remove temporary file
    os.remove('temp.csv')

    # Number of sample part that we need(can be changed in config/main.yaml)
    number_of_sample = cfg.db.sample_part
    df_sortes = df.sort_values(by=['lunchTime'])
    df_sample = df_sortes[(len(df_sortes) // 5) * (number_of_sample - 1):(len(df_sortes) // 5) * (number_of_sample)]
    df_sample.to_csv(cfg.db.sample_path)
    pass

