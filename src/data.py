import os
from datetime import datetime
import pandas as pd
import hydra
from omegaconf import DictConfig, open_dict
import yaml
import great_expectations as gx
from great_expectations.data_context import FileDataContext
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zenml
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class TextDataset(Dataset):
    def __init__(self, text_series):
        self.text_series = text_series

    def __len__(self):
        return len(self.text_series)

    def __getitem__(self, idx):
        return self.text_series.iloc[idx]

def collate_fn(batch):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

def generate_embeddings(text_series, model, device, batch_size=512):
    # Create DataLoader
    dataset = TextDataset(text_series)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    embeddings_list = []

    # Generate embeddings in batches
    with torch.no_grad():
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()  # Take the mean of the last hidden states
            embeddings_list.append(embeddings)

    # Concatenate all batch embeddings
    embeddings = torch.cat(embeddings_list, dim=0)

    return embeddings


def sample_data(cfg):
    """This function download data from initial sourse(in this case Kaggle)
        using API. Then this function sample data by dividing initial raw
        data on 5 equal part regarding time feature. After all resulted
        sample saved in the data/samples/sample.csv
    Args:
        cfg (DictConfig, optional): _description_. Defaults to None.
    """
    with open("configs/confidential.yaml", "r") as stream:
        config_confidential_data = yaml.safe_load(stream)

    # Try to open the local copy of raw data
    try:
        df = pd.read_csv(cfg.db.data_path)
        print('Dataset local cache is detected.')
    # In case of failure, download the raw data from kaggle
    except Exception:
        print('Dataset local cache not detected. Downloading...')
        kaggle_data = config_confidential_data['kaggle']

        os.environ['KAGGLE_USERNAME'] = kaggle_data['kaggle_username']
        os.environ['KAGGLE_KEY'] = kaggle_data['kaggle_key']

        # We need to import API library after setting up environmental variables
        # if the order will be changed, API will not work properly
        from kaggle.api.kaggle_api_extended import KaggleApi

        def download_kaggle_dataset(dataset_name, output_dir):
            """Function to download Kaggle dataset using API
            Args:
                dataset_name (_type_): _description_
                output_dir (_type_): _description_
            """
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

        # Save the raw dataset
        df.to_csv(cfg.db.data_path)

        # Remove temporary file
        os.remove(f'{cfg.db.kaggle_filename}_csv.csv')
        os.remove(f'{cfg.db.kaggle_filename}_json.json')
        
    # Take actual sample`s vertion
    with open(cfg.dvc.data_version_yaml_path) as stream:
        try:
            data_version = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    df_sortes = df.sort_values(by=['lunchTime'])
    sample_part_int = int(data_version['version'])
    if not 1 <= sample_part_int <= 5:
        print('Sample_part should be < that 6 and > 0')
        exit(0)
    df_sample = df_sortes[(len(df_sortes) // 5) * (sample_part_int - 1):(len(df_sortes) // 5) * (sample_part_int)]
    return df_sample


def handle_initial_data(sample):
    """
    This function cleans the raw data.
    """
    df = sample
    df = df.drop_duplicates(['id'])

    def clean_sold(sold_data: str) -> int:
        try:
            sold_count, _ = str.split(sold_data)
        except ValueError:
            sold_count = sold_data
        return int(sold_count)
    df['shippingCost'] = df['shippingCost'].replace('None', np.nan).astype(float)
    mean_shipping_cost = df['shippingCost'].mean()
    df['shippingCost'].fillna(mean_shipping_cost, inplace=True)
    df['sold'] = df['sold'].apply(clean_sold)

    return df


def validate_initial_data(cfg, sample):
    """This function declares expectations about the data features and validates them.
    Returns:
        result (bool): Status of data validation.
    """
    os.environ['HYDRA_FULL_ERROR'] = '1'
    # Create or open a data context
    try:
        context = gx.get_context(context_root_dir = "services/gx")
    except Exception:
        context = FileDataContext(context_root_dir = "services/gx")

    # Add data source and data asset
    data_source = context.sources.add_or_update_pandas(name="sample")

    data_asset = data_source.add_dataframe_asset(name = "pandas_dataframe")

    batch_request = data_asset.build_batch_request(dataframe = sample)

    # Get an existing expectation suit
    context.get_expectation_suite("expectation_suite")


    # Create a validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="expectation_suite",
    )
    try:
        batch_list = data_asset.get_batch_list_from_batch_request(batch_request)

        validator.load_batch_list(batch_list)

        validations = [
            {
                "batch_request": batch.batch_request,
                "expectation_suite_name": "expectation_suite"
            }
            for batch in batch_list
        ]
        checkpoint = context.add_or_update_checkpoint(
            name="validator_checkpoint",
            validations=validations
        )

        checkpoint_result = checkpoint.run()
    except Exception as e:
        print(e)
        return False

    return checkpoint_result.success


def read_datastore():
    hydra.initialize(config_path="../configs", job_name="preprocess_data", version_base=None)
    cfg = hydra.compose(config_name="data_version")
    
    # TODO: add config with path instad hardcode
    df = pd.read_csv('data/samples/sample.csv')
    version = cfg.version
    return df, version

# Class for a binary encoder that had fixed columns
# Its necessary for data validation and decreasing number of columns
class FixedBinsBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=6):
        self.max_bins = max_bins
        self.encoder = BinaryEncoder()
    
    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self
    
    def transform(self, X):
        binary_encoded = self.encoder.transform(X)
        
        num_columns = binary_encoded.shape[1]
        
        if num_columns > self.max_bins:
            raise ValueError(f"Number of required bits ({num_columns}) exceeds the specified maximum number of bins ({self.max_bins}).")
        
        # Padding with zeros if necessary
        padded_encoded = np.pad(binary_encoded, ((0, 0), (0, self.max_bins - num_columns)), 'constant', constant_values=0)
        return padded_encoded
    

def preprocess_data(df: pd.DataFrame):
    # Adding configuration for preprocessing
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", job_name="preprocess_data", version_base=None)
    cfg = hydra.compose(config_name="features")
    
    # FIXME:
    df = df[(df['sold'] > 10) & (df['rating'] > 0)]
    df = df[df['storeName'].map(df['storeName'].value_counts()) > 3]
    
    X = df.drop('price', axis=1)
    
    # Convert Date to year, month, and day
    X['year'] = X['lunchTime'].apply(lambda date: datetime.strptime(date.split()[0], '%Y-%m-%d').year).astype(str)
    X['month'] = X['lunchTime'].apply(lambda date: datetime.strptime(date.split()[0], '%Y-%m-%d').month)
    X['day'] = X['lunchTime'].apply(lambda date: datetime.strptime(date.split()[0], '%Y-%m-%d').day)
    X = X.drop(columns=['lunchTime'])
    X = X[list(cfg.features.all)]
    y = df[str(cfg.features.target)]
    numerical_features = list(cfg.features.numerical)
    try:
        preprocessor = zenml.load_artifact(name_or_id='preprocessor', version='1')
    except:
        preprocessor = None
    if preprocessor is None:
        print('Doesn\'t find preprocessor. Creating...')
        # Define the transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # Standardize numerical features
        ])

        # Encode "type" and "month" via basic OneHot
        type_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply one-hot encoding
        ])

        month_transformer = Pipeline(steps=[
            ('mon', OneHotEncoder(handle_unknown='ignore', categories=[[i for i in range(12)]]))
        ])

        # Encode "year", "category_name" using Binary encoder with fixed number of columns
        year_transformer = Pipeline(steps=[
            ('fb_1', FixedBinsBinaryEncoder(max_bins=6))
        ])

        category_name_transformer = Pipeline(steps=[
            ('fb_2', FixedBinsBinaryEncoder(max_bins=9))
        ])


        # Combine the transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', type_transformer, ['type']),
                ('mon', month_transformer, ['month']),
                ('fb_year', year_transformer, ['year']),
                ('fb_cn', category_name_transformer, ['category_name']),
                # ('text', text_transformer, text_features)
            ]
        )
        zenml.save_artifact(preprocessor,name='preprocessor',version='1')
    try:
        target_preprocessor = zenml.load_artifact(name_or_id='target_preprocessor', version='1')
    except Exception as e:
        print(e)
        target_preprocessor = None
    if target_preprocessor is None:
        # If there is no preprocessor, and the sample version is not 1, then raise error
        if cfg.data_version.version != 1:
            raise ValueError("No target preprocessor found. Sample version is not 1.\
                             Please, run preprocess_data.py with sample version 1.")

        print('Doesn\'t find target preprocessor. Creating...')
        # Define the transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # Standardize numerical features
        ])
        # Combine the transformers into a ColumnTransformer
        target_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, y.columns),
            ]
        )
        # Fit the pipeline on the training data
        target_preprocessor = target_preprocessor.fit(y)
        # Save the preprocessor
        zenml.save_artifact(target_preprocessor,name='target_preprocessor',version='1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        roberta_model = zenml.load_artifact(name_or_id='roberta_model', version='1')
    except:
        roberta_model = None
    if roberta_model is None:
        print('Doesn\'t find roberta. Creating...')
        model_name = 'roberta-base'
        roberta_model = RobertaModel.from_pretrained(model_name).to(device)
        roberta_model.eval()
        zenml.save_artifact(roberta_model,name='roberta_model',version='1')
    # Create the pipeline with the preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    X_transformed = pipeline.fit_transform(X)
    
    # Convert transformed data back to the dataframe
    num_features_names = numerical_features
    type_features_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['type'])
    all_feature_names = list(num_features_names) + list(type_features_names) + list([f'month_{i}' for i in range(12)]) + list([f'year_{i}' for i in range(6)]) + list([f'cn_{i}' for i in range(9)])
    df_transformed = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    # Preprocess text feature
    #model_name = 'roberta-base'

    print(device)
    # model = RobertaModel.from_pretrained(model_name).to(device)
    # model.eval()
    embeddings = generate_embeddings(X['title'], roberta_model,device)
    title_feature_name = [f'title_{i}' for i in range(embeddings.shape[1])]
    df_titles = pd.DataFrame(embeddings.numpy(), columns=title_feature_name)
    df_transformed = pd.concat([df_transformed,df_titles],axis=1)
    del roberta_model
    torch.cuda.empty_cache()
    y_df = pd.DataFrame(y)
    df_transformed.to_csv('test.csv')
    return df_transformed, y_df


def validate_features(X,y):
    # Create or open a data context
    try:
        context = gx.get_context(context_root_dir = "services/gx")
    except Exception:
        context = FileDataContext(context_root_dir = "services/gx")

    # Add data source and data asset
    data_source = context.sources.add_or_update_pandas(name="features_sample")
    data_asset = data_source.add_dataframe_asset(
        name="features_sample",
        dataframe=X
    )

    # Create batch request
    batch_request = data_asset.build_batch_request()

    # Get an existing expectation suit
    context.get_expectation_suite("features_expectation_suite")


    # Create a validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="features_expectation_suite",
    )
    batch_list = data_asset.get_batch_list_from_batch_request(batch_request)

    validator.load_batch_list(batch_list)

    validations = [
        {                "batch_request": batch.batch_request,
            "expectation_suite_name": "features_expectation_suite"
        }
        for batch in batch_list
    ]
    checkpoint = context.add_or_update_checkpoint(
            name="validator_checkpoint",
            validations=validations
    )

    checkpoint_result = checkpoint.run()
    return checkpoint_result.success

def load_features(X, y, version):
    version = str(version)
    df = pd.concat([X, y], axis=1)
    zenml.save_artifact(data=df, name="features_target", version=version, tags=[version])

    # Specify the name or ID of the artifact you want to load
    artifact_name_or_id = "features_target"  # Replace with the appropriate name or ID if needed

    if zenml.load_artifact(name_or_id=artifact_name_or_id, version=version) is None:
        raise Exception("Artifact not loaded")


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def test_data(cfg: DictConfig = None):
    """
    This function creates a data sample and validates it.
    """

    warnings.filterwarnings("ignore", category=DeprecationWarning)

     # Parse data_version
    with open(cfg.dvc.data_version_yaml_path) as stream:
        try:
            data_version = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    with open_dict(cfg):
        cfg.db.sample_part = data_version['version']

    # take a sample
    sample = sample_data(cfg)

    # validate the sample
    try:
        # if the validation failed, then try to handle the initial data
        assert validate_initial_data(cfg, sample)
    except Exception:
        sample = handle_initial_data(sample)
        #sample.to_csv(cfg.db.sample_path)
    assert validate_initial_data(cfg, sample)

    # If the data is validated, then save it
    sample.to_csv(cfg.db.sample_path)

    print('Data is valid.')

if __name__ == '__main__':
    test_data()
