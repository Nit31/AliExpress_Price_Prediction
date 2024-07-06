import os
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import great_expectations as gx
from great_expectations.data_context import FileDataContext
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import zenml

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

    # Remove temporary file
    os.remove(f'{cfg.db.kaggle_filename}_csv.csv')
    os.remove(f'{cfg.db.kaggle_filename}_json.json')
    
    # Number of sample part that we need(can be changed in config/main.yaml)
    number_of_sample = cfg.db.sample_part
    df_sortes = df.sort_values(by=['lunchTime'])
    if number_of_sample not in [1, 2, 3, 4, 5]:
        print('Number of the sample should be < that 6 and > 0')
        exit(0)
    if number_of_sample not in [1, 2, 3, 4, 5]:
        print('Number of the sample should be < that 6 and > 0')
        exit(0)
    df_sample = df_sortes[(len(df_sortes) // 5) * (number_of_sample - 1):(len(df_sortes) // 5) * (number_of_sample)]
    df_sample.to_csv(cfg.db.sample_path)


def handle_initial_data(cfg):
    """
    This function cleans the raw data.
    """

    df = pd.read_csv(cfg.db.sample_path)
    df = df.drop_duplicates(['id'])

    def clean_sold(sold_data: str) -> int:
        try:
            sold_count, _ = str.split(sold_data)
        except ValueError:
            sold_count = sold_data
        return int(sold_count)

    df['sold'] = df['sold'].apply(clean_sold)
    df.to_csv(cfg.db.sample_path, index=False)


def validate_initial_data(cfg):
    """This function declares expectations about the data features and validates them.
    Returns:
        result (bool): Status of data validation.
    """
    # Create or open a data context
    try:
        context = gx.get_context(context_root_dir = "services/gx")
    except Exception as e:
        context = FileDataContext(context_root_dir = "services/gx")

    # Add data source and data asset
    data_source = context.sources.add_or_update_pandas(name="sample")
    data_asset = data_source.add_csv_asset(
        name = "sample1",
        filepath_or_buffer=cfg.db.sample_path
    )

    # Create batch request
    batch_request = data_asset.build_batch_request()

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

    # Build the data docs (website files)
    # context.build_data_docs()

    # Open the data docs in a browser
    # context.open_data_docs()

    return checkpoint_result.success

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def test_data(cfg: DictConfig = None):
    """
    This function creates a data sample and validates it.
    """

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # take a sample
    sample_data(cfg)
    # validate the sample
    try:
        # if the validation failed, then try to handle the initial data
        assert validate_initial_data(cfg)
    except Exception as e:
        handle_initial_data(cfg)
    assert validate_initial_data(cfg)

    print('Data is valid.')


def read_datastore():
    df = pd.read_csv("data/samples/sample.csv")
    version = 'v.1'
    return df, version


def preprocess_data(df: pd.DataFrame):
    X = df.drop('price', axis=1)
    X = X[['title', 'rating', 'sold', 'discount', 'shippingCost', 'year', 'month', 'category_name', 'type']]
    y = df['price']
    numerical_features = ['rating', 'sold', 'discount', 'shippingCost']
    categorical_features = ['category_name', 'type', 'year', 'month']

    # Define the transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply one-hot encoding
    ])

    # Combine the transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            # ('text', text_transformer, text_features)
        ]
    )

    # Create the pipeline with the preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    X_transformed = pipeline.fit_transform(X)
    num_features_names = numerical_features
    cat_features_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = list(num_features_names) + list(cat_features_names)
    df_transformed = pd.DataFrame(X_transformed, columns=all_feature_names)
    model_name = 'roberta-base'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = generate_embeddings(X['title'], model,device)
    title_feature_name = [f'title_{i}' for i in range(embeddings.shape[1])]
    df_titles = pd.DataFrame(embeddings.numpy(), columns=title_feature_name)
    df_transformed = pd.concat([df_transformed,df_titles],axis=1)
    del model
    torch.cuda.empty_cache()
    return df_transformed, y


def validate_features(X,y):
    # Create or open a data context
    try:
        context = gx.get_context(context_root_dir = "services/gx")
    except Exception as e:
        context = FileDataContext(context_root_dir = "services/gx")

    # Add data source and data asset
    data_source = context.sources.add_or_update_pandas(name="features_sample")
    data_asset = data_source.add_csv_asset(
        name = "features_sample",
        dataframe=pd.concat([X,y],axis=1)
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


    # Build the data docs (website files)
    # context.build_data_docs()

    # Open the data docs in a browser
    # context.open_data_docs()

    return checkpoint_result.success


def load_features(X,y,version):
    df = pd.concat([X,y],axis=1)
    zenml.save_artifact(data = df, name = "features_target", version=version, tags=[version])
    if zenml.load_artifact(version=version) is None:
        raise Exception("Artifact not loaded")


if __name__ == '__main__':
    test_data()
