# take a sample and validate it
python -c 'from src.data import *; test_data()'
# version the data
dvc add data/samples/sample.csv
dvc push