# take a sample and validate it
python -c 'from src.data import *; test_data()'
# version the data
dvc add data/samples/sample.csv
git add data/samples/sample.csv
git commit -m "Added data.csv to DVC"
git push origin master:dev
git tag v1.0.0
git push origin dev v1.0.0
dvc pushgit a