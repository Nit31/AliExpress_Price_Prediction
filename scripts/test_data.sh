# take a sample and validate it
python src/data.py
# version the data
dvc add data/samples/sample.csv
git add data/samples/sample.csv.dvc
git commit -m "Added data.csv to DVC"
git push origin master:dev
# Read version from data.data_version.yaml
version=$(cat data.data_version.yaml | grep 'version' | cut -d ' ' -f 2)

git tag v$version
git push origin master:dev v$version
dvc push