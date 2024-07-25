# Loop through example_version from 1 to 5
for example_version in {1..5}
do
    echo "Running example_version $example_version"
    python src/predict.py --example_version $example_version
done