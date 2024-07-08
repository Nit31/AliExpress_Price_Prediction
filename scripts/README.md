# Scripts

This folder contains shell scripts that automate key tasks related to data management and testing for the project. These scripts primarily execute functions defined in the `srd.data.py` module.

## Script Descriptions:

* *`install_requirements.sh`:*  Installs all the necessary dependencies for the project. This script is usually the first one to run when setting up the project environment.
* *`test_data.sh`:* Sample data from initial source. Runs data validation checks on the sampled data to ensure its integrity and quality.
And version the sample using DVC.
- When we need to change the sample. You need to choose new part of the sample in configs/main.yaml and change version in configs/data_version.yaml. After that you use test_data.sh . Script will save new sample to the data/samples/

## Usage:

Each script can be executed from the command line:


