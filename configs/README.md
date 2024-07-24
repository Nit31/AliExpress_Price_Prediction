# Configuration Files

This folder contains configuration files for the project.

## File Structure

* *[main.yaml]* -  ["Main config file. There we store most of the information. By now, there is data about kaggle dataset that we using in our project."]
* *[confidential.yaml]* -  ["Config with confidential information like API tokens for KaggleAPI."]
* *[.gitignore, __pycache__, __init__]* - ["System files."]
* *[api.yaml]* -  ["Config for "]
* *[data_version.yaml]* -  ["Config for storing version of sampled data"]
* *[experiment.yaml]* -  ["Config for mlflow experiments"]
* *[giskard.yaml]* -  ["Config for Giskard validation service"]
* *[zenml.yaml]* -  ["Config fordifferent types of data features"]

## Configuration Format

Configuration files are in YAML format.

## Usage

All configs usage is automated using hydra library. You can use it for example by adding special decorators to your functions.
