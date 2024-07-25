# Scripts

This folder contains shell scripts that automate key tasks related to data management and testing for the project. These scripts primarily execute functions defined in the `srd.data.py` module.

## Script Descriptions:

* *`install_requirements.sh`:*  Installs all the necessary dependencies for the project. This script is usually the first one to run when setting up the project environment.
* *`test_data.sh`:* Sample data from initial source. Runs data validation checks on the sampled data to ensure its integrity and quality.
  And version the sample using DVC.
* *`run_service.sh`:* Host Flask API and Streamlit UI localy, so user can see the functionalities od the model. More detailed instruction in root folder README

Each script can be executed from the command line
