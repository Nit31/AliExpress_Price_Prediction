# Tests

This folder contains tests for the project's data handling functionality, specifically focusing on the `src.data` module.

## File Structure:

* *`test_sample_data.py`:* This file includes pytest test functions specifically designed to test the `sample_data` function from the `src.data` module.
* *`__pycache__/`:* This directory stores compiled Python bytecode files (`.pyc` files) generated by Python during test execution. It's automatically managed by Python.
* *`pytest_cache/`:*  This directory stores pytest's internal cache data to speed up subsequent test runs. It is managed by pytest. 
* *`__init__.py`:* This file makes Python treat the "test" folder as a package. This is important for pytest to discover and execute the tests correctly.

## Running Tests

Make sure you installed the requirements.txt using scripts/install_requirements.sh. Then you can run "pytest comand from the root of the repository and testing will be processed."
