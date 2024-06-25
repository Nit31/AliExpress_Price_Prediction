# Data

This folder contains the initial dataset used in this project. 

## Structure

* *samples:*  Sample data files. 
    * `sample.csv`:  A sample of dataset, tracked with DVC. *See important note below.*
* *imgs:*  Images used in the project or related to the sample data.

## Important Notes

* *DVC Tracking:* The file `samples/sample.csv` is tracked using Data Version Control (DVC). 
   *  To get the latest version of this file, make sure you have DVC installed and run `dvc pull` inside this directory.
   * For more information about DVC and how to use it, refer to the [DVC documentation](https://dvc.org/doc). 

## Updating Data

* *Do not directly modify* the files in this folder, especially `samples/sample.csv`. 
* To add or change data:
    1. Modify the original data source.
    2. Update the sample data generation process (if needed).
    3. Regenerate the sample data files.
    4. Use DVC to track the changes (e.g., `dvc add data/samples/sample.csv` and then `dvc push`).