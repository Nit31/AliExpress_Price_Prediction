## Services Folder

This folder contains the following subfolders:

 airflow: This folder contains the DAGs for our ETL pipelines and all configuration for the Apache Airflow service.
 gx: This folder contains the Great Expectations service.
 zenml: This folder contains information about the ZenML server.

### Airflow

The Airflow service is responsible for orchestrating our ETL pipelines. The DAGs in this folder define the data sources, transformations, and destinations for our pipelines.

To start the Airflow service follow instrutions in inner README file

Great Expectations

The Great Expectations service is responsible for data validation and testing. The configuration in this folder defines the expectations that our data must meet.

ZenML

The ZenML server provides a centralized interface for managing and monitoring our machine learning pipelines. The configuration in this folder defines the pipelines that are deployed to the server.

To start the ZenML server, run the following command:

```bash
zenml up
```
