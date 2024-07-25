## Models Folder

This folder contains two subfolders:

* **challengers**: This folder stores trained challenger models.
* **champion**: This folder stores the currently best champion model.

Each subfolder contains:

* **models** folder with weights and configurations of the models
* Performance graphs of models

You can use these models to make predictions on new data. To load a model, you can use the following Python code:

```python
import mlflow
import torch

model = mlflow.pytorch.load_model(f'path/to/the/models/folder', 
                                  map_location=device)

```
