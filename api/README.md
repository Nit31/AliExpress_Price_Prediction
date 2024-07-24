## API Folder Structure

This folder contains the following files:

* **app.py**: This file contains the Flask API code.
* **nn_model.py**: This file describes the model architecture.

## Getting Started

To get started with this folder, you will need to:

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Train the model and save it to model_dir
3. Start the Flask API by running `python app.py`.

## Usage

To use the Flask API, you can send a POST request to the `/predict` endpoint with the following JSON payload:

```json
{
  "inputs": {
	features and values in dict format
}
```

The API will return a JSON response with the predicted values.
