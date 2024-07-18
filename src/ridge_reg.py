import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

def run_ridge_regression(cfg, X_train, X_val, X_test, y_train, y_val, y_test):
    ridge = DecisionTreeRegressor()
    # Define the hyperparameters and their variations for the grid search
    # TODO: Write them into config
    param_grid = dict(cfg.models.ridge_regression)
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='r2', return_train_score=True)
    # Fit the model on your training data
    grid_search.fit(X_train, y_train)
    for i, params in enumerate(ParameterGrid(param_grid)):
        # Extract metrics from the grid search results
        mean_train_score = grid_search.cv_results_['mean_train_score'][i]
        mean_test_score = grid_search.cv_results_['mean_test_score'][i]
        mean_fit_time = grid_search.cv_results_['mean_fit_time'][i]
        mean_score_time = grid_search.cv_results_['mean_score_time'][i]
        print(f"{i}: {mean_train_score} - {mean_test_score} - {mean_fit_time} - {mean_score_time}")

        """
            # Start a nested MLflow run for the current iteration
            with mlflow.start_run(run_name=f"{cfg.model.name}_run_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_train_score", mean_train_score)
                mlflow.log_metric("mean_test_score", mean_test_score)
                mlflow.log_metric("mean_fit_time", mean_fit_time)
                mlflow.log_metric("mean_score_time", mean_score_time)
                mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

                # Log the source code and config files as artifacts
                mlflow.log_artifact("config/config.yaml")
                mlflow.log_artifact(f"config/model/{cfg.model.name}.yaml")"""



