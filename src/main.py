import pandas as pd
import hydra
import torch
import zenml
from sklearn.model_selection import ParameterGrid
from nn_model import set_seed
from omegaconf import OmegaConf
import mlflow
from mlflow.tracking import MlflowClient
from nn_model import train_and_evaluate_model
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_split_data(cfg):
    try:
        train = zenml.load_artifact(
            name_or_id="features_target", version=str(cfg.mlflow.train_val_data_version)
        )
        test = zenml.load_artifact(
            name_or_id="features_target", version=str(cfg.mlflow.test_data_version)
        )
    except Exception as e:
        print("Error loading zenml artifacts\n")
        raise e

    X_train, y_train = train.drop(columns=["price"]), train["price"]

    # sample out of test data
    test = pd.DataFrame(test)
    test = test.sample(
        frac=cfg.mlflow.test_size, random_state=cfg.experiment.random_state
    )

    X_test = test.drop(columns=["price"])
    y_test = test["price"]
    return X_train, X_test, y_train, y_test


def log_metadata(cfg, models, X_train, X_test, y_train, y_test):
    # FIXME: Need to start mlflow server. Maybe need to reconsider
    mlflow.set_tracking_uri("http://localhost:5000")
    architecture_params = models[0]["params"]
    experiment_name = f"Model_dim{architecture_params['hidden_dim']}_layers{architecture_params['hidden_layers']}_experiment"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        # experiment_id = mlflow.create_experiment(experiment_name).experiment
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(experiment_id)
    # FIXME: In first run mlflow always save in default experiment
    with mlflow.start_run(run_name="first_run", experiment_id=experiment_id) as run:
        pass
    for i, whole_params in enumerate(models):
        try:
            model = whole_params["model"]
            mean_score = whole_params["mean_score"]
            params = whole_params["params"]
            run_name = f"r2_score{mean_score}_run{i}"

            with mlflow.start_run(
                run_name=run_name, experiment_id=experiment_id
            ) as run:
                print(f"exp:{mlflow.get_run(run.info.run_id).info.experiment_id}")
                try:
                    mlflow.log_params(params)
                except mlflow.exceptions.MlflowException as e:
                    if "Changing param values is not allowed" in str(e):
                        print("Parameter already logged. Skipping...")
                mlflow.log_metrics({"train r2": mean_score})
                mlflow.set_tag(
                    "Training Info", "Fully-connected model architecture for aliexpress"
                )
                # Infer the model signature
                signature = infer_signature(X_train, y_train)
                # Log the model
                model.to("cpu")
                mlflow.pytorch.log_model(
                    pytorch_model=model.model,
                    artifact_path=f"models/{experiment_name}/{run_name}",
                    signature=signature,
                    input_example=X_train.head(),  # Use X_train or X_test as needed
                    registered_model_name=f"{experiment_name}_{run_name}",
                )
                model.to("cuda" if torch.cuda.is_available() else "cpu")
                torch.cuda.empty_cache()
                eval_data = pd.DataFrame(y_test.values, columns=["real"])
                # Get predictions from the model
                device = next(model.model.parameters()).device
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(
                    device
                )

                # Get predictions from the model
                eval_data["predictions"] = (
                    model.model(X_test_tensor).detach().cpu().numpy()
                )
                mae = mean_absolute_error(eval_data["real"], eval_data["predictions"])
                mse = mean_squared_error(eval_data["real"], eval_data["predictions"])
                r2 = r2_score(eval_data["real"], eval_data["predictions"])

                # Print the metrics
                print(f"Mean Absolute Error: {mae}")
                print(f"Mean Squared Error: {mse}")
                print(f"R² Score: {r2}")

                # Log metrics to MLflow
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                print("Done!")
        except Exception as e:
            print(e)
    return experiment_name


def train(X_train, y_train, cfg):
    param_grid = dict(cfg.model.params)
    models = list()
    # experiment_name = f"Model_dim{param_grid['hidden_dim']}_layers{param_grid['hidden_layers']}_experiment"
    # try:
    #     # Create a new MLflow Experiment
    #     experiment_id = mlflow.create_experiment(name=experiment_name)
    # except mlflow.exceptions.MlflowException as e:
    #     experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"Run {i}")
        model, mean_score = train_and_evaluate_model(params, X_train, y_train)
        models.append({"model": model, "mean_score": mean_score, "params": params})
    return models


def log_charts(experiment_name, cfg=None):
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
    except Exception:
        pass

    # Initialize MLflow client
    client = MlflowClient()

    # Get all experiments
    experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'")
    print(experiments)
    # List to hold all metrics data
    all_metrics_data = []

    # Iterate through all experiments
    for exp in experiments:
        exp_id = exp.experiment_id
        # Get all runs for the experiment
        runs = client.search_runs(experiment_ids=[exp_id])

        # Iterate through all runs
        for run in runs:
            run_id = run.info.run_id
            # Get metrics for the run
            data = client.get_run(run_id).data
            metrics = data.metrics

            # Collect metrics with the run ID and experiment ID
            for metric, value in metrics.items():
                all_metrics_data.append(
                    {
                        "experiment_id": exp_id,
                        "run_id": run_id,
                        "run_name": run.data.tags.get(
                            "mlflow.runName", run_id
                        ),  # Use run name if available
                        "metric": metric,
                        "value": value,
                    }
                )

            # Create and log individual metric plots
            plt.figure(figsize=(10, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            plt.bar(metric_names, metric_values)
            plt.title(f"Run ID: {run_id} Metrics")
            plt.xlabel("Metric")
            plt.ylabel("Value")
            plt.tight_layout()

            # Log the plot to MLflow within the context of the current run
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_figure(plt.gcf(), f"{run.info.run_name}_metrics_plot.png")
                plt.savefig(
                    f"results/{experiment_name}_{run.info.run_name}_metrics_plot.png"
                )
            plt.close()

    # Convert the collected metrics data to a DataFrame
    metrics_df = pd.DataFrame(all_metrics_data)

    # Pivot the DataFrame for plotting
    pivot_df = metrics_df.pivot(index="metric", columns="run_name", values="value")

    # Sort the DataFrame by the second metric
    if pivot_df.shape[0] > 1:
        pivot_df = pivot_df.sort_values(by=pivot_df.index[1], axis=1)

    # Plot combined bar chart for comparison across all models
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")

    # Create a color palette with as many colors as there are models
    palette = sns.color_palette("husl", len(pivot_df.columns))

    # Plot with seaborn for better aesthetics
    pivot_df.plot(kind="bar", figsize=(20, 10), color=palette, width=0.8)
    plt.title("Comparison of Metrics Across All Models")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Log the combined plot to MLflow
    combined_plot_path = "combined_metrics_plot.png"
    with mlflow.start_run() as run:
        mlflow.log_figure(plt.gcf(), combined_plot_path)
        plt.savefig(f"results/whole_{experiment_name}_metrics_plot.png")
    plt.close()


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.random_state)
    # Extract data
    X_train, X_test, y_train, y_test = get_split_data(cfg)

    # Train the models
    models = train(X_train.to_numpy(), y_train.to_numpy(), cfg=cfg)

    experiment_name = log_metadata(cfg, models, X_train, X_test, y_train, y_test)
    log_charts(experiment_name, cfg)


if __name__ == "__main__":
    main()
