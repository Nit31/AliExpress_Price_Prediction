import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


try:
    mlflow.set_tracking_uri("http://localhost:5000")
except:
    pass
# Initialize MLflow client
client = MlflowClient()

# Get all experiments using mlflow.search_experiments
experiments = mlflow.search_experiments()

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
            all_metrics_data.append({
                'experiment_id': exp_id,
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', run_id),  # Use run name if available
                'metric': metric,
                'value': value
            })

        # Create and log individual metric plots
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        plt.bar(metric_names, metric_values)
        plt.title(f'Run ID: {run_id} Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.tight_layout()

        # Log the plot to MLflow within the context of the current run
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_figure(plt.gcf(), f'{run_id}_metrics_plot.png')

        plt.close()

# Convert the collected metrics data to a DataFrame
metrics_df = pd.DataFrame(all_metrics_data)

# Pivot the DataFrame for plotting
pivot_df = metrics_df.pivot(index='metric', columns='run_name', values='value')

# Sort the DataFrame by the second metric
if pivot_df.shape[0] > 1:
    pivot_df = pivot_df.sort_values(by=pivot_df.index[1], axis=1)

# Plot combined bar chart for comparison across all models
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")

# Create a color palette with as many colors as there are models
palette = sns.color_palette("husl", len(pivot_df.columns))

# Plot with seaborn for better aesthetics
pivot_df.plot(kind='bar', figsize=(20, 10), color=palette, width=0.8)
plt.title('Comparison of Metrics Across All Models')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Log the combined plot to MLflow
combined_plot_path = 'combined_metrics_plot.png'
with mlflow.start_run() as run:
    mlflow.log_figure(plt.gcf(), combined_plot_path)

plt.close()
