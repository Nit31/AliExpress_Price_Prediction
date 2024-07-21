import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed
import random
import numpy as np
from mlflow.models import infer_signature


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Some other options if you are using cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Create a simple perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super(Perceptron, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.fc.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.fc.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        x = self.fc[-1](x)
        return x


# Custom regressor to use with GridSearchCV
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim=10, output_dim=1, lr=0.01, epochs=10, hidden_layers=1, batch_size=128, optimizer='Adam'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.optimizer = optimizer  # Store optimizer name as a string
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self._initialize_model()

    def _initialize_model(self):
        self.model = Perceptron(self.input_dim, self.hidden_dim, self.output_dim, self.hidden_layers).to(self.device)
        self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer == 'Adam':
            self.optimizer_ = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer_ = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        elif self.optimizer == 'RMSprop':
            self.optimizer_ = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-8)
        else:
            self.optimizer_ = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        self._initialize_model()
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer_.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                self.optimizer_.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)



def train_and_evaluate_model(params, X_train, y_train):
    pytorch_regressor = PyTorchRegressor(input_dim=X_train.shape[1], output_dim=1)
    model = pytorch_regressor.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    model.fit(X_train, y_train)
    return model, np.mean(scores)


def nn_run(cfg,model_architecture, X_train, X_test, y_train, y_test):
    if model_architecture == 1:
        set_seed(cfg.models.model1.random_state)
        param_grid = dict(cfg.models.model1.params)
    else:
        set_seed(cfg.models.model2.random_state)
        param_grid = dict(cfg.models.model2.params)


    experiment_name = f"Model_dim{param_grid['hidden_dim']}_layers{param_grid['hidden_layers']}_experiment"
    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    for i, params in enumerate(ParameterGrid(param_grid)):
        model, mean_score = train_and_evaluate_model(params, X_train, y_train)
        run_name = f'r2_score{mean_score}_run{i}'
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"r2": mean_score})
            mlflow.set_tag("Training Info", f"Fully-connected model architecture for aliexpress using")
            # Infer the model signature
            signature = infer_signature(X_train, y_train)
            # Log the model
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path=f'models/{experiment_name}/{run_name}',
                signature=signature,
                input_example=X_train,  # Use X_train or X_test as needed
                registered_model_name=f'{experiment_name}_{run_name}'
            )
            #
            # pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
            # predictions = pytorch_pyfunc.predict(X_test)
            # eval_data["predictions"] = net(X).detach().numpy()
            # print(eval_data.shape)
            # results = mlflow.evaluate(
            #     data=eval_data,
            #     model_type="regressor",
            #     targets= "label",
            #     predictions="predictions",
            #     evaluators = ["default"]
            # )
            # predictions = sk_pyfunc.predict(X_test)
            # print(predictions)
            # eval_data = pd.DataFrame(y_test)
            # eval_data.columns = ["label"]
            # eval_data["predictions"] = predictions
            #
            # results = mlflow.evaluate(
            #     data=eval_data,
            #     model_type="classifier",
            #     targets= "label",
            #     predictions="predictions",
            #     evaluators = ["default"]
            # )
