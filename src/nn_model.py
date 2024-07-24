import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import random
import numpy as np


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
        self.fc.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.fc.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        x = self.fc[-1](x)
        return x


# Custom regressor to use with GridSearchCV
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim,
        hidden_dim=10,
        output_dim=1,
        lr=0.01,
        epochs=10,
        hidden_layers=1,
        batch_size=128,
        optimizer="Adam",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.optimizer = optimizer  # Store optimizer name as a string
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self):
        self.model = Perceptron(
            self.input_dim, self.hidden_dim, self.output_dim, self.hidden_layers
        ).to(self.device)
        self._initialize_optimizer()

    def _initialize_optimizer(self):
        if hasattr(optim, self.optimizer):
            optimizer_class = getattr(optim, self.optimizer)
            if self.optimizer == "SGD":
                self.optimizer_ = optimizer_class(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            elif self.optimizer == "RMSprop":
                self.optimizer_ = optimizer_class(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-8)
            else:
                self.optimizer_ = optimizer_class(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer} is not supported")

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

    def to(self, device):
        self.model.to(device)


def train_and_evaluate_model(params, X_train, y_train):
    pytorch_regressor = PyTorchRegressor(input_dim=X_train.shape[1], output_dim=1)
    model = pytorch_regressor.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
    model.fit(X_train, y_train)
    return model, np.mean(scores)
