import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed

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
    def __init__(self, input_dim, hidden_dim=10, output_dim=1, lr=0.01, epochs=10, hidden_layers=1, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = Perceptron(input_dim, hidden_dim, output_dim, hidden_layers).to(self.device)

    def fit(self, X, y):
        self.model = Perceptron(self.input_dim, self.hidden_dim, self.output_dim, self.hidden_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0001)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def nn_run(cfg, X_train, X_val, X_test, y_train, y_val, y_test):
    # Define the parameter grid
    param_grid = dict(cfg.models.perceptron)

    # Perform GridSearchCV
    pytorch_regressor = PyTorchRegressor(input_dim=X_train.shape[1], output_dim=1)
    grid_search = GridSearchCV(estimator=pytorch_regressor, param_grid=param_grid, cv=3, scoring='r2', verbose=3)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    # Evaluate on the test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test set score: {test_score}")
