import hydra
import zenml
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def run_ridge_regression():
    # TODO: Write them into config
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky'],
        'max_iter': [100, 500, 1000]
    }
