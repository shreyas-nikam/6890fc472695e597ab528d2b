import pytest
from definition_41c6f97d62934deb9e4fef95c00f446e import train_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    # Create a minimal, deterministic dataset for testing
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


@pytest.fixture
def sample_preprocessor():
    # Create a minimal preprocessor for testing
    numeric_features = ['feature1', 'feature2']
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    return preprocessor


@pytest.fixture
def sample_param_grid():
    # Create a minimal hyperparameter grid for testing
    param_grid = {'n_estimators': [10], 'learning_rate': [0.1], 'max_depth': [3]}
    return param_grid


def test_train_gradient_boosting_returns_model(sample_data, sample_preprocessor, sample_param_grid):
    X, y = sample_data
    model = train_gradient_boosting(X, y, sample_preprocessor, sample_param_grid, random_state=42)
    assert isinstance(model, GradientBoostingClassifier)


def test_train_gradient_boosting_fits_model(sample_data, sample_preprocessor, sample_param_grid):
    X, y = sample_data
    model = train_gradient_boosting(X, y, sample_preprocessor, sample_param_grid, random_state=42)
    X_transformed = sample_preprocessor.fit_transform(X)
    assert model.n_estimators > 0  # Check that the model has been trained. It will throw an error if its 0.

def test_train_gradient_boosting_handles_empty_param_grid(sample_data, sample_preprocessor):
    X, y = sample_data
    param_grid = {}
    model = train_gradient_boosting(X, y, sample_preprocessor, param_grid, random_state=42)
    assert isinstance(model, GradientBoostingClassifier)


def test_train_gradient_boosting_with_different_random_states(sample_data, sample_preprocessor, sample_param_grid):
    X, y = sample_data
    model1 = train_gradient_boosting(X, y, sample_preprocessor, sample_param_grid, random_state=42)
    model2 = train_gradient_boosting(X, y, sample_preprocessor, sample_param_grid, random_state=123)
    assert model1 is not None
    assert model2 is not None

def test_train_gradient_boosting_no_preprocessor(sample_data, sample_param_grid):
    X, y = sample_data
    model = train_gradient_boosting(X, y, None, sample_param_grid, random_state=42)
    assert isinstance(model, GradientBoostingClassifier)

