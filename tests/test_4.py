import pytest
import numpy as np
from definition_86974a263e58447e920b90c7175abb82 import split_data
from sklearn.model_selection import train_test_split


def test_split_data_valid_split():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    train_size, val_size, test_size = 0.6, 0.2, 0.2
    random_state = 42
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size+val_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=val_size/(val_size+test_size), random_state=random_state, stratify=y_test)
    
    np.testing.assert_equal(X_train.shape[0], 3)
    np.testing.assert_equal(X_val.shape[0], 1)
    np.testing.assert_equal(X_test.shape[0], 1)
    

def test_split_data_empty_input():
    X = np.array([])
    y = np.array([])
    train_size, val_size, test_size = 0.6, 0.2, 0.2
    random_state = 42
    with pytest.raises(ValueError):
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size+val_size, random_state=random_state)

def test_split_data_invalid_sizes():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    train_size, val_size, test_size = 0.8, 0.1, 0.2  # Sum > 1
    random_state = 42
    with pytest.raises(ValueError):
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size+val_size, random_state=random_state)

def test_split_data_different_lengths():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1]) # Different Length
    train_size, val_size, test_size = 0.6, 0.2, 0.2
    random_state = 42
    with pytest.raises(ValueError):
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size+val_size, random_state=random_state)

def test_split_data_stratification():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    train_size, val_size, test_size = 0.6, 0.2, 0.2
    random_state = 42
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size+val_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=val_size/(val_size+test_size), random_state=random_state, stratify=y_test)

    train_zeros = np.sum(y_train == 0)
    train_ones = np.sum(y_train == 1)
    assert abs(train_zeros/len(y_train) - 3/5) < 0.000001
    assert abs(train_ones/len(y_train) - 2/5) < 0.000001

