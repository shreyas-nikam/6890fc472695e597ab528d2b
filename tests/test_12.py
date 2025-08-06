import pytest
import matplotlib.pyplot as plt
import io
from definition_d11fbd246c514d3abcad1d037f055073 import plot_class_balance
import pandas as pd

def create_dummy_data(class_0_size, class_1_size):
    data = {'target': [0] * class_0_size + [1] * class_1_size}
    return pd.DataFrame(data)


def test_plot_class_balance_valid_data():
    # Test with valid data and check if the function runs without errors
    y_train = [0, 0, 0, 1, 1]
    output_path = "test_class_balance.png"
    try:
        plot_class_balance(y_train, output_path)
        plt.close()
    except Exception as e:
        assert False, f"Function raised an exception: {e}"

def test_plot_class_balance_empty_data():
    # Test with empty data and check for appropriate handling
    y_train = []
    output_path = "test_empty_class_balance.png"
    try:
        plot_class_balance(y_train, output_path)
        plt.close()
    except Exception as e:
        assert False, f"Function raised an exception: {e}"

def test_plot_class_balance_imbalanced_data():
    #Test with imbalanced data and ensure plot saves
    y_train = [0] * 90 + [1] * 10
    output_path = "test_imbalanced_class_balance.png"
    try:
        plot_class_balance(y_train, output_path)
        plt.close()
    except Exception as e:
        assert False, f"Function raised an exception: {e}"


def test_plot_class_balance_large_data():
     # Test with large dataset
    df = create_dummy_data(500, 500)
    output_path = "test_large_dataset.png"
    try:
        plot_class_balance(df['target'].tolist(), output_path)
        plt.close()
    except Exception as e:
        assert False, f"Function raised an exception: {e}"

def test_plot_class_balance_invalid_output_path():
    # Test with an invalid output path (e.g., no write permissions)
    y_train = [0, 1, 0, 1]
    output_path = "/invalid/path/class_balance.png" # Assuming this path doesn't exist or has no write permission

    with pytest.raises(Exception):
        plot_class_balance(y_train, output_path)
        plt.close()
