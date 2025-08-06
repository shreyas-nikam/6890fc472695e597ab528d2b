import pytest
from unittest.mock import MagicMock
from definition_1280b26a9fe14a9491dc7b5c69b62508 import plot_feature_importance
import matplotlib.pyplot as plt
import pandas as pd

def test_plot_feature_importance_valid_data(tmp_path):
    # Mock a Gradient Boosting model
    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.4, 0.3, 0.2, 0.1]

    # Mock feature names
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

    # Define the output path
    output_path = str(tmp_path / "feature_importance.png")

    # Call the function
    plot_feature_importance(mock_model, feature_names, output_path)

    # Assert that the plot was saved (basic check - existence of file)
    assert (tmp_path / "feature_importance.png").exists()

def test_plot_feature_importance_empty_feature_names(tmp_path):
    # Mock a Gradient Boosting model
    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.4, 0.3, 0.2, 0.1]

    # Empty feature names
    feature_names = []

    # Define the output path
    output_path = str(tmp_path / "feature_importance.png")

    # Call the function - should not raise
    plot_feature_importance(mock_model, feature_names, output_path)

    # Assert that the plot was saved (basic check - existence of file)
    assert (tmp_path / "feature_importance.png").exists()

def test_plot_feature_importance_mismatched_lengths(tmp_path):
    # Mock a Gradient Boosting model
    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.4, 0.3, 0.2]

    # Mock feature names with a different length
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

    # Define the output path
    output_path = str(tmp_path / "feature_importance.png")

    # Call the function - should not raise, but plot will be incomplete if the function is not modified to handle this case.
    plot_feature_importance(mock_model, feature_names, output_path)

    # Assert that the plot was saved (basic check - existence of file)
    assert (tmp_path / "feature_importance.png").exists()

def test_plot_feature_importance_zero_importance(tmp_path):
     # Mock a Gradient Boosting model with zero importances
    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.0, 0.0, 0.0, 0.0]

    # Mock feature names
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

    # Define the output path
    output_path = str(tmp_path / "feature_importance.png")

    # Call the function
    plot_feature_importance(mock_model, feature_names, output_path)

    # Assert that the plot was saved (basic check - existence of file)
    assert (tmp_path / "feature_importance.png").exists()

def test_plot_feature_importance_invalid_output_path(tmp_path):
    # Mock a Gradient Boosting model
    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.4, 0.3, 0.2, 0.1]

    # Mock feature names
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

    # Define an invalid output path (directory does not exist)
    output_path = str(tmp_path / "nonexistent_dir" / "feature_importance.png")

    # Call the function and expect a FileNotFoundError (or similar) if the directory doesn't exist
    with pytest.raises(FileNotFoundError):  # or OSError, depending on the OS
        plot_feature_importance(mock_model, feature_names, output_path)
