import pytest
import matplotlib.pyplot as plt
import numpy as np
from definition_3f733af56bf948bcb9122a8edeaf8362 import plot_calibration_curve
from unittest.mock import patch


@pytest.fixture
def mock_plt_show():
    with patch("matplotlib.pyplot.show") as mock_show:
        yield mock_show


def test_plot_calibration_curve_basic(mock_plt_show):
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    output_path = "test_calibration_curve.png"
    model_name = "Test Model"

    plot_calibration_curve(y_true, y_pred_proba, output_path, model_name)
    mock_plt_show.assert_called_once()


def test_plot_calibration_curve_empty_data(mock_plt_show):
    y_true = np.array([])
    y_pred_proba = np.array([])
    output_path = "test_calibration_curve_empty.png"
    model_name = "Empty Model"

    plot_calibration_curve(y_true, y_pred_proba, output_path, model_name)
    mock_plt_show.assert_called_once()


def test_plot_calibration_curve_perfect_calibration(mock_plt_show):
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    y_pred_proba = y_pred_proba
    output_path = "test_calibration_curve_perfect.png"
    model_name = "Perfect Model"

    plot_calibration_curve(y_true, y_pred_proba, output_path, model_name)
    mock_plt_show.assert_called_once()

def test_plot_calibration_curve_uncalibrated(mock_plt_show):
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_proba = np.array([0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7])
    output_path = "test_calibration_curve_uncalibrated.png"
    model_name = "Uncalibrated Model"

    plot_calibration_curve(y_true, y_pred_proba, output_path, model_name)
    mock_plt_show.assert_called_once()

def test_plot_calibration_curve_mismatched_lengths(mock_plt_show):
    y_true = np.array([0, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    output_path = "test_calibration_curve_mismatch.png"
    model_name = "Mismatched Model"
    with pytest.raises(ValueError):
        plot_calibration_curve(y_true, y_pred_proba, output_path, model_name)

