import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from io import BytesIO
from definition_3911b2cdd851493480f6acd8ac9c5b78 import plot_precision_recall_curve
import numpy as np


@patch('matplotlib.pyplot.savefig')
def test_plot_precision_recall_curve_valid_data(mock_savefig):
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])
    output_path = "test_pr_curve.png"
    model_name = "TestModel"
    
    plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name)
    
    mock_savefig.assert_called_once_with(output_path)

@patch('matplotlib.pyplot.show')
def test_plot_precision_recall_curve_empty_data(mock_show):
    y_true = np.array([])
    y_pred_proba = np.array([])
    output_path = "test_pr_curve_empty.png"
    model_name = "TestModelEmpty"

    plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name)
    #Expect the function to execute, but not save anything because no plot can be created
    mock_show.assert_not_called()


@patch('matplotlib.pyplot.savefig')
def test_plot_precision_recall_curve_identical_predictions(mock_savefig):
    y_true = np.array([0, 1])
    y_pred_proba = np.array([0.5, 0.5])
    output_path = "test_pr_curve_identical.png"
    model_name = "TestModelIdentical"

    plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name)

    mock_savefig.assert_called_once_with(output_path)



@patch('matplotlib.pyplot.savefig')
def test_plot_precision_recall_curve_binary_predictions(mock_savefig):
    y_true = np.array([0, 1])
    y_pred_proba = np.array([0, 1])
    output_path = "test_pr_curve_binary.png"
    model_name = "TestModelBinary"

    plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name)

    mock_savefig.assert_called_once_with(output_path)

@patch('matplotlib.pyplot.savefig')
def test_plot_precision_recall_curve_mixed_data(mock_savefig):
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.6, 0.35, 0.9, 0.2])
        output_path = "test_pr_curve_mixed.png"
        model_name = "TestModelMixed"

        plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name)

        mock_savefig.assert_called_once_with(output_path)
