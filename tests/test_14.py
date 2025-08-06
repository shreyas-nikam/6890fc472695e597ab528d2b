import pytest
import matplotlib.pyplot as plt
import numpy as np
from definition_8d1146ec61744fedb5f73b2ac5473de5 import plot_roc_curve
from unittest.mock import patch


@pytest.fixture
def mock_plt_show():
    with patch("matplotlib.pyplot.show") as mock:
        yield mock

def test_plot_roc_curve_valid_input(tmp_path, mock_plt_show):
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])
    output_path = str(tmp_path / "test_roc.png")
    model_name = "TestModel"
    
    plot_roc_curve(y_true, y_pred_proba, output_path, model_name)
    
    assert (tmp_path / "test_roc.png").exists()
    mock_plt_show.assert_called_once()


def test_plot_roc_curve_empty_input(tmp_path, mock_plt_show):
    y_true = np.array([])
    y_pred_proba = np.array([])
    output_path = str(tmp_path / "empty_roc.png")
    model_name = "EmptyModel"
    
    plot_roc_curve(y_true, y_pred_proba, output_path, model_name)
    
    assert (tmp_path / "empty_roc.png").exists()
    mock_plt_show.assert_called_once()

def test_plot_roc_curve_all_same_class(tmp_path, mock_plt_show):
    y_true = np.array([1, 1, 1, 1])
    y_pred_proba = np.array([0.2, 0.3, 0.7, 0.9])
    output_path = str(tmp_path / "same_class_roc.png")
    model_name = "SameClassModel"

    plot_roc_curve(y_true, y_pred_proba, output_path, model_name)

    assert (tmp_path / "same_class_roc.png").exists()
    mock_plt_show.assert_called_once()
    

def test_plot_roc_curve_invalid_proba(tmp_path, mock_plt_show):
    y_true = np.array([0, 1])
    y_pred_proba = np.array([1.2, -0.2]) 
    output_path = str(tmp_path / "invalid_proba_roc.png")
    model_name = "InvalidProbaModel"
    
    with pytest.raises(ValueError) as excinfo:
        plot_roc_curve(y_true, y_pred_proba, output_path, model_name)

def test_plot_roc_curve_different_lengths(tmp_path, mock_plt_show):
    y_true = np.array([0, 1, 0])
    y_pred_proba = np.array([0.1, 0.9])
    output_path = str(tmp_path / "diff_len_roc.png")
    model_name = "DiffLenModel"
    
    with pytest.raises(ValueError) as excinfo:
        plot_roc_curve(y_true, y_pred_proba, output_path, model_name)
