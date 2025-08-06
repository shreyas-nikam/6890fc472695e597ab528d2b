import pytest
import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import patch
from definition_dfff9616c4f6487283282b7fa02b892d import plot_confusion_matrix

@pytest.fixture
def dummy_data():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0]
    class_names = ['Class 0', 'Class 1']
    output_path = 'test_confusion_matrix.png'
    return y_true, y_pred, class_names, output_path

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_plot_confusion_matrix_nominal(mock_close, mock_show, mock_savefig, dummy_data):
    y_true, y_pred, class_names, output_path = dummy_data
    plot_confusion_matrix(y_true, y_pred, class_names, output_path)
    mock_savefig.assert_called_once_with(output_path)


def test_plot_confusion_matrix_empty_data():
    with pytest.raises(ValueError, match="Confusion matrix must have at least one element."):
        plot_confusion_matrix([], [], ['A', 'B'], 'test.png')

def test_plot_confusion_matrix_different_lengths():
        with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
            plot_confusion_matrix([1,2,3], [1,2], ['A','B'], 'test.png')

def test_plot_confusion_matrix_invalid_class_names(dummy_data):
    y_true, y_pred, _, output_path = dummy_data
    with pytest.raises(ValueError, match="The length of class_names must match the number of classes."):
        plot_confusion_matrix(y_true, y_pred, ['Class 0'], output_path)


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_plot_confusion_matrix_binary_classes(mock_close, mock_show, mock_savefig):
    y_true = [0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1]
    class_names = ['Negative', 'Positive']
    output_path = 'binary_confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, class_names, output_path)
    mock_savefig.assert_called_once_with(output_path)
