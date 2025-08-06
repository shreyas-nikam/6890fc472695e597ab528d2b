import pytest
import numpy as np
from definition_aa6ae3107e3747838cdc74c0d10734b6 import select_threshold

def test_select_threshold_typical_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.6, 0.8])
    threshold = select_threshold(y_true, y_pred_proba)
    assert threshold == 0.6

def test_select_threshold_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(y_true, y_pred_proba)
    assert threshold == 0.8

def test_select_threshold_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = select_threshold(y_true, y_pred_proba)
    assert threshold == 0.1

def test_select_threshold_all_same_proba():
    y_true = np.array([0, 1])
    y_pred_proba = np.array([0.5, 0.5])
    threshold = select_threshold(y_true, y_pred_proba)
    assert threshold == 0.5

def test_select_threshold_no_negatives():
    y_true = np.array([1, 1, 1, 1])
    y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])
    threshold = select_threshold(y_true, y_pred_proba)
    assert threshold == 0.6
