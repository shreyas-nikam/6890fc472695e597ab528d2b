import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
from definition_5b7cc62f5bf849f686eb5c5ce3ddef07 import calculate_metrics

def dummy_data():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.7])
    threshold = 0.5
    return y_true, y_pred_proba, threshold


def test_calculate_metrics_basic():
    y_true, y_pred_proba, threshold = dummy_data()
    metrics = calculate_metrics(y_true, y_pred_proba, threshold)
    assert isinstance(metrics, dict)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "brier_score" in metrics
    assert "ks_statistic" in metrics


def test_calculate_metrics_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = 0.5
    metrics = calculate_metrics(y_true, y_pred_proba, threshold)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_calculate_metrics_worst_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.9, 0.8, 0.2, 0.1])
    threshold = 0.5
    metrics = calculate_metrics(y_true, y_pred_proba, threshold)
    assert metrics["roc_auc"] == 0.0

def test_calculate_metrics_empty_input():
    y_true = np.array([])
    y_pred_proba = np.array([])
    threshold = 0.5
    metrics = calculate_metrics(y_true, y_pred_proba, threshold)
    assert isinstance(metrics, dict)
    assert metrics["roc_auc"] == 0.5
    assert metrics["pr_auc"] == 0.5
    assert metrics["brier_score"] == 0.0
    assert metrics["ks_statistic"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_calculate_metrics_all_same_class():
    y_true = np.array([1, 1, 1, 1])
    y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])
    threshold = 0.5
    metrics = calculate_metrics(y_true, y_pred_proba, threshold)
    assert metrics["roc_auc"] == 0.5

