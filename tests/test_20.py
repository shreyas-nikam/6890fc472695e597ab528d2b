import pytest
from definition_46a6d40735fd4b75beff6578eb5ca27b import save_metrics
import json
import os

@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "temp_metrics.json"
    return str(file_path)

def test_save_metrics_success(temp_file):
    metrics = {"accuracy": 0.9, "precision": 0.8}
    save_metrics(metrics, temp_file)
    with open(temp_file, "r") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics

def test_save_metrics_empty(temp_file):
    metrics = {}
    save_metrics(metrics, temp_file)
    with open(temp_file, "r") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics

def test_save_metrics_overwrite(temp_file):
    metrics1 = {"accuracy": 0.9}
    save_metrics(metrics1, temp_file)
    metrics2 = {"precision": 0.8}
    save_metrics(metrics2, temp_file)
    with open(temp_file, "r") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics2

def test_save_metrics_non_dict(temp_file):
    with pytest.raises(TypeError):
        save_metrics("not a dict", temp_file)
    assert not os.path.exists(temp_file)

def test_save_metrics_invalid_file_path():
    with pytest.raises(TypeError):
        save_metrics({"accuracy": 0.9}, 123)