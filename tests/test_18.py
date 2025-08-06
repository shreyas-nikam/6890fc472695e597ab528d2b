import pytest
from definition_94a38af3ebbc43939d0fd8b41635aa89 import save_model
import joblib
import os
from unittest.mock import MagicMock

@pytest.fixture
def mock_model():
    return MagicMock()

@pytest.fixture
def temp_file_path(tmpdir):
    return os.path.join(tmpdir, "model.joblib")

def test_save_model_success(mock_model, temp_file_path):
    save_model(mock_model, temp_file_path)
    assert os.path.exists(temp_file_path)

def test_save_model_valid_data(mock_model, temp_file_path):
    save_model(mock_model, temp_file_path)
    loaded_model = joblib.load(temp_file_path)
    assert loaded_model == mock_model

def test_save_model_file_path_none(mock_model):
    with pytest.raises(TypeError):
        save_model(mock_model, None)

def test_save_model_model_none(temp_file_path):
    with pytest.raises(TypeError):
        save_model(None, temp_file_path)
