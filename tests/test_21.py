import pytest
import numpy as np
from definition_0240eac0b57a424c95f1dec1068c079b import save_array
import os

@pytest.fixture
def cleanup_files():
    file_paths = ["test_array.npy", "test_array_2.npy"]
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    yield
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_save_array_valid_input(cleanup_files):
    array = np.array([1, 2, 3])
    file_path = "test_array.npy"
    save_array(array, file_path)
    assert os.path.exists(file_path)
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(array, loaded_array)

def test_save_array_empty_array(cleanup_files):
    array = np.array([])
    file_path = "test_array.npy"
    save_array(array, file_path)
    assert os.path.exists(file_path)
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(array, loaded_array)

def test_save_array_overwrite_existing_file(cleanup_files):
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    file_path = "test_array_2.npy"
    np.save(file_path, array1) # Create an existing file
    save_array(array2, file_path)
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(array2, loaded_array)

def test_save_array_large_array(cleanup_files):
    array = np.random.rand(1000, 1000)
    file_path = "test_array.npy"
    save_array(array, file_path)
    assert os.path.exists(file_path)
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(array, loaded_array)

def test_save_array_non_numpy_array(cleanup_files):
    array = [1, 2, 3]
    file_path = "test_array.npy"
    with pytest.raises(Exception): # or specific exception, if known and applicable
        save_array(array, file_path)
