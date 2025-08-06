import pytest
import pandas as pd
from definition_f412255c90ca4d10ae4e2eb235674a75 import read_csv_data

def test_read_csv_data_success(tmp_path):
    # Create a dummy CSV file
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    # Read the CSV file using the function
    result_df = read_csv_data(file_path)

    # Assert that the result is a pandas DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Assert that the DataFrame has the correct data
    assert result_df.equals(df)

def test_read_csv_data_file_not_found():
    # Provide a non-existent file path
    with pytest.raises(FileNotFoundError):
        read_csv_data("non_existent_file.csv")

def test_read_csv_data_empty_file(tmp_path):
    # Create an empty CSV file
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    # Read the CSV file using the function
    result_df = read_csv_data(file_path)

    # Assert that the result is a pandas DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Assert that the DataFrame is empty
    assert result_df.empty

def test_read_csv_data_different_delimiters(tmp_path):
    # Create a dummy CSV file with semicolon delimiter
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    file_path = tmp_path / "test_semicolon.csv"
    df.to_csv(file_path, index=False, sep=';')

    # try to read the semicolon seperated csv without providing sep argument. 
    # Expect error from pandas read_csv, which should propagate. 
    with pytest.raises(pd.errors.ParserError):
        read_csv_data(file_path)

def test_read_csv_data_invalid_file_path():
    #Provide integer as file path
    with pytest.raises(TypeError):
        read_csv_data(123)
