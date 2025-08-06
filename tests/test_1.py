import pytest
import pandas as pd
from definition_3fc8821e39d44e3f9dd89cf0d79d5ccb import validate_dataframe_schema

def test_validate_dataframe_schema_valid():
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    expected_columns = ['col1', 'col2']
    validate_dataframe_schema(df, expected_columns)  # Should not raise an exception

def test_validate_dataframe_schema_missing_column():
    df = pd.DataFrame({'col1': [1, 2]})
    expected_columns = ['col1', 'col2']
    with pytest.raises(ValueError) as excinfo:
        validate_dataframe_schema(df, expected_columns)
    assert "Missing columns in DataFrame: ['col2']" in str(excinfo.value)

def test_validate_dataframe_schema_extra_column():
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b'], 'col3': [3,4]})
    expected_columns = ['col1', 'col2']
    with pytest.raises(ValueError) as excinfo:
        validate_dataframe_schema(df, expected_columns)
    assert "DataFrame has extra columns: ['col3']" in str(excinfo.value)

def test_validate_dataframe_schema_empty_dataframe():
    df = pd.DataFrame()
    expected_columns = ['col1', 'col2']
    with pytest.raises(ValueError) as excinfo:
        validate_dataframe_schema(df, expected_columns)
    assert "Missing columns in DataFrame: ['col1', 'col2']" in str(excinfo.value)

def test_validate_dataframe_schema_empty_expected_columns():
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    expected_columns = []
    validate_dataframe_schema(df, expected_columns) # Should not raise an error
