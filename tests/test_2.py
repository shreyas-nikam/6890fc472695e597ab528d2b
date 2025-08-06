import pytest
import pandas as pd
from definition_85de5237befb430f9817e2eb49b1f9a7 import handle_missing_values

@pytest.fixture
def sample_dataframe():
    data = {'numerical_col': [1, 2, None, 4, 5],
            'categorical_col': ['A', 'B', 'A', None, 'B'],
            'other_col': [6, 7, 8, 9, 10]}
    return pd.DataFrame(data)

def test_handle_missing_values_numerical(sample_dataframe):
    numerical_features = ['numerical_col']
    categorical_features = []
    df = handle_missing_values(sample_dataframe.copy(), numerical_features, categorical_features)
    assert df['numerical_col'].isnull().sum() == 0
    assert df['numerical_col'].median() == df['numerical_col'].iloc[2]

def test_handle_missing_values_categorical(sample_dataframe):
    numerical_features = []
    categorical_features = ['categorical_col']
    df = handle_missing_values(sample_dataframe.copy(), numerical_features, categorical_features)
    assert df['categorical_col'].isnull().sum() == 0
    assert df['categorical_col'].mode()[0] == df['categorical_col'].iloc[3]

def test_handle_missing_values_both(sample_dataframe):
    numerical_features = ['numerical_col']
    categorical_features = ['categorical_col']
    df = handle_missing_values(sample_dataframe.copy(), numerical_features, categorical_features)
    assert df['numerical_col'].isnull().sum() == 0
    assert df['categorical_col'].isnull().sum() == 0

def test_handle_missing_values_no_missing(sample_dataframe):
    df = sample_dataframe.dropna()
    numerical_features = ['numerical_col']
    categorical_features = ['categorical_col']
    df_processed = handle_missing_values(df.copy(), numerical_features, categorical_features)
    pd.testing.assert_frame_equal(df, df_processed)

def test_handle_missing_values_empty_features(sample_dataframe):
    numerical_features = []
    categorical_features = []
    df_processed = handle_missing_values(sample_dataframe.copy(), numerical_features, categorical_features)
    pd.testing.assert_frame_equal(sample_dataframe, df_processed)
