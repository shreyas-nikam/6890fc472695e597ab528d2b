import pytest
import pandas as pd
from definition_427849387b8b4092a7f6803de8c5b2c5 import one_hot_encode

def test_one_hot_encode_empty_list():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    categorical_features = []
    result = one_hot_encode(df, categorical_features)
    assert result.equals(df)

def test_one_hot_encode_single_feature():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    categorical_features = ['B']
    result = one_hot_encode(df.copy(), categorical_features) # avoid modifying original df
    assert 'B_a' in result.columns
    assert 'B_b' in result.columns
    assert 'B_c' in result.columns
    assert 'B' not in result.columns

def test_one_hot_encode_multiple_features():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [True, False, True]})
    categorical_features = ['B', 'C']
    result = one_hot_encode(df.copy(), categorical_features) # avoid modifying original df
    assert 'B_a' in result.columns
    assert 'B_b' in result.columns
    assert 'B_c' in result.columns
    assert 'C_True' in result.columns
    assert 'C_False' in result.columns
    assert 'B' not in result.columns
    assert 'C' not in result.columns

def test_one_hot_encode_numeric_categorical():
     df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
     categorical_features = ['B']
     result = one_hot_encode(df.copy(), categorical_features) # avoid modifying original df
     assert 'B_10' in result.columns
     assert 'B_20' in result.columns
     assert 'B_30' in result.columns
     assert 'B' not in result.columns

def test_one_hot_encode_mixed_dtypes():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [1.1, 2.2, 3.3]})
    categorical_features = ['B']
    result = one_hot_encode(df.copy(), categorical_features) # avoid modifying original df
    assert 'B_a' in result.columns
    assert 'C' in result.columns  # Ensure non-categorical features are preserved

