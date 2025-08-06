import pytest
import pandas as pd
import matplotlib.pyplot as plt
from definition_e431379aae3346bb9506d35e40a92ec4 import plot_correlation_heatmap

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [5, 4, 3, 2, 1],
            'col3': [1, 1, 1, 1, 1]}
    return pd.DataFrame(data)

def test_plot_correlation_heatmap_success(sample_dataframe, tmp_path):
    output_path = tmp_path / "test_heatmap.png"
    plot_correlation_heatmap(sample_dataframe, str(output_path))
    assert output_path.exists()

def test_plot_correlation_heatmap_empty_dataframe(tmp_path):
    output_path = tmp_path / "empty_heatmap.png"
    df = pd.DataFrame()
    try:
        plot_correlation_heatmap(df, str(output_path))
    except Exception as e:
        assert "DataFrame passed has no columns" in str(e)

def test_plot_correlation_heatmap_non_numeric_data(tmp_path):
    output_path = tmp_path / "non_numeric_heatmap.png"
    df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['d', 'e', 'f']})
    try:
        plot_correlation_heatmap(df, str(output_path))
    except Exception as e:
        assert "must be numeric or convert to numeric" in str(e)

def test_plot_correlation_heatmap_single_column(tmp_path):
    output_path = tmp_path / "single_column_heatmap.png"
    df = pd.DataFrame({'col1': [1, 2, 3]})
    plot_correlation_heatmap(df, str(output_path))
    assert output_path.exists()

def test_plot_correlation_heatmap_invalid_output_path(sample_dataframe):
    try:
        plot_correlation_heatmap(sample_dataframe, 123)
    except Exception as e:
        assert "path must be a string" in str(e)
