import pytest
from definition_14934215581d4428914fe89b114843d4 import create_preprocessing_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

@pytest.fixture
def sample_numerical_features():
    return ['age', 'income']

@pytest.fixture
def sample_categorical_features():
    return ['gender', 'education']

def test_create_preprocessing_pipeline_standard_scaler(sample_numerical_features, sample_categorical_features):
    pipeline = create_preprocessing_pipeline(sample_numerical_features, sample_categorical_features, 'StandardScaler')
    assert isinstance(pipeline, ColumnTransformer)

    # Check if numerical features are preprocessed with StandardScaler
    numerical_transformer = None
    for transformer_tuple in pipeline.transformers_:
        if transformer_tuple[0] == 'numerical':
            numerical_transformer = transformer_tuple[1]
            break

    assert isinstance(numerical_transformer, Pipeline)
    assert isinstance(numerical_transformer.steps[0][1], StandardScaler)
    assert transformer_tuple[2] == sample_numerical_features

def test_create_preprocessing_pipeline_minmax_scaler(sample_numerical_features, sample_categorical_features):
    pipeline = create_preprocessing_pipeline(sample_numerical_features, sample_categorical_features, 'MinMaxScaler')
    assert isinstance(pipeline, ColumnTransformer)
    # Check if numerical features are preprocessed with MinMaxScaler
    numerical_transformer = None
    for transformer_tuple in pipeline.transformers_:
        if transformer_tuple[0] == 'numerical':
            numerical_transformer = transformer_tuple[1]
            break
    assert isinstance(numerical_transformer, Pipeline)
    assert isinstance(numerical_transformer.steps[0][1], MinMaxScaler)
    assert transformer_tuple[2] == sample_numerical_features
def test_create_preprocessing_pipeline_no_numerical_features(sample_categorical_features):
    pipeline = create_preprocessing_pipeline([], sample_categorical_features, 'StandardScaler')
    assert isinstance(pipeline, ColumnTransformer)

    # Ensure only categorical preprocessing exists
    assert len(pipeline.transformers_) == 1
    assert pipeline.transformers_[0][0] == 'categorical'
    assert pipeline.transformers_[0][2] == sample_categorical_features

def test_create_preprocessing_pipeline_no_categorical_features(sample_numerical_features):
    pipeline = create_preprocessing_pipeline(sample_numerical_features, [], 'StandardScaler')
    assert isinstance(pipeline, ColumnTransformer)

    # Ensure only numerical preprocessing exists
    assert len(pipeline.transformers_) == 1
    assert pipeline.transformers_[0][0] == 'numerical'
    assert pipeline.transformers_[0][2] == sample_numerical_features

def test_create_preprocessing_pipeline_invalid_scaler_type(sample_numerical_features, sample_categorical_features):
    with pytest.raises(ValueError):
        create_preprocessing_pipeline(sample_numerical_features, sample_categorical_features, 'InvalidScaler')
