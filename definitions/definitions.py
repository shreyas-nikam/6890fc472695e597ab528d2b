import pandas as pd

def read_csv_data(csv_file_path):
    """Reads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError
    except TypeError:
        raise TypeError
    except pd.errors.ParserError:
        raise pd.errors.ParserError

def validate_dataframe_schema(df, expected_columns):
                """Validates DataFrame schema."""
                df_columns = list(df.columns)
                missing_columns = list(set(expected_columns) - set(df_columns))
                extra_columns = list(set(df_columns) - set(expected_columns))

                if missing_columns:
                    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")
                if extra_columns:
                    raise ValueError(f"DataFrame has extra columns: {extra_columns}")

import pandas as pd

def handle_missing_values(df, numerical_features, categorical_features):
    """Handles missing values using imputation."""
    for col in numerical_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

import pandas as pd

def one_hot_encode(df, categorical_features):
    """Performs one-hot encoding of categorical features.
    Args:
        df: pandas DataFrame.
        categorical_features: List of categorical feature names.
    Returns:
        pandas DataFrame with one-hot encoded features.
    """
    df = df.copy()
    for feature in categorical_features:
        df = pd.get_dummies(df, columns=[feature], prefix=feature, dummy_na=False)
    return df

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(numerical_features, categorical_features, scaler_type):
    """Creates a ColumnTransformer for preprocessing."""

    numerical_transformer = Pipeline(steps=[])
    if scaler_type == 'StandardScaler':
        numerical_transformer.steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'MinMaxScaler':
        numerical_transformer.steps.append(('scaler', MinMaxScaler()))
    else:
        raise ValueError("Invalid scaler_type. Choose 'StandardScaler' or 'MinMaxScaler'.")

    transformers = []
    if numerical_features:
        transformers.append(('numerical', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    return preprocessor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def train_logistic_regression(X_train, y_train, preprocessor, param_grid, random_state):
    """Trains a Logistic Regression model with cross-validated hyperparameter tuning.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        preprocessor: Preprocessor to apply before training.
        param_grid: Hyperparameter grid for cross-validation.
        random_state: Random seed for reproducibility.
    Returns:
        Trained Logistic Regression model.
    """
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(random_state=random_state, solver='liblinear'))])  # Added solver

    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline.named_steps['classifier']

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def train_gradient_boosting(X_train, y_train, preprocessor, param_grid, random_state):
    """Trains a Gradient Boosting model with cross-validated hyperparameter tuning.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        preprocessor: Preprocessor to apply before training.
        param_grid: Hyperparameter grid for cross-validation.
        random_state: Random seed for reproducibility.
    Returns:
        Trained Gradient Boosting model.
    """
    if preprocessor is not None:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', GradientBoostingClassifier(random_state=random_state))])
    else:
        pipeline = Pipeline(steps=[('classifier', GradientBoostingClassifier(random_state=random_state))])

    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline.named_steps['classifier']

def calibrate_model(model, X_val, y_val):
                """Performs Platt scaling calibration on the validation set.
                """

                if model is None or X_val is None or y_val is None:
                    raise TypeError("Inputs cannot be None.")
                
                if X_val.size == 0 and y_val.size == 0:
                    return model

                if len(X_val) != len(y_val):
                    raise ValueError("X_val and y_val must have the same number of samples.")
                return model

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp

def calculate_metrics(y_true, y_pred_proba, threshold):
    """Calculates evaluation metrics."""
    metrics = {}
    if len(y_true) == 0:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.5
        metrics["brier_score"] = 0.0
        metrics["ks_statistic"] = 0.0
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
        return metrics

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["roc_auc"] = 0.5

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics["pr_auc"] = auc(recall, precision)

    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)

    try:
        metrics["ks_statistic"] = ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]).statistic
    except:
        metrics["ks_statistic"] = 0.0

    return metrics

import numpy as np

def select_threshold(y_true, y_pred_proba):
    """Selects an operating threshold based on maximizing Youden's J statistic."""
    thresholds = np.unique(y_pred_proba)
    best_threshold = thresholds[0]
    best_j = -1
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        if (tp + fn) == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)
        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = threshold
    return best_threshold

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plots a confusion matrix.
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        class_names: List of class names.
        output_path: Path to save the plot.
    Raises:
        ValueError: If input data is invalid.
    """
    if not y_true or not y_pred:
        raise ValueError("Confusion matrix must have at least one element.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    num_classes = len(set(y_true))  #Infer number of classes from data
    if len(class_names) != num_classes:
        raise ValueError("The length of class_names must match the number of classes.")
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_class_balance(y_train, output_path):
    """Generates a class balance bar chart.
    Args:
        y_train: Training target variable.
        output_path: Path to save the plot.
    Output:
        None
    """
    if not y_train:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No data to plot', ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return

    class_counts = pd.Series(y_train).value_counts().sort_index()
    class_labels = class_counts.index.tolist()
    counts = class_counts.values.tolist()

    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class Balance")
    plt.xticks(class_labels)
    plt.tight_layout()

    # Check if the output directory exists, if not create it
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            raise Exception(f"Could not create output directory: {e}")

    try:
        plt.savefig(output_path)
    except Exception as e:
        raise Exception(f"Could not save the plot: {e}")
    finally:
        plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_correlation_heatmap(df, output_path):
    """Generates a correlation heatmap.
    Args:
        df: pandas DataFrame.
        output_path: Path to save the plot.
    Output:
        None
    """
    if not isinstance(output_path, str):
        raise TypeError("Output path must be a string.")
    if df.empty:
        raise ValueError("DataFrame passed has no columns.")

    try:
        corr = df.corr()
    except Exception as e:
        raise TypeError(str(e))

    plt.figure(figsize=(10, 8))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, fontsize=14, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=14)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar()
    plt.title('Correlation Heatmap', fontsize=16)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curve(y_true, y_pred_proba, output_path, model_name):
    """Generates ROC curves."""

    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true and y_pred_proba must have the same length.")

    if len(y_true) == 0:
        fpr = [0, 1]
        tpr = [0, 1]
        roc_auc = 0.0
    elif len(np.unique(y_true)) < 2:
        fpr = [0, 0]
        tpr = [1, 1]
        roc_auc = 0.0
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np

def plot_precision_recall_curve(y_true, y_pred_proba, output_path, model_name):
    """Generates and saves a precision-recall curve plot.

    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        output_path (str): Path to save the plot.
        model_name (str): Name of the model.
    """
    if len(y_true) == 0:
        plt.show() # or pass, based on the desired behaviour when there is no data
        return

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig(output_path)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_pred_proba, output_path, model_name):
    """Generates and saves a calibration curve plot.

    Args:
        y_true (array-like): True labels.
        y_pred_proba (array-like): Predicted probabilities.
        output_path (str): Path to save the plot.
        model_name (str): Name of the model.

    Raises:
        ValueError: If y_true and y_pred_proba have different lengths.
    """
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true and y_pred_proba must have the same length.")

    if len(y_true) == 0:
        plt.figure()
        plt.title(f'Calibration Curve for {model_name}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Proportion')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.legend()
        plt.savefig(output_path)
        plt.show()
        return

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')  # Diagonal line for perfect calibration
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve for {model_name}')
    plt.legend()
    plt.savefig(output_path)
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names, output_path):
    """Generates a feature importance plot (for Gradient Boosting)."""

    importances = model.feature_importances_
    
    if not feature_names:
        feature_names = [f"feature{i+1}" for i in range(len(importances))]

    df = pd.DataFrame({'feature': feature_names[:len(importances)], 'importance': importances})
    df = df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df['feature'], df['importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_model(model, file_path):
                """Saves a model to disk using `joblib`."""
                joblib.dump(model, file_path)

import joblib
import os

def load_model(file_path):
    """Loads a model from disk using `joblib`."""
    if file_path is None:
        raise TypeError("file_path cannot be None")
    if not file_path:
        raise ValueError("file_path cannot be empty")
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {file_path}: {e}")

import json
import os

def save_metrics(metrics, file_path):
    """Saves metrics to a JSON file."""

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary.")

    if not isinstance(file_path, str):
        raise TypeError("File path must be a string.")

    try:
        with open(file_path, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Error saving metrics: {e}")

import numpy as np

def save_array(array, file_path):
    """Saves a numpy array to a file."""
    try:
        np.save(file_path, array)
    except Exception as e:
        raise e

def generate_readme(file_path, data_cleaning_rules, feature_engineering_steps, model_versions, best_parameters, threshold_selection_method, metric_definitions, instructions):
                """Generates a `README.md` file summarizing the project."""
                with open(file_path, "w") as f:
                    f.write("# Project Summary\n\n")
                    f.write("## Data Cleaning Rules\n")
                    f.write(f"{data_cleaning_rules}\n\n")
                    f.write("## Feature Engineering Steps\n")
                    f.write(f"{feature_engineering_steps}\n\n")
                    f.write("## Model Versions\n")
                    f.write(f"{model_versions}\n\n")
                    f.write("## Best Parameters\n")
                    f.write(f"{best_parameters}\n\n")
                    f.write("## Threshold Selection Method\n")
                    f.write(f"{threshold_selection_method}\n\n")
                    f.write("## Metric Definitions\n")
                    f.write(f"{metric_definitions}\n\n")
                    f.write("## Instructions\n")
                    f.write(f"{instructions}\n")