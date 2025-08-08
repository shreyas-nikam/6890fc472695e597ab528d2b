
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Ensure artifacts directories exist
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

create_directory_if_not_exists('artifacts/models')
create_directory_if_not_exists('artifacts/metrics')
create_directory_if_not_exists('artifacts/plots')
create_directory_if_not_exists('artifacts/data')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

@st.cache_resource
def load_data_splits():
    try:
        # In a real application, you'd load X_cleaned and y from a shared state or file saved by Page1.
        # For this demonstration, we'll load the original CSV and re-apply cleaning and splitting.
        # This ensures page2 can run independently for testing, but in production, data flow should be optimized.
        df_full = pd.read_csv("UCI_Credit_Card.csv", encoding="utf-8")

        expected_columns = [
            'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
            'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
            'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
            'PAY_AMT6', 'default.payment.next.month'
        ]
        df_full = df_full[expected_columns] # Ensure only expected columns are kept
        df_full.drop_duplicates(inplace=True)
        
        y = df_full["default.payment.next.month"]
        X_cols = [col for col in df_full.columns if col not in ['ID', 'default.payment.next.month']]
        X = df_full[X_cols]

        # Deterministic cleaning
        X_cleaned = X.copy()
        X_cleaned['EDUCATION'] = X_cleaned['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
        X_cleaned['MARRIAGE'] = X_cleaned['MARRIAGE'].replace({0: 3})

        # Load indices
        train_indices = np.load('artifacts/data/train_indices.npy')
        val_indices = np.load('artifacts/data/val_indices.npy')

        X_train = X_cleaned.loc[train_indices]
        y_train = y.loc[train_indices]
        X_val = X_cleaned.loc[val_indices]
        y_val = y.loc[val_indices]

        numerical_features = [
            'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]
        ordinal_status_features = [
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
        ]
        categorical_raw_features = [
            'SEX', 'EDUCATION', 'MARRIAGE'
        ]

        return X_train, y_train, X_val, y_val, numerical_features, ordinal_status_features, categorical_raw_features
    except FileNotFoundError:
        st.error("Error: Data files (UCI_Credit_Card.csv or numpy index files) not found. Please ensure they are in the correct directory after running Page 1.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_resource
def create_and_fit_preprocessors(X_train, numerical_features, ordinal_status_features, categorical_raw_features):
    with st.spinner("Creating and fitting preprocessing pipelines..."):
        # Define all features that need one-hot encoding, combining categorical and ordinal status
        features_to_onehot_encode = categorical_raw_features # OneHotEncoder will handle integer-like categories correctly

        # Preprocessor for Logistic Regression (with StandardScaler for numerical features)
        preprocessor_LR = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_to_onehot_encode)
            ], remainder='passthrough')

        # Preprocessor for Gradient Boosting (no scaler on numerical, as tree models are scale-invariant)
        # Using an identity transformer or just passing through numerical features without scaling
        preprocessor_GB = ColumnTransformer(
            transformers=[
                # Gradient Boosting doesn't need scaling on numerical features, so we just pass them through
                ('num_passthrough', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_to_onehot_encode)
            ], remainder='passthrough') # 'passthrough' for any unlisted columns
        
        # Fit preprocessors on training data
        preprocessor_LR.fit(X_train)
        preprocessor_GB.fit(X_train)

        joblib.dump(preprocessor_LR, 'artifacts/models/preprocessor_LR.joblib')
        joblib.dump(preprocessor_GB, 'artifacts/models/preprocessor_GB.joblib')
        st.success("Preprocessing pipelines created, fitted, and saved.")
    return preprocessor_LR, preprocessor_GB

@st.cache_resource
def train_logistic_regression(X_train, y_train, preprocessor_LR):
    with st.spinner("Training Logistic Regression model..."):
        pipeline_lr = Pipeline(steps=[
            ('preprocessor', preprocessor_LR),
            ('classifier', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', class_weight='balanced', max_iter=1000))
        ])

        param_grid_lr = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0]
        }

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search_lr.fit(X_train, y_train)

        best_lr_model = grid_search_lr.best_estimator_
        best_lr_params = grid_search_lr.best_params_

        joblib.dump(best_lr_model, 'artifacts/models/pd_logreg_v1.pkl')
        with open('artifacts/models/best_params_logreg_v1.json', 'w') as f:
            json.dump(best_lr_params, f)

        st.success("Logistic Regression model trained and saved.")
        st.write(f"Best parameters for Logistic Regression: {best_lr_params}")
    return best_lr_model

@st.cache_resource
def train_gradient_boosting(X_train, y_train, preprocessor_GB):
    with st.spinner("Training Gradient Boosting model..."):
        pipeline_gb = Pipeline(steps=[
            ('preprocessor', preprocessor_GB),
            ('classifier', GradientBoostingClassifier(random_state=RANDOM_SEED))
        ])

        param_grid_gb = {
            'classifier__n_estimators': [200, 400],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [2, 3],
            'classifier__subsample': [1.0]
        }

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search_gb.fit(X_train, y_train)

        best_gb_model = grid_search_gb.best_estimator_
        best_gb_params = grid_search_gb.best_params_

        joblib.dump(best_gb_model, 'artifacts/models/pd_gb_v1.pkl')
        with open('artifacts/models/best_params_gb_v1.json', 'w') as f:
            json.dump(best_gb_params, f)

        st.success("Gradient Boosting model trained and saved.")
        st.write(f"Best parameters for Gradient Boosting: {best_gb_params}")
    return best_gb_model

@st.cache_resource
def calibrate_models(best_lr_model, best_gb_model, X_val, y_val):
    with st.spinner("Calibrating models on validation set..."):
        # Calibrate Logistic Regression
        calibrated_lr_model = CalibratedClassifierCV(best_lr_model, method='sigmoid', cv='prefit')
        calibrated_lr_model.fit(X_val, y_val)
        joblib.dump(calibrated_lr_model, 'artifacts/models/pd_logreg_calibrated_v1.pkl')
        st.success("Calibrated Logistic Regression model saved.")

        # Calibrate Gradient Boosting
        calibrated_gb_model = CalibratedClassifierCV(best_gb_model, method='sigmoid', cv='prefit')
        calibrated_gb_model.fit(X_val, y_val)
        joblib.dump(calibrated_gb_model, 'artifacts/models/pd_gb_calibrated_v1.pkl')
        st.success("Calibrated Gradient Boosting model saved.")
    return calibrated_lr_model, calibrated_gb_model

@st.cache_resource
def select_optimal_threshold(calibrated_lr_model, calibrated_gb_model, X_val, y_val):
    with st.spinner("Selecting optimal threshold on validation set..."):
        y_val_pred_proba_lr = calibrated_lr_model.predict_proba(X_val)[:, 1]
        y_val_pred_proba_gb = calibrated_gb_model.predict_proba(X_val)[:, 1]

        # Determine champion based on PR-AUC on validation set
        precision_lr, recall_lr, _ = precision_recall_curve(y_val, y_val_pred_proba_lr)
        pr_auc_lr = auc(recall_lr, precision_lr)

        precision_gb, recall_gb, _ = precision_recall_curve(y_val, y_val_pred_proba_gb)
        pr_auc_gb = auc(recall_gb, precision_gb)

        champion_model_name = ""
        if pr_auc_gb >= pr_auc_lr:
            champion_model = calibrated_gb_model
            champion_model_name = "Gradient Boosting"
            y_val_champion_proba = y_val_pred_proba_gb
        else:
            champion_model = calibrated_lr_model
            champion_model_name = "Logistic Regression"
            y_val_champion_proba = y_val_pred_proba_lr
        
        st.info(f"Champion model selected based on PR-AUC on validation set: **{champion_model_name}** (LR PR-AUC: {pr_auc_lr:.4f}, GB PR-AUC: {pr_auc_gb:.4f})")

        # Compute Youden's J for the champion model
        fpr, tpr, thresholds = roc_curve(y_val, y_val_champion_proba)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        tau_star = thresholds[best_threshold_idx]

        with open('artifacts/metrics/threshold_tau_v1.txt', 'w') as f:
            f.write(str(tau_star))
        
        with open('artifacts/metrics/champion_model.txt', 'w') as f:
            f.write(champion_model_name)

        st.success(f"Optimal threshold ($\tau^*$) selected using Youden's J for **{champion_model_name}**: `{tau_star:.4f}`")
        st.info("Optimal threshold and champion model name saved to `artifacts/metrics/`.")
    return tau_star, champion_model_name

def run_page2():
    st.header("Model Training & Calibration")
    st.markdown("""
    This section covers the core machine learning pipeline, including data preprocessing,
    training of Logistic Regression and Gradient Boosting models, and model calibration.
    """)

    # Load data splits and feature names
    X_train, y_train, X_val, y_val, numerical_features, ordinal_status_features, categorical_raw_features = load_data_splits()

    st.subheader("1. Preprocessing Pipeline (ColumnTransformer)")
    st.markdown("""
    We use `ColumnTransformer` to apply different preprocessing steps to different types of features:
    - **Numerical Features**: Scaled using `StandardScaler` for Logistic Regression.
    - **Categorical & Ordinal Status Features**: One-Hot Encoded using `OneHotEncoder`.
    
    For Gradient Boosting, numerical features are passed through without scaling as tree-based models are scale-invariant.
    """)
    preprocessor_LR, preprocessor_GB = create_and_fit_preprocessors(X_train, numerical_features, ordinal_status_features, categorical_raw_features)

    st.subheader("2. Model Training – Logistic Regression (Baseline)")
    st.markdown("""
    Logistic Regression is a fundamental linear model for binary classification. We optimize its `C` parameter
    (inverse of regularization strength) using `GridSearchCV` and stratified 5-fold cross-validation on the training data,
    aiming to maximize ROC-AUC. The `class_weight='balanced'` option is used to handle potential class imbalance.
    """)
    best_lr_model = train_logistic_regression(X_train, y_train, preprocessor_LR)

    st.subheader("3. Model Training – Gradient Boosting (Tree-based)")
    st.markdown("""
    Gradient Boosting is an ensemble method that builds a strong predictor from multiple weak learners (decision trees).
    We tune key hyperparameters (`n_estimators`, `learning_rate`, `max_depth`) using `GridSearchCV` with stratified
    5-fold cross-validation on the training data, optimizing for ROC-AUC. We fix `subsample` to `1.0`.
    """)
    best_gb_model = train_gradient_boosting(X_train, y_train, preprocessor_GB)

    st.subheader("4. Validation-based Calibration (Platt Scaling)")
    st.markdown("""
    To ensure that the predicted probabilities truly reflect the likelihood of default, we calibrate both models
    using Platt scaling (`method='sigmoid'`) on the validation set. This helps transform raw model outputs
    into well-calibrated probabilities.
    """)
    calibrated_lr_model, calibrated_gb_model = calibrate_models(best_lr_model, best_gb_model, X_val, y_val)

    st.subheader("5. Threshold Selection (Validation)")
    st.markdown(r"""
    An optimal operating threshold ($\tau^*$) is crucial for converting predicted probabilities into binary
    default/non-default decisions. We select the "champion" model based on its Precision-Recall AUC on the validation set.
    For the champion model, we calculate Youden's J statistic across all possible thresholds on the validation set:

    $$J = \text{True Positive Rate (TPR)} - \text{False Positive Rate (FPR)}$$

    The threshold that maximizes $J$ is chosen as $\tau^*$, as it balances sensitivity and specificity.
    This $\tau^*$ will then be used for final evaluation on the test set for both models.
    """)
    tau_star, champion_model_name = select_optimal_threshold(calibrated_lr_model, calibrated_gb_model, X_val, y_val)

    st.markdown("---")
    st.success("Model training, calibration, and threshold selection completed for this page.")
