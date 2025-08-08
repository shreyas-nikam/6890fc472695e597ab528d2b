
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# Ensure artifacts directories exist for saving EDA plots and data splits
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

create_directory_if_not_exists('artifacts/models')
create_directory_if_not_exists('artifacts/metrics')
create_directory_if_not_exists('artifacts/plots')
create_directory_if_not_exists('artifacts/data')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_page1():
    st.header("Data Preparation & Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section focuses on loading the raw data, performing necessary cleaning and transformations,
    and conducting initial exploratory data analysis to understand the dataset's characteristics.
    """)

    st.subheader("1. Data Loading")
    st.markdown("""
    Please upload the `UCI_Credit_Card.csv` file to proceed. This dataset contains information on default payments in Taiwan.
    """)

    uploaded_file = st.file_uploader("Upload UCI_Credit_Card.csv", type=["csv"])

    @st.cache_data
    def load_and_preprocess_data(uploaded_file):
        if uploaded_file is None:
            return None, None, None, None, None, None, None, None, None, None, None

        df = pd.read_csv(uploaded_file, encoding="utf-8")

        expected_columns = [
            'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
            'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
            'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
            'PAY_AMT6', 'default.payment.next.month'
        ]
        
        # Validate columns (simplified for Streamlit display)
        df_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        if not expected_columns_set.issubset(df_columns):
            missing_cols = list(expected_columns_set - df_columns)
            st.error(f"Error: Missing expected columns in the uploaded file: {missing_cols}")
            st.stop()
        
        if not df_columns.issubset(expected_columns_set):
            extra_cols = list(df_columns - expected_columns_set)
            st.warning(f"Warning: Dropping extra columns found in the dataset: {extra_cols}")
            df = df.drop(columns=extra_cols)

        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - df.shape[0]

        st.info(f"Initial DataFrame shape: {initial_rows} rows, {df.shape[1]} columns")
        st.info(f"Number of duplicate rows removed: {duplicates_removed}")
        st.info(f"Final DataFrame shape after dropping duplicates: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())

        # Target and feature selection
        y = df["default.payment.next.month"]
        X_cols = [col for col in df.columns if col not in ['ID', 'default.payment.next.month']]
        X = df[X_cols]

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

        # Deterministic cleaning
        X_cleaned = X.copy()
        X_cleaned['EDUCATION'] = X_cleaned['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
        X_cleaned['MARRIAGE'] = X_cleaned['MARRIAGE'].replace({0: 3})

        st.success("Data loaded and deterministically cleaned.")
        st.write("Unique values in EDUCATION after cleaning:", sorted(X_cleaned['EDUCATION'].unique()))
        st.write("Unique values in MARRIAGE after cleaning:", sorted(X_cleaned['MARRIAGE'].unique()))

        # Train/Validation/Test split
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_cleaned, y,
                                                                    test_size=0.15,
                                                                    stratify=y,
                                                                    random_state=RANDOM_SEED)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          test_size=(0.15/0.85),
                                                          stratify=y_train_val,
                                                          random_state=RANDOM_SEED)
        
        # Save indices for reproducibility
        try:
            np.save('artifacts/data/train_indices.npy', X_train.index.values)
            np.save('artifacts/data/val_indices.npy', X_val.index.values)
            np.save('artifacts/data/test_indices.npy', X_test.index.values)
            st.info("Train, Validation, and Test indices saved.")
        except Exception as e:
            st.error(f"Could not save indices: {e}")

        return X, y, X_train, y_train, X_val, y_val, X_test, y_test, numerical_features, ordinal_status_features, categorical_raw_features

    X, y, X_train, y_train, X_val, y_val, X_test, y_test, numerical_features, ordinal_status_features, categorical_raw_features = load_and_preprocess_data(uploaded_file)

    if uploaded_file is None:
        st.warning("Please upload the CSV file to see data details and EDA.")
        st.stop() # Stop execution if file is not uploaded

    st.subheader("2. Target and Feature Selection")
    st.markdown(f"""
    The target variable for our prediction is `default.payment.next.month`.
    The features used for modeling are:
    - **Numerical Features**: `{", ".join(numerical_features)}`
    - **Ordinal Status Features (PAY_X)**: `{", ".join(ordinal_status_features)}`
    - **Categorical Raw Features**: `{", ".join(categorical_raw_features)}`
    """)
    st.write(f"Shape of features (X): {X.shape}")
    st.write(f"Shape of target (y): {y.shape}")

    st.subheader("3. Deterministic Cleaning")
    st.markdown("""
    The `EDUCATION` and `MARRIAGE` columns have some inconsistent values that are mapped to valid categories.
    Specifically:
    - `EDUCATION`: values `{0, 5, 6}` are mapped to `4` (Others).
    - `MARRIAGE`: value `{0}` is mapped to `3` (Others).
    """)

    st.subheader("4. Train/Validation/Test Split")
    st.markdown(f"""
    The dataset is split into training, validation, and test sets with a 70/15/15 ratio, stratified by the target variable
    (`default.payment.next.month`) to ensure class balance across splits. The `RANDOM_SEED` for reproducibility is `{RANDOM_SEED}`.
    """)
    st.write(f"Training set shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
    st.write(f"Validation set shape (X_val, y_val): {X_val.shape}, {y_val.shape}")
    st.write(f"Test set shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

    st.subheader("5. Exploratory Data Analysis (EDA)")

    st.markdown("#### Class Balance")
    st.markdown("""
    This chart shows the distribution of the target variable (`default.payment.next.month`) in the training set.
    Understanding the class balance is crucial, especially in imbalanced datasets like credit default,
    as it influences model evaluation metrics and strategies.
    """)
    class_counts = y_train.value_counts().sort_index()
    fig_class_balance = px.bar(class_counts, x=class_counts.index, y=class_counts.values,
                               labels={'x': 'Default (1) / Non-Default (0)', 'y': 'Number of Samples'},
                               title='Class Balance on Training Data')
    fig_class_balance.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Non-Default', 'Default']))
    st.plotly_chart(fig_class_balance)

    st.markdown("#### Correlation Heatmap")
    st.markdown("""
    A correlation heatmap visualizes the correlation matrix of numerical features.
    It helps identify relationships between variables, which can inform feature engineering and selection.
    Strong correlations (positive or negative) are indicated by brighter colors.
    """)
    if numerical_features: # Ensure there are numerical features to plot
        corr_df = X_train[numerical_features + ordinal_status_features].corr()
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto",
                             title="Correlation Heatmap of Numeric and Ordinal Features on Training Data")
        fig_corr.update_layout(height=800, width=800)
        st.plotly_chart(fig_corr)
    else:
        st.info("No numerical or ordinal status features to display correlation heatmap.")
