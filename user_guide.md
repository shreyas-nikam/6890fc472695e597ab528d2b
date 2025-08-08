id: 6890fc472695e597ab528d2b_user_guide
summary: Lab 2.1: PD Models - Development User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Probability of Default (PD) Model Development: A User Guide

This codelab provides a comprehensive walkthrough of building and evaluating Probability of Default (PD) models using a Streamlit application. PD models are crucial for financial institutions to assess the likelihood that a borrower will default on their debt obligations. Through this guide, you'll learn about data preparation, model training, calibration, threshold selection, and performance evaluation – all essential steps in developing robust PD models. By the end of this codelab, you will have a solid understanding of how to use this application to build and compare different PD models.

## Data Preparation & EDA
Duration: 00:10

In this step, we will explore the first page of the Streamlit application, which focuses on data preparation and Exploratory Data Analysis (EDA). This is a crucial initial step in any machine learning project as it helps us understand the data we are working with and prepare it for model training.

First, select "Data Preparation & EDA" from the navigation bar in the sidebar.

### Data Loading
Duration: 00:05

The application starts by allowing you to upload the `UCI_Credit_Card.csv` dataset. This dataset contains information about credit card defaults in Taiwan and will be the foundation for our PD models.

1.  Click on the "Browse files" button and select the `UCI_Credit_Card.csv` file from your local machine.

<aside class="positive">
<b>Tip:</b> Make sure the file is in CSV format for the application to process it correctly.
</aside>

Once the file is uploaded, the application will perform several checks and display key information about the dataset, including:

*   Initial and final DataFrame shapes.
*   The number of duplicate rows removed.
*   A preview of the first few rows of the DataFrame.

### Target and Feature Selection
Duration: 00:03

After loading the data, the application highlights the target variable and the features that will be used for modeling.

*   **Target Variable:** `default.payment.next.month` (Indicates whether a customer defaulted on their credit card payment next month).
*   **Features:** The application specifies the different types of features used:
    *   Numerical Features.
    *   Ordinal Status Features (PAY_X).
    *   Categorical Raw Features.

Understanding which features are used is essential for interpreting the model's results and understanding its behavior.

### Deterministic Cleaning
Duration: 00:02

The application then performs some deterministic cleaning on the data to handle inconsistent values in the `EDUCATION` and `MARRIAGE` columns.

*   `EDUCATION`: Values `{0, 5, 6}` are mapped to `4` (Others).
*   `MARRIAGE`: Value `{0}` is mapped to `3` (Others).

These cleaning steps ensure data consistency and can improve model performance.

### Train/Validation/Test Split
Duration: 00:03

To properly train and evaluate our PD models, the data is split into training, validation, and test sets.

*   **Ratio:** 70/15/15 split for training/validation/test sets.
*   **Stratification:** The split is stratified by the target variable to ensure class balance across the different sets.
*   **Random Seed:** A `RANDOM_SEED` is used for reproducibility.

The application displays the shapes of the resulting training, validation, and test sets.

<aside class="negative">
<b>Warning:</b>  The test set should only be used at the end of the model development process to get an unbiased estimate of the model's performance on unseen data.
</aside>

### Exploratory Data Analysis (EDA)
Duration: 00:05

Finally, the application provides some basic EDA visualizations to help us understand the data:

*   **Class Balance:** A bar chart showing the distribution of the target variable in the training set. This helps us assess whether the dataset is imbalanced, which can impact model performance.
*   **Correlation Heatmap:** A heatmap visualizing the correlation matrix of numerical features. This helps us identify relationships between variables, which can inform feature engineering and selection.

These visualizations provide valuable insights into the data and can guide our modeling decisions.

## Model Training & Calibration
Duration: 00:15

Now that we have prepared and explored our data, we can move on to training our PD models. This step focuses on the second page of the Streamlit application, where we will train and calibrate Logistic Regression and Gradient Boosting models.

Select "Model Training & Calibration" from the navigation bar in the sidebar.

### Preprocessing Pipeline (ColumnTransformer)
Duration: 00:03

Before training our models, we need to preprocess the data to handle different types of features. The application uses `ColumnTransformer` to apply different preprocessing steps to different types of features:

*   **Numerical Features:** Scaled using `StandardScaler` for Logistic Regression.
*   **Categorical & Ordinal Status Features:** One-Hot Encoded using `OneHotEncoder`.

For Gradient Boosting, numerical features are passed through without scaling as tree-based models are scale-invariant.

<aside class="positive">
<b>Tip:</b>  Proper data preprocessing is crucial for achieving good model performance. StandardScaler helps bring all numerical features to the same scale, while OneHotEncoder converts categorical features into a format that machine learning models can understand.
</aside>

### Model Training – Logistic Regression (Baseline)
Duration: 00:03

Logistic Regression is a fundamental linear model for binary classification. In this step, we train a Logistic Regression model on the training data, optimizing its `C` parameter (inverse of regularization strength) using `GridSearchCV` and stratified 5-fold cross-validation.

* `class_weight='balanced'` is used to handle potential class imbalance.

The application displays the best parameters found for the Logistic Regression model.

### Model Training – Gradient Boosting (Tree-based)
Duration: 00:03

Gradient Boosting is a powerful ensemble method that builds a strong predictor from multiple weak learners (decision trees). We tune key hyperparameters (`n_estimators`, `learning_rate`, `max_depth`) using `GridSearchCV` with stratified 5-fold cross-validation on the training data.

The application displays the best parameters found for the Gradient Boosting model.

### Validation-based Calibration (Platt Scaling)
Duration: 00:03

To ensure that the predicted probabilities truly reflect the likelihood of default, we calibrate both models using Platt scaling (`method='sigmoid'`) on the validation set.

*   Platt scaling transforms raw model outputs into well-calibrated probabilities.

### Threshold Selection (Validation)
Duration: 00:03

An optimal operating threshold ($\tau^*$) is crucial for converting predicted probabilities into binary default/non-default decisions. We select the "champion" model based on its Precision-Recall AUC on the validation set.

For the champion model, we calculate Youden's J statistic across all possible thresholds on the validation set:

$$J = \text{True Positive Rate (TPR)} - \text{False Positive Rate (FPR)}$$

The threshold that maximizes $J$ is chosen as $\tau^*$, as it balances sensitivity and specificity.

The application displays the selected champion model and the optimal threshold ($\tau^*$).

## Model Evaluation
Duration: 00:10

Now that we have trained and calibrated our models and selected an optimal threshold, we can evaluate their performance on the test set. This step focuses on the third page of the Streamlit application.

Select "Model Evaluation" from the navigation bar in the sidebar.

### Load Models and Threshold

The application loads the trained and calibrated models, the optimal threshold ($\tau^*$), and the selected champion model from the previous steps.

### Evaluate Models on Test Set

The application evaluates the performance of both models on the test set using a variety of metrics:

*   **ROC AUC:** Measures the ability of the model to distinguish between classes.
*   **PR AUC:** Particularly useful for imbalanced datasets, it focuses on the trade-off between precision and recall.
*   **KS-statistic:** Measures the maximum difference between the cumulative true positive and cumulative false positive rates.
*   **Brier Score:** Quantifies the accuracy of probabilistic predictions.
*   **Accuracy, Precision, Recall, F1-score:** Standard classification metrics calculated at the selected optimal threshold.

The application displays these metrics for both models, allowing you to compare their performance.

### Visualization of Performance Metrics

The application also provides visualizations of key performance metrics:

*   **ROC Curve:** A plot of the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
*   **Precision-Recall Curve:** A plot of precision (PPV) against recall (TPR) at various threshold settings.
*   **Calibration Curve:** A plot of the predicted probabilities against the actual observed frequencies.

These visualizations provide further insights into the models' performance and can help you identify potential issues.

By completing these steps, you will have a solid understanding of how to use this application to build and evaluate PD models.
