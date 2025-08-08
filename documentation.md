id: 6890fc472695e597ab528d2b_documentation
summary: Lab 2.1: PD Models - Development Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Probability of Default (PD) Model Development and Evaluation

This codelab provides a comprehensive guide to building and evaluating Probability of Default (PD) models using a Streamlit application called QuLab. PD models are crucial for financial institutions to assess the likelihood of borrowers defaulting on their debt obligations. This application allows you to explore different stages of PD model development, including data preparation, model training (Logistic Regression and Gradient Boosting), calibration, threshold selection, and performance evaluation.

## Importance

This application is significant because it provides a practical, hands-on approach to understanding the complexities of PD model development. By using this application, developers and risk managers can gain insights into:

*   Data preprocessing techniques essential for building robust models.
*   The application of different machine learning algorithms, such as Logistic Regression and Gradient Boosting, in the context of credit risk.
*   The importance of model calibration for reliable probability estimates.
*   Threshold selection strategies and their impact on model performance.
*   The evaluation of model performance using various metrics.

## Concepts Explained

Throughout this codelab, you will learn about the following key concepts:

*   **Probability of Default (PD):** The likelihood that a borrower will default on their debt obligations.
*   **Logistic Regression:** A linear model that estimates the probability of a binary outcome.
*   **Gradient Boosting:** A powerful ensemble technique that builds a model in a stage-wise fashion.
*   **Model Calibration:** The process of ensuring that predicted probabilities accurately reflect the true likelihood of an event.
*   **Youden's J Statistic:** A metric used for threshold selection that maximizes the difference between the True Positive Rate (TPR) and False Positive Rate (FPR).
*   **ROC AUC:** A metric that measures the ability of a model to distinguish between classes.
*   **PR AUC:** A metric particularly useful for imbalanced datasets, focusing on the trade-off between precision and recall.
*   **KS-statistic:** A metric that measures the maximum difference between the cumulative true positive and cumulative false positive rates.
*   **Brier Score:** A metric that quantifies the accuracy of probabilistic predictions.

## Codelab Step
Duration: 00:05

### Setup and Overview

1.  **Clone the Repository:** Begin by cloning the repository containing the Streamlit application.
2.  **Install Dependencies:** Navigate to the project directory and install the required Python packages using `pip install -r requirements.txt`. (Note: You'll need a `requirements.txt` file listing the dependencies. Common ones include `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, and `joblib`.)
3.  **Run the Application:** Execute the Streamlit application by running `streamlit run app.py` in your terminal.

Once the application is running, you will see the main page with a sidebar navigation menu. The application consists of three main pages:

*   **Data Preparation & EDA:** Focuses on loading, cleaning, and exploring the data.
*   **Model Training & Calibration:** Covers model training, hyperparameter tuning, and calibration techniques.
*   **Model Evaluation:** Provides tools for evaluating the performance of the trained models.

## Codelab Step
Duration: 00:15

### Data Preparation & EDA

This page guides you through loading the credit card default dataset, performing cleaning operations, and conducting exploratory data analysis.

1.  **Data Loading:**
    *   Upload the `UCI_Credit_Card.csv` file using the file uploader widget.
    *   The application will load the data, display the first few rows, and provide information about the dataset's shape.
    *   Duplicate rows are automatically removed.
2.  **Target and Feature Selection:**
    *   The target variable is `default.payment.next.month`.
    *   The features are categorized into numerical, ordinal status (PAY\_X), and categorical raw features.
3.  **Deterministic Cleaning:**
    *   The `EDUCATION` column's values 0, 5, and 6 are mapped to 4 (Others).
    *   The `MARRIAGE` column's value 0 is mapped to 3 (Others).
4.  **Train/Validation/Test Split:**
    *   The dataset is split into training (70%), validation (15%), and test (15%) sets, stratified by the target variable.
    *   The application displays the shapes of the resulting datasets.
5.  **Exploratory Data Analysis (EDA):**
    *   **Class Balance:** A bar chart shows the distribution of the target variable in the training set. This visual helps assess if the dataset is imbalanced.
    *   **Correlation Heatmap:** A heatmap visualizes the correlation matrix of numerical features in the training set. This helps identify potential multicollinearity issues.

<aside class="positive">
<b>Tip:</b> Pay close attention to the class balance. Imbalanced datasets can significantly impact model performance.
</aside>

<aside class="negative">
<b>Warning:</b> Ensure that the uploaded CSV file has the correct format and column names to avoid errors.
</aside>

## Codelab Step
Duration: 00:25

### Model Training & Calibration

This page focuses on training Logistic Regression and Gradient Boosting models, as well as calibrating their outputs.

1.  **Preprocessing Pipeline (ColumnTransformer):**
    *   A `ColumnTransformer` is used to apply different preprocessing steps to different feature types.
    *   Numerical features are scaled using `StandardScaler` for Logistic Regression.
    *   Categorical and ordinal status features are one-hot encoded using `OneHotEncoder`.
    *   For Gradient Boosting, numerical features are passed through without scaling.
2.  **Model Training – Logistic Regression (Baseline):**
    *   A Logistic Regression model is trained using `GridSearchCV` to optimize the `C` parameter (inverse of regularization strength).
    *   Stratified 5-fold cross-validation is used to ensure robust performance.
    *   The `class_weight='balanced'` option handles potential class imbalance.
3.  **Model Training – Gradient Boosting (Tree-based):**
    *   A Gradient Boosting model is trained using `GridSearchCV` to tune key hyperparameters (`n_estimators`, `learning_rate`, `max_depth`).
    *   Stratified 5-fold cross-validation is used to optimize performance.
4.  **Validation-based Calibration (Platt Scaling):**
    *   Both models are calibrated using Platt scaling (`method='sigmoid'`) on the validation set.  This transforms the raw model outputs into well-calibrated probabilities.
5.  **Threshold Selection (Validation):**
    *   The "champion" model is selected based on its Precision-Recall AUC on the validation set.
    *   Youden's J statistic is calculated for the champion model on the validation set:

    $$J = TPR - FPR$$

    *   The threshold that maximizes $J$ is chosen as $\tau^*$.

<aside class="positive">
<b>Tip:</b> Experiment with different hyperparameter values in the `GridSearchCV` to potentially improve model performance.
</aside>

<aside class="negative">
<b>Warning:</b> Model calibration is essential for reliable probability estimates.  Skipping this step can lead to poor decision-making.
</aside>

## Codelab Step
Duration: 00:15

### Model Evaluation

This page provides tools for evaluating the performance of the trained and calibrated models on the test set.

1.  **Load Models and Data:** The page will load the calibrated models and the test dataset previously split in the first page.
2.  **Probability Predictions:** Predict probabilities on the test set using both the Logistic Regression and Gradient Boosting models.
3.  **Performance Metrics Calculation:**
    *   **ROC AUC:** Calculated for both models to assess their ability to discriminate between classes.
    *   **PR AUC:** Calculated for both models, particularly useful for imbalanced datasets.
    *   **KS-Statistic:** Calculated for both models to measure the maximum difference between the cumulative true positive and cumulative false positive rates.
    *   **Brier Score:** Calculated for both models to quantify the accuracy of probabilistic predictions.
    *   **Accuracy, Precision, Recall, F1-score:** Calculated at the optimal threshold ($\tau^*$) for both models.
4.  **Visualization:**
    *   **ROC Curves:** Interactive plot showing the ROC curves for both models.
    *   **Precision-Recall Curves:** Interactive plot showing the precision-recall curves for both models.
5.  **Threshold effect visualization:**
    *   Visualize the effect of different threshold on precision, recall and F1-score.
    *   Display the confusion matrix.

<aside class="positive">
<b>Tip:</b> Focus on metrics that are most relevant to your specific business needs. For example, PR AUC is often more informative than ROC AUC for imbalanced datasets.
</aside>

## Codelab Step
Duration: 00:05

### Conclusion

This codelab provided a comprehensive guide to building and evaluating Probability of Default (PD) models using the QuLab Streamlit application. By working through the steps, you have gained hands-on experience with data preparation, model training, calibration, threshold selection, and performance evaluation. You can further extend this application by incorporating other machine learning algorithms, exploring different feature engineering techniques, and adding more sophisticated model evaluation metrics.
