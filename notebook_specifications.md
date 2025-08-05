
# Jupyter Notebook Specification: PD Model Development Lab

## 1. Notebook Overview

**Learning Goals:**

This notebook guides the user through the development of a Probability of Default (PD) classification model using the UCI Taiwan Credit Card Default dataset.

**Expected Outcomes:**

Upon completion of this notebook, the user will be able to:

*   Load, validate, and clean the UCI Taiwan dataset.
*   Perform feature engineering suitable for PD modeling.
*   Handle class imbalance using appropriate techniques and choose suitable metrics.
*   Train and evaluate baseline (Logistic Regression) and tree-based (Gradient Boosting) PD models.
*   Calibrate predicted PDs and assess probability quality.
*   Select an operating threshold and generate a confusion matrix and classification report.
*   Persist all artifacts, metrics, and plots in a structured and documented format.

## 2. Mathematical and Theoretical Foundations

This section provides the theoretical background for the PD model development process.

*   **Probability of Default (PD):**  The probability that a borrower will default on their debt obligations over a specified time horizon.

*   **Logistic Regression:** A linear model that uses a sigmoid function to predict the probability of a binary outcome (default or no default).

    $$P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$

    Where:
    *   $P(Y=1 | X)$ is the probability of default given the features $X$.
    *   $\beta_0$ is the intercept.
    *   $\beta$ is the vector of coefficients.
    *   $X$ is the vector of input features.

*   **Gradient Boosting:** A tree-based ensemble method that combines multiple weak learners (decision trees) to create a strong learner. It minimizes a loss function by iteratively adding trees that correct the errors of previous trees.

*   **Class Imbalance:**  A situation where the classes in a classification problem are not represented equally. In PD modeling, default events are typically much rarer than non-default events.  This requires special handling, such as using class weights or appropriate evaluation metrics.

*   **ROC AUC (Area Under the Receiver Operating Characteristic Curve):**  A metric that measures the ability of a model to distinguish between positive and negative classes. It represents the area under the ROC curve, which plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.  An AUC of 1 indicates perfect classification, while an AUC of 0.5 indicates random chance.
    $$AUC = \int_{0}^{1} TPR(FPR) d(FPR)$$
    Where:
    * $TPR$ is the True Positive Rate
    * $FPR$ is the False Positive Rate

*   **PR AUC (Area Under the Precision-Recall Curve):**  A metric that measures the trade-off between precision and recall. Precision is the proportion of positive predictions that are actually positive, while recall is the proportion of actual positives that are correctly predicted.  PR AUC is particularly useful for imbalanced datasets.
   $$AUC = \int_{0}^{1} Precision(Recall) d(Recall)$$
    Where:
    * $Precision$ is the proportion of true positives out of all predicted positives
    * $Recall$ is the proportion of true positives out of all actual positives

*   **KS Statistic (Kolmogorov-Smirnov Statistic):**  A metric that measures the maximum difference between the cumulative distribution functions of the positive and negative classes.  It is a good indicator of the model's ability to separate the two classes.

*   **Brier Score:** Measures the accuracy of probabilistic predictions. It calculates the mean squared difference between the predicted probability and the actual outcome (0 or 1). Lower Brier scores indicate better calibration.

    $$Brier Score = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

    Where:
    *   $N$ is the number of predictions.
    *   $f_i$ is the predicted probability.
    *   $o_i$ is the actual outcome (0 or 1).

*   **Calibration:** The process of ensuring that predicted probabilities accurately reflect the true likelihood of an event.  A well-calibrated model will predict probabilities that are close to the observed event rates. Platt scaling is used.

*   **Platt Scaling:** A method for calibrating the output of a classification model by fitting a logistic regression model to the model's predicted probabilities.

*  **Youden's J statistic:** A single statistic that captures the performance of a diagnostic test.

$$J = Sensitivity + Specificity - 1$$
Or, equivalently:
$$J = TPR - FPR$$
Where:
* $TPR$ is the true positive rate
* $FPR$ is the false positive rate

## 3. Code Requirements

**Expected Libraries:**

*   pandas (data manipulation and analysis)
*   numpy (numerical computing)
*   scikit-learn (machine learning algorithms and tools)
*   matplotlib (plotting and visualization)
*   joblib (saving and loading Python objects)

**Input/Output Expectations:**

*   **Input:** The notebook will load the UCI Taiwan Credit Card Default dataset from a CSV file (`data/UCI_Credit_Card.csv`).
*   **Output:** The notebook will produce various model artifacts, metrics, and plots, which will be saved in the `artifacts/` directory. The notebook will also generate a `README.md` file summarizing the key findings and instructions for using the generated artifacts.

**Algorithms/Functions to be Implemented:**

1.  **Data Loading and Validation:**
    *   Function to read the CSV file into a pandas DataFrame.
    *   Function to validate the DataFrame's schema (column names, data types).
    *   Function to handle missing values using imputation (median for numerical, most frequent for categorical).
    *   Function to perform one-hot encoding of categorical features.
2.  **Data Splitting:**
    *   Function to split the dataset into training, validation, and test sets, maintaining class proportions (stratified splitting).
3.  **Preprocessing Pipeline:**
    *   Function to create a ColumnTransformer that applies different preprocessing steps to different feature types. Includes scaling of numerical features.
4.  **Model Training:**
    *   Function to train a Logistic Regression model with cross-validated hyperparameter tuning.
    *   Function to train a Gradient Boosting model with cross-validated hyperparameter tuning.
    *   Function to perform Platt scaling calibration on the validation set.
5.  **Model Evaluation:**
    *   Function to calculate and save evaluation metrics (ROC AUC, PR AUC, KS statistic, Brier score, accuracy, precision, recall, F1).
    *   Function to select an operating threshold based on maximizing Youden's J statistic on the validation set.
    * Function to plot confusion matrix based on selected threshold.
6.  **Visualization:**
    *   Function to generate a class balance bar chart.
    *   Function to generate a correlation heatmap.
    *   Function to generate ROC curves.
    *   Function to generate precision-recall curves.
    *   Function to generate a calibration curve.
    *   Function to generate a confusion matrix plot.
    *   Function to generate a feature importance plot (for Gradient Boosting).
7.  **Artifact Persistence:**
    *   Function to save models, preprocessors, and other artifacts to disk using `joblib`.
    *   Function to save metrics to CSV and JSON files.
    *   Function to save plots as PNG files.
    *   Function to generate a `README.md` file summarizing the project.

**Visualizations:**

*   **Class Balance Bar Chart:**  A bar chart showing the number of instances in each class (default vs. no default) in the training set.  This will highlight the class imbalance problem.
*   **Correlation Heatmap:**  A heatmap showing the pairwise correlations between numeric features. This helps to identify highly correlated features that may need to be addressed.
*   **ROC Curves (Both Models):**  ROC curves for both the Logistic Regression and Gradient Boosting models, plotted on the same graph for comparison.  The ROC curves will be generated on the test set.
*   **Precision-Recall Curves (Both Models):** Precision-Recall curves for both models, plotted on the same graph. Useful because of class imbalance. The PR curves will be generated on the test set.
*   **Calibration (Reliability) Diagram:** A calibration diagram showing the relationship between predicted probabilities and observed event rates. This will help to assess the calibration of the model's probability predictions.
*   **Confusion Matrix at Chosen Threshold:**  A confusion matrix showing the number of true positives, true negatives, false positives, and false negatives at the selected operating threshold (τ\*).
*   **Feature Importance (GB):** A bar chart showing the importance of each feature in the Gradient Boosting model.  Feature importance is typically measured by the gain or reduction in impurity achieved by each feature.

## 4. Additional Notes or Instructions

*   **Environment:**  The notebook should be run in an environment with the specified library versions installed (Python ≥3.9; pandas ≥2.0; numpy ≥1.26; scikit-learn ≥1.4; matplotlib ≥3.7; joblib ≥1.3).
*   **Random Seeds:** Ensure that all random number generators are seeded for reproducibility (e.g., `numpy.random.seed(42)`, `random_state=42` in scikit-learn models).
*   **Imbalance Handling:**  Use `class_weight="balanced"` for the Logistic Regression and Gradient Boosting models. This assigns higher weights to the minority class (default) to compensate for the class imbalance.
*   **Metrics:** Compute and save all the specified metrics (ROC AUC, PR AUC, KS statistic, Brier score, Accuracy, Precision, Recall, F1) on both the validation and test sets.
*   **Calibration:** Use Platt scaling (sigmoid calibration) on the validation set and apply the calibrated model to the test set.
*   **Threshold Selection:**  Choose the operating threshold (τ\*) on the validation set by maximizing Youden's J statistic. Apply the same threshold to the test set.
*   **File Structure:**  Create the specified directory structure (`artifacts/models/`, `artifacts/metrics/`, `artifacts/plots/`, `artifacts/data/`) to store all generated artifacts.
*   **Saved Artifacts:** Persist all artifacts with the exact filenames specified in the requirements.
*   **Data Snapshots:** Save the indices of the training, validation, and test sets to `artifacts/data/` for reproducibility.
*   **Documentation:**  Create a `README.md` file that describes the data cleaning rules, feature engineering steps, model versions, best parameters, threshold selection method, metric definitions, and instructions for reloading the pipeline and scoring new data.

## 5. Exact step-by-step notebook plan (cells & outputs)

1.  **Setup & Folders**
    *   Create folders: `artifacts/models`, `artifacts/metrics`, `artifacts/plots`, `artifacts/data`.
    *   Set global seed `42`; print library versions.
    *   *Output:* Print library versions to console.

2.  **Load data**
    *   Read `data/UCI_Credit_Card.csv` with `encoding="utf-8"`.
    *   Assert expected column set exactly (order not required).
    *   Drop duplicates (if any) keeping first; verify final row count printed.
    *   *Output:* Print dataframe shape

3.  **Target and feature selection**
    *   Define `y = df["default.payment.next.month"]`.
    *   Define feature list `X_cols = [all columns except ID and target]`; drop `ID`.
    *   Split features into categorical\_raw, ordinal\_status, and numeric lists.

4.  **Deterministic cleaning**
    *   Map EDUCATION: values `{0,5,6} → 4`.
    *   Map MARRIAGE: value `{0} → 3`.
    *   Assert final categorical unique sets.

5.  **Train/Validation/Test split**
    *   Stratified split into 70/15/15 with seed 42.
    *   Save index arrays to `artifacts/data/train_indices.npy`, etc.

6.  **EDA (light, deterministic)**
    *   Class balance bar chart on **train** only → `artifacts/plots/class_balance.png`.
    *   Correlation heatmap on numeric features of **train** → `artifacts/plots/corr_heatmap.png`.

7.  **Preprocessing pipeline (ColumnTransformer)**
    *   Create ColumnTransformer for Logistic Regression and one for Gradient Boosting.
    *   Fit **both** preprocessors on **train** only; save.

8.  **Model training – Logistic Regression (baseline)**
    *   Pipeline: `preprocessor_LR` → `LogisticRegression`.
    *   Grid Search, Cross-validation; optimize ROC-AUC on **train**; refit on **train**.
    *   Save best params and model.

9.  **Model training – Gradient Boosting (tree-based)**
    *   Pipeline: `preprocessor_GB` → `GradientBoostingClassifier`.
    *   Grid Search, Cross-validation; optimize ROC-AUC on **train**; refit best on **train**.
    *   Save best params and model.

10. **Validation-based calibration (Platt scaling)**
    *   For **each best model**, get **validation** set probabilities; fit `CalibratedClassifierCV` on the **validation** data only.
    *   Save calibrated models.

11. **Threshold selection (validation)**
    *   Compute ROC curve on **validation** for the **calibrated** model with higher PR-AUC on validation (select “champion”).
    *   Compute Youden’s J, set τ\* to argmax(J).
    *   Save τ\* numeric value to `artifacts/metrics/threshold_tau_v1.txt`.

12. **Final evaluation (test)**
    *   Compute and save metrics: ROC-AUC, PR-AUC, KS-statistic, Brier score, Accuracy, Precision, Recall, F1 at τ\*.
    *   Plots: ROC, PR, Calibration, Confusion Matrix.
    *   Save calibration stats.

13. **Feature importance (tree-based only)**
    *   Extract `feature_importances_` from the **best GB**.
    *   Save CSV and bar plot.

14. **Persist splits & sample predictions**
    *   Save train/val/test indices.
    *   Save sample predictions.

15. **Artifact map**
    *   Write `artifacts/README.md`.

