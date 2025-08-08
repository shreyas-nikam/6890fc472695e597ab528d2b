# QuLab: Probability of Default (PD) Model Development and Evaluation

## Project Title and Description

QuLab is a Streamlit application designed to facilitate the development, evaluation, and comparison of Probability of Default (PD) models.  PD models are crucial in financial risk management, quantifying the likelihood that a borrower will default on their debt obligations over a specified time horizon. This application provides an interactive interface to explore different stages of PD model development, including data preparation, model training (Logistic Regression and Gradient Boosting), calibration, threshold selection, and performance evaluation.

This lab uses the UCI Credit Card Default dataset to predict whether a customer will default on their credit card payment next month.

## Features

*   **Interactive Data Exploration:** Upload and explore the UCI Credit Card dataset with visualizations and data summaries.
*   **Data Preparation and Preprocessing:** Clean and transform the data, handling missing values and encoding categorical features.
*   **Model Training:** Train two different PD models:
    *   Logistic Regression: A linear model for baseline performance and interpretability.
    *   Gradient Boosting: A powerful ensemble technique for higher accuracy.
*   **Model Calibration:** Calibrate model probabilities using Platt scaling for more accurate predictions.
*   **Threshold Selection:**  Determine an optimal probability threshold using Youden's J statistic on a validation set.
*   **Performance Evaluation:** Evaluate model performance using various metrics on a held-out test set, including ROC AUC, PR AUC, KS-statistic, Brier Score, and standard classification metrics.
*   **Model Selection:** Champions challenger framework that selects the champion model based on the PR-AUC score on the validation set.
*   **Visualization:** Visualize key aspects of model performance, such as calibration curves and ROC curves.
*   **Reproducibility:** Indices used in the train/val/test split are saved to disk for reproducibility.
*   **Artifact Saving:** Models, Preprocessors and Metrics are saved to disk.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` contents:**

    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    joblib
    ```

## Usage

1.  **Download the UCI Credit Card Default dataset:**

    Download the `UCI_Credit_Card.csv` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) or a mirror. Make sure that the file is available in the same directory as the `app.py`

2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

3.  **Using the Application:**

    *   The application will open in your web browser.
    *   The sidebar provides navigation to different stages of the PD model development pipeline.
    *   **Data Preparation & EDA:** Upload the CSV file and explore the dataset.
    *   **Model Training & Calibration:** Train and calibrate Logistic Regression and Gradient Boosting models.
    *   **Model Evaluation:** Evaluate the trained models and compare their performance.

## Project Structure

```
QuLab/
├── app.py                      # Main Streamlit application file
├── application_pages/
│   ├── page1.py                # Data Preparation & EDA page
│   ├── page2.py                # Model Training & Calibration page
│   ├── page3.py                # Model Evaluation page (Implementation pending)
├── artifacts/                   # Directory to store models, metrics, and plots
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   ├── data/
├── requirements.txt            # List of Python dependencies
├── README.md                   # This file
└── UCI_Credit_Card.csv          # UCI Credit Card dataset (download separately)
```

## Technology Stack

*   **Streamlit:**  For creating the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical computing.
*   **Scikit-learn:** For machine learning models, preprocessing, and evaluation.
*   **Plotly:** For interactive visualizations.
*   **Joblib:**  For efficient model persistence.

## Contributing

(Optional)

We welcome contributions to improve QuLab! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure they are well-tested.
4.  Submit a pull request with a clear description of your changes.

## License

[MIT License](LICENSE) (Replace with your desired license)

Copyright (c) 2024 \[Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

[Your Name] - [Your Email]

[Link to your GitHub profile]
