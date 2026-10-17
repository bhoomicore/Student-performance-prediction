# Student Performance Prediction Using Machine Learning

## 1. Project Overview

This project builds a regression model to predict students' final grades using the UCI Student Performance dataset.

It compares a baseline model with Ridge Regression and Random Forest to see how well they can predict final grades given factors like gender, age, grades, extra-curricular activities, etc.

## 2. Objective

- Build and evaluate a regression model.
- Prepare and process features.
- Scale features appropriately.
- Evaluate the model using train-test splitting and cross-validation.
- Compare against a baseline.
- Plot evaluation charts and analyze prediction errors.
- Discuss assumptions, limitations and potential failure cases.

## 3. Dataset

Dataset: Student Performance
Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/320/student%2Bperformance
Target variable: G3

This dataset contains students' academic, demographic and social information. The data is collected from two Portuguese schools.

The earlier grade variables G1 and G2 are excluded from the model inputs to evaluate prediction without directly relying on these earlier grades.

## 4. Methodology

### 4.1 Loading the data

The dataset is downloaded from the UCI Machine Learning Repository using the `ucimlrepo` package. The target variable is converted to numeric form, and the records without a valid target are removed.

### 4.2 Processing and feature engineering

- Identify numerical and categorical columns.
- Impute missing values: median for numerical, most frequent for categorical.
- Standardize numerical features using the StandardScaler.
- Encode categorical features as numerical columns using the OneHotEncoder.
- Combine preprocessing and model building in a scikit-learn Pipeline.

### 4.3 Train-Test split

The dataset is split into training and testing sets with ratios of 80% and 20%, respectively. The split uses the `random_state=42` seed.

### 4.4 Cross-Validation

Five-fold cross-validation is performed on the training data, and the models are compared using cross-validation Mean Absolute Error (MAE).

The model comparison is done on the test set which is held out from the model selection process.
### 4.5 Models

Dummy Regressor: A baseline model that predicts the mean of the training target values.
Ridge Regression: A regularized linear regression estimator that helps in limiting the coefficients from becoming too large.
Random Forest Regressor: An ensemble estimator that combines multiple decision trees to learn the relationship between the features and target variable.

### 4.6 Evaluation metrics

Mean Absolute Error (MAE): This measures the absolute difference between the actual and predicted grade values. A lower MAE is better.
Root Mean Squared Error (RMSE): This is similar to MAE but penalizes the larger errors more. A lower RMSE is better.

### 4.7 Plots
The project produces an Actual vs Predicted plot, Residual plot and a Feature importance plot.

Permutation importance is used to estimate the decrease in performance when an input feature is randomly shuffled.

## 5. Performance Benchmarks

The baseline model is compared with Ridge Regression and Random Forest using 5-fold cross-validation and test-set evaluation. Lower MAE and RMSE values indicate better prediction performance.

Model| Cross-Validation MAE| Test MAE| Test RMSE
Dummy Regressor (Baseline)| 2.4278| 2.3946| 3.1726
Ridge Regression| 2.0423| 2.1518| 2.8580
Random Forest| 1.9916| 2.0193| 2.7630

Source of results: "outputs/model_comparison.csv"

Result analysis: Random Forest achieved the best performance, with the lowest cross-validation MAE (1.9916), test MAE (2.0193), and test RMSE (2.7630). Compared with the baseline model, Random Forest reduced test MAE by approximately 15.7%, indicating improved prediction accuracy on the test dataset.

## 6. Generated Outputs
- `outputs/model_comparison.csv` — model evaluation comparison
- `outputs/test_predictions.csv` — actual grades, predictions and residuals
- `outputs/evaluation_summary.txt` — final evaluation summary
- `outputs/actual_vs_predicted.png` — prediction comparison chart
- `outputs/residual_plot.png` — prediction error analysis
- `outputs/feature_importance.csv` — feature importance scores
- `outputs/feature_importance.png` — feature importance visualization
## 7. Known Limitations and Failure Cases
1. We cannot generalize the findings to all countries and schools as this represents only two Portuguese schools.
2. The number of samples is small, and different train-test splits might yield different results.
3. There are factors that we haven't captured that affect the target variable.
4. There are unusual students in the input data that the model might fail to predict, shown by the large residual errors in the residual plot and the test predictions file.
5. There might be other social or demographic patterns in the historical data that we're not capturing fairly.
6. The feature importance chart doesn't indicate the degree of interaction between features and the target variable.
7. The model trained with random splits might perform differently on future data, especially if the school population differs significantly.
8. The exclusion of G1 and G2 from the features decreases the amount of information about the target variable.
9. Students shouldn't be penalized solely based on this model's scores.
## 8. Assumptions and Leakage Prevention
The model assumes that the available features are recorded consistently, and the new cases are reasonably similar to the training data.
G1 and G2 are excluded because they are strongly associated with the final grade G3 and are unavailable at the time of prediction.
Imputation, scaling and encoding are performed within a pipeline. This is fitted on the training portion in each cross-validation fold to help prevent preprocessing leakage.
The test set, which is held out from the model selection process, is not used for the model selection step.
## 9. How to Run the Project
Install the required libraries:
python -m pip install -r requirements.txt
Run the model:

python main.py
An internet connection is needed to download the dataset if it's not cached.
## 10. Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- ucimlrepo

## 11. Dataset Reference

Cortez, P. (2008). Student Performance.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5TG7T