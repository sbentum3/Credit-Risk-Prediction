# Credit Risk Analysis & Loan Default Prediction

An end-to-end credit-risk analytics project that uses exploratory data analysis and supervised machine learning to identify patterns associated with loan default and evaluate predictive performance.

## Project Overview

Credit risk analysis involves assessing the likelihood that a borrower will fail to meet their repayment obligations. In this project, borrower demographics, financial characteristics, loan attributes, and prior credit information are analyzed to understand their relationship with observed loan-default outcomes.

The project combines descriptive analytics with predictive modeling to answer two practical questions:

1. **What borrower and loan characteristics are associated with observed default outcomes?**
2. **How effectively can machine learning models distinguish default from non-default borrowers?**

## Objectives

- Assess data quality and prepare the dataset for analysis.
- Explore default patterns across borrower and loan characteristics.
- Quantify relationships between numerical variables and loan status.
- Establish an interpretable Logistic Regression baseline.
- Develop a Random Forest classification model.
- Compare model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Examine Random Forest feature importance.
- Document limitations and considerations for responsible credit-risk modeling.

## Dataset

The project uses a loan-level credit-risk dataset containing **32,581 original observations and 12 variables**.

Key variables include:

| Variable | Description |
|---|---|
| `person_age` | Borrower's age |
| `person_income` | Annual borrower income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Target variable: default / non-default |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_default_on_file` | Previous default recorded on file |
| `cb_person_cred_hist_length` | Length of credit history |

### Data preparation

The preprocessing workflow:

1. Removed observations with implausible age and employment-length values.
2. Imputed missing interest rates using the median interest rate within each loan grade.
3. Imputed missing employment length using the median.
4. Removed duplicate observations.
5. Verified that no missing values remained.

After preprocessing and duplicate removal, the final modeling dataset contained **32,409 observations**.

## Exploratory Analysis

The target variable is moderately imbalanced:

- **25,321 non-default observations — 78.13%**
- **7,088 default observations — 21.87%**

### Key observations

Default rates vary substantially across loan grades:

| Loan Grade | Observed Default Rate |
|---|---:|
| A | 9.96% |
| B | 16.32% |
| C | 20.76% |
| D | 59.05% |
| E | 64.49% |
| F | 70.54% |
| G | 98.44% |

Defaulted borrowers also show different average financial characteristics in the dataset:

| Measure | Non-default | Default |
|---|---:|---:|
| Annual income | GHS 70,597 | GHS 49,093 |
| Loan amount | GHS 9,239 | GHS 10,854 |
| Interest rate | 10.45% | 13.05% |
| Loan-to-income ratio | 0.15 | 0.25 |

Selected categorical variables also show differences in observed default rates. For example, the default rate is 31.61% among renters versus 7.49% among homeowners, while borrowers with a previous default recorded on file have a 37.86% default rate versus 18.44% for those without one.

These are descriptive associations within the dataset and should not be interpreted as causal relationships.

## Machine Learning Approach

Two classification models were evaluated.

### 1. Logistic Regression

Logistic Regression was used as an interpretable baseline for binary default classification.

### 2. Random Forest

A Random Forest classifier with **300 trees** and `class_weight='balanced'` was used to capture nonlinear relationships and interactions while accounting for class imbalance.

### Preprocessing

The modeling pipeline applies:

- `StandardScaler` to numerical variables.
- `OneHotEncoder(handle_unknown='ignore')` to categorical variables.
- A stratified 80/20 train-test split with `random_state=42`.

## Model Performance

The models were evaluated on the same held-out test set.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8630 | 0.9355 |
| Precision | 0.7563 | 0.9771 |
| Recall | 0.5515 | 0.7221 |
| F1 Score | 0.6378 | 0.8305 |
| ROC-AUC | 0.8703 | 0.9359 |

For the default class, Random Forest achieved a recall of **72.21%** and an F1 score of **83.05%** on the test set.

These results describe performance on this dataset and holdout split; they do not establish how the model would perform in a different population or future period.

## Feature Importance

The Random Forest model identified the following variables among its most influential predictive features:

1. `loan_percent_income`
2. `person_income`
3. `loan_int_rate`
4. `loan_amnt`
5. `person_emp_length`
6. `person_age`
7. Selected loan-grade and home-ownership indicators

The results suggest that repayment burden, borrower income, loan pricing, and loan size contain substantial predictive information within the fitted model.

Feature importance should be interpreted as predictive contribution rather than causal impact.

## Business & Credit-Risk Relevance

The analysis demonstrates a practical workflow that can support credit-risk analytics:

- **Portfolio monitoring:** identify variables associated with higher observed default rates.
- **Credit assessment:** evaluate predictive signals before developing a production scorecard.
- **Risk segmentation:** understand how risk patterns differ across borrower and loan characteristics.
- **Model benchmarking:** compare interpretable statistical models with nonlinear machine learning approaches.
- **Decision support:** provide analytical evidence that can complement, rather than replace, established credit policies and governance.

A production credit model would require additional validation, calibration, monitoring, explainability, and fairness assessment before being used for lending decisions.

## Limitations & Future Work

### 1. External validation

The analysis uses a single dataset and a single stratified holdout test set. Time-based and external validation would provide stronger evidence of generalization.

### 2. Additional borrower information

Future versions could incorporate repayment history, debt obligations, credit utilization, account behavior, and macroeconomic variables.

### 3. Explainability

SHAP values and partial-dependence analysis could provide more detailed explanations of model behavior and individual predictions.

### 4. Fairness and model governance

Future work should evaluate performance across relevant borrower groups, investigate potential bias, and incorporate appropriate model-risk controls.

### 5. Production monitoring

A production implementation should include probability calibration, decision-threshold analysis, data-quality monitoring, model-drift monitoring, and periodic model validation.

## Project Structure

```text
.
├── Credit_Risk_Analysis_Professional.ipynb
├── credit_risk_dataset.csv
└── README.md
```

## Technologies

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, modeling, and evaluation


## Key Takeaway

This project demonstrates an end-to-end approach to credit-risk analytics, moving from data-quality assessment and exploratory analysis to predictive modeling and model interpretation.

The analysis shows meaningful differences in observed default rates across borrower and loan characteristics and demonstrates how Logistic Regression and Random Forest can provide complementary perspectives on loan-default prediction.

---

**Author:** Samuel Bentum  
**Focus:** Credit Risk Analytics | Data Analytics | Portfolio Analytics | Machine Learning

