## Recharge Propensity Model (0 → 1 Conversion Prediction)

<img width="1536" height="1024" alt="fe6d9f88-8733-4b4a-b213-3c10df09b62c" src="https://github.com/user-attachments/assets/7e21c1aa-111e-4ac3-8d7f-01811a3c5a15" />

## Overview


This repository contains an end-to-end Machine Learning pipeline for predicting which users are most likely to make their first recharge or first purchase.

The project uses LightGBM for binary classification and generates:

* User-level propensity scores
* Ranked targeting lists
* Feature importance analysis
* Threshold performance analysis
* Decile distribution reports
* Model evaluation metrics
* CRM-ready prediction files


The primary goal is to help marketing, CRM, growth, and retention teams identify high-intent users and improve campaign efficiency.

---

## Problem Statement

Many businesses acquire large numbers of users, but only a small percentage make their first purchase.

Instead of targeting all users equally, this model predicts:

> Which users are most likely to make their first purchase?

The resulting probability score can be used to:

* Prioritize CRM campaigns
* Optimize marketing budgets
* Improve conversion rates
* Personalize user journeys
* Improve customer targeting efficiency

---

## Features Used

The model can utilize behavioral, demographic, acquisition, and device-related attributes.

### User Profile

* Age
* Gender
* Country
* State
* City

### Signup Information

* Signup Day
* Signup Time

### Consultation Behavior

* IsFreeConsultationTaken
* HasTakenChatConsultation
* HasTakenCallConsultation
* HasTakenOtherConsultation

### User Engagement

* HasRated
* HasReviewed
* HadPositiveInteraction
* HadNegativeInteraction
* HasUsedGift

### Device Information

* DeviceType
* device_manufacturer
* device_name
* os_name
* os_version
* app_version
* platform
* store

### Activity Metrics

* lifetime_session_count
* session_count
* time_spent

### Acquisition Data

* Acquisition Source
* Campaign
* Campaign Adgroup
* Campaign Group

---

## Model Architecture

### Algorithm

LightGBM Classifier

### Objective

Binary Classification

Target Variable:

```text
HasRecharged

0 = Did Not Recharge
1 = Recharged
```

### Key Components

* Train/Test Split
* Missing Value Handling
* Categorical Encoding
* Feature Importance Analysis
* Early Stopping
* Threshold Optimization
* Decile Analysis
* Probability Scoring

---

## Project Structure

```text
open_0_to_1_recharge_model/
│
├── data/
│   ├── train_data.csv
│   ├── predict_data.csv
│   └── sample_data_dictionary.xlsx
│
├── models/
│   └── trained_model.pkl
│
├── notebooks/
│   └── model_exploration.ipynb
│
├── outputs/
│   ├── all_predicted_users.csv
│   ├── top_10pct_users.csv
│   ├── top_20pct_users.csv
│   └── Model_Analysis_Report.xlsx
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── utils.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Input Data Requirements

### Training Dataset

The training dataset must contain:

```text
UserLoginId
HasRecharged
```

along with all feature columns used by the model.

Example:

| UserLoginId | Age | Gender | Session_Count | HasRecharged |
| ----------- | --- | ------ | ------------- | ------------ |
| 1001        | 25  | Male   | 10            | 1            |
| 1002        | 31  | Female | 3             | 0            |

### Prediction Dataset

The prediction dataset should contain:

```text
UserLoginId
```

and all feature columns.

It should not contain:

```text
HasRecharged
```

Example:

| UserLoginId | Age | Gender | Session_Count |
| ----------- | --- | ------ | ------------- |
| 2001        | 28  | Male   | 15            |
| 2002        | 35  | Female | 4             |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/recharge-propensity-model.git

cd recharge-propensity-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main libraries:

```text
pandas
numpy
lightgbm
scikit-learn
matplotlib
openpyxl
pyyaml
```

Install:

```bash
pip install -r requirements.txt
```

---

## Running The Model

### Step 1: Add Input Files

Place your datasets inside the data folder:

```text
data/
├── train_data.csv
└── predict_data.csv
```

### Step 2: Run Training Script

```bash
python src/train.py
```

The script will prompt for:

```text
Enter path for Train+Test dataset CSV file:
```

Example:

```text
data/train_data.csv
```

Then:

```text
Enter path for Predictions dataset CSV file:
```

Example:

```text
data/predict_data.csv
```

### Step 3: Select Export Percentage

Enter the percentage of highest-scoring users you want exported.

Example:

```text
10
```

The model will generate:

```text
top_10pct_users.csv
```

containing the highest propensity users.

---

## Outputs Generated

### all_predicted_users.csv

Contains scores for all users in the prediction dataset.

Example:

| UserLoginId | Score |
| ----------- | ----- |
| 10001       | 0.91  |
| 10002       | 0.73  |
| 10003       | 0.18  |

---

### top_Xpct_users.csv

Contains only the highest propensity users.

Examples:

```text
top_5pct_users.csv
top_10pct_users.csv
top_20pct_users.csv
```

Typically used for CRM campaigns, push notifications, remarketing, and conversion programs.

---

### Model_Analysis_Report.xlsx

Generated automatically and contains multiple sheets:

#### Feature_Importance

Ranks features by contribution to model predictions.

#### Prediction_Decile_Summary

Distribution of prediction scores across deciles.

#### Test_Decile_Summary

Model performance across test-set deciles.

#### Dataset_Info

Training, testing, and prediction dataset sizes.

#### Threshold_Analysis

Performance metrics across multiple score thresholds.

---

## Model Evaluation

The model automatically evaluates:

* Accuracy
* True Positives
* False Positives
* True Negatives
* False Negatives
* Probability Distribution
* Decile Capture Rates
* Feature Importance

Thresholds evaluated:

```text
0.10
0.25
0.50
0.75
0.80
0.90
```

---

## Example Use Cases

### CRM Campaign Prioritization

Target only the users most likely to convert.

### Marketing Optimization

Focus spend on high-intent users.

### Push Notification Targeting

Send personalized nudges to users with strong purchase intent.

### Customer Segmentation

Create High, Medium, and Low Intent user segments.

### Conversion Analytics

Measure conversion likelihood across acquisition channels and user cohorts.

---

## Machine Learning Workflow

```text
Raw User Data
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Categorical Encoding
      │
      ▼
Train/Test Split
      │
      ▼
LightGBM Training
      │
      ▼
Model Evaluation
      │
      ▼
Probability Scoring
      │
      ▼
User Ranking
      │
      ▼
Target User Export
```

---

## Future Improvements

Potential enhancements:

* SHAP Explainability
* Hyperparameter Optimization
* Automated Feature Selection
* MLflow Experiment Tracking
* XGBoost Benchmarking
* CatBoost Benchmarking
* Model Monitoring
* API Deployment
* Real-Time Scoring

---

## Disclaimer

This repository is intended for educational and commercial machine learning applications.

The included sample data is synthetic and does not contain any real user information.

---

## License

MIT License

Feel free to use, modify, and distribute this project under the terms of the MIT License.
