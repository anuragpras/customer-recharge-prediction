# Open 0→1 Recharge Propensity Model

A generic open-source **0→1 conversion propensity model** for predicting which non-paying / never-recharged users are most likely to make their **first recharge**.

This project is adapted from an internal LightGBM recharge prediction workflow and converted into a reusable GitHub portfolio project.

## What this model does

The model scores users from `0` to `1`.

- Higher score = higher probability of first recharge
- Lower score = lower probability of first recharge

## Target definition

```text
HasRecharged = 1 → User completed first recharge within the target window
HasRecharged = 0 → User did not complete first recharge within the target window
```

This makes it a **0→1 model**, not a 1→Many or churn model.

## Included outputs

The mock output folder includes:

1. `feature_importance_mock.csv`
2. `train_percentile_distribution_mock.csv`
3. `test_percentile_distribution_mock.csv`
4. `prediction_percentile_distribution_mock.csv`
5. `feature_level_score_bucket_analysis_mock.csv`
6. `all_predicted_users_mock.csv`
7. `Model_Analysis_Report_Mock.xlsx`

## Project structure

```text
open_0_to_1_recharge_model/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── sample_train.csv
│   └── sample_prediction.csv
├── src/
│   ├── generate_sample_data.py
│   ├── train_model.py
│   └── predict.py
├── outputs/
│   └── mock_run/
│       ├── feature_importance_mock.csv
│       ├── train_percentile_distribution_mock.csv
│       ├── test_percentile_distribution_mock.csv
│       ├── prediction_percentile_distribution_mock.csv
│       ├── feature_level_score_bucket_analysis_mock.csv
│       ├── all_predicted_users_mock.csv
│       └── Model_Analysis_Report_Mock.xlsx
└── models/
```

## Setup

```bash
pip install -r requirements.txt
```

## Generate sample data

```bash
python src/generate_sample_data.py
```

## Train model

```bash
python src/train_model.py --config config.yaml
```

## Predict users

```bash
python src/predict.py --config config.yaml
```

## Main business use cases

This type of 0→1 model can be used for:

- First recharge targeting
- CRM campaign prioritization
- New user activation
- Promotional offer selection
- Reducing blanket campaign costs
- Prioritizing high-intent users for WhatsApp, push, email, or call campaigns

## Output interpretation

### Score buckets

| Score Bucket | Meaning |
|---|---|
| 0.80–1.00 | Very high first-recharge intent |
| 0.60–0.80 | High intent |
| 0.40–0.60 | Medium intent |
| 0.20–0.40 | Low intent |
| 0.00–0.20 | Very low intent |

### Percentile distribution

Percentiles split users into ranked groups by probability score.

For example:

- Top 10% users = highest-scoring users
- Bottom 10% users = lowest-scoring users

## Feature-level score bucket analysis

This output helps explain how user behavior changes across score buckets.

Example:

```text
High score users may have:
- More app launches
- More sessions
- More time spent
- Higher consultation initiation rate
- More notification clicks
```

## Model used

Default model:

```text
LightGBM Binary Classifier
```

Why LightGBM?

- Fast on large tabular datasets
- Handles nonlinear relationships
- Works well with mixed numeric and categorical features
- Provides feature importance
- Good for CRM propensity scoring use cases

## Notes

The sample data and mock outputs are synthetic and only for testing/demo purposes.
