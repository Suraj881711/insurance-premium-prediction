# 🛡️ Insurance Premium Prediction

A Flask web app that estimates a person's annual health-insurance charges
from age, sex, BMI, number of children, smoker status and region, using a
Linear Regression model.

## What it does

Fill in the form and the app instantly estimates the insurance premium.
Smoker status and BMI influence the estimate the most — try the sample:
a 40-year-old male smoker (BMI 30, 2 children, southeast) estimates at
about **$32,394**, while a 25-year-old non-smoker estimates under $1,000.

## Features

- Instant premium estimate from a trained Linear Regression model
- Your entered details are echoed on the result page
- Friendly validation errors (age 18–100, BMI 10–60, …) instead of
  raw error pages
- `/health` endpoint

## Verified model performance

- **R² = 0.7507**, MAE ≈ $4,172 — computed by scoring the saved model on
  the full dataset it was trained on (1,338 records; the training
  notebook fits on the full data without a separate test split, so this
  is a training-set score, stated honestly)

## Tech stack

- Python, Flask
- scikit-learn (LinearRegression), pandas, NumPy, joblib

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and submit the form.

## Files

- `app.py` — Flask routes (`/`, `/about`, `/predict`, `/health`)
- `model.lb` — trained Linear Regression model
- `vvfv.ipynb` — model training notebook
- `insurance.csv` — the dataset (downloaded from the public source the
  notebook uses, saved here so the project is self-contained)

## Dataset note

The well-known `insurance.csv` dataset (1,338 US health-insurance
records): age, sex, BMI, children, smoker, region → charges.
