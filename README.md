# Assignment 2: Data Wrangling and Exploratory Analysis

## Dataset

**Dataset:** Titanic Dataset

**Source:** Kaggle Titanic dataset

**Source url:**  https://www.kaggle.com/datasets/sakshisatre/titanic-dataset


**Purpose:** This dataset was selected because it contains more than 200 passenger records and includes both numerical and categorical variables, making it suitable for practicing data cleaning, Pandas transformations, NumPy computations, and exploratory visualization.

The dataset source, license information, and detailed schema are also documented in the Assignment 2 notebook.

## Analysis

This assignment applies NumPy and Pandas to a real-world dataset. The analysis includes:

* Dataset schema and data-quality checks
* Missing-value analysis and cleaning
* GroupBy aggregation and merge operations
* NumPy-based fare standardization
* Exploratory visualizations using Matplotlib
* Feature engineering
* A final analysis report and reflection

## Results

The final cleaned and feature-engineered dataset contains **1,309 rows and 19 columns**.

The analysis found differences in survival rates across passenger classes and age groups. First-class passengers had a higher survival rate than lower-class passengers, while children had the highest survival rate among the age groups analyzed.

## Project Structure

```text
data/
├── raw/
└── processed/
    └── titanic_cleaned.csv

notebooks/
└── assignment2.ipynb

reports/
├── a2_chart1.png
├── a2_chart2.png
└── weekend-a2-report.md
```

## Assignment Branch

This work is completed on the `assignment2-data-wrangling` branch.

## Assignment 3 — Machine Learning Regression

For Assignment 3, the cleaned Titanic dataset from Assignment 2 was used to
build regression models for predicting passenger fare.

### Question

Can passenger fare be predicted using passenger class, age, number of
siblings/spouses, and number of parents/children aboard?

### Features

- `pclass`
- `age`
- `sibsp`
- `parch`

### Target

- `fare`

### Models

- Linear Regression
- Random Forest Regressor

### Train/Test Split

The dataset was divided into:
- 80% training data
- 20% testing data

A `random_state` of 42 was used for reproducibility.

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 21.77 | 40.93 | 0.369 |
| Random Forest | 16.33 | 43.98 | 0.272 |

Linear Regression performed better overall based on RMSE and R², while Random
Forest achieved a lower MAE.

### Dataset

The Titanic dataset was obtained from the source documented in Assignment 2.
