# IDS 568 – Milestone 3  
## End-to-End Machine Learning Pipeline with MLflow, CI/CD, and Data Lineage


# Project Overview

This project implements an **end-to-end machine learning pipeline** for training and validating a classification model using the **Iris dataset**. The pipeline demonstrates modern **MLOps practices**, including experiment tracking, automated validation, continuous integration, and reproducibility.

The system automatically performs the following tasks:

- Data preprocessing
- Model training
- Experiment tracking using MLflow
- Model validation with quality gates
- Artifact logging
- CI/CD automation with GitHub Actions

The goal is to build a **reproducible and automated machine learning workflow** that can reliably train and validate models.

# Pipeline Architecture
The machine learning workflow follows this structure:
Raw Data  
↓  
`preprocess.py`  
↓  
Processed Dataset  
↓  
`train.py`  
↓  
Model Training + MLflow Logging  
↓  
`model_validation.py`  
↓  
Quality Gate Evaluation  
↓  
Metrics & Model Artifacts
# Repository Structure
IDS568-milestone3-abhow5
│
├── .github/workflows/train_and_validate.yml
├── dags/
│ └── training_pipeline.py
├── data/
│ └── processed/
├── models/
│ └── model.pkl
├── reports/
│ └── metrics.json
├── preprocess.py
├── train.py
├── model_validation.py
├── register_best.py
├── requirements.txt
├── README.md
└── lineage_report.md

# Tools and Technologies

The project uses the following tools:

| Tool | Purpose |
|-----|------|
| Python | Core programming language |
| Scikit-learn | Machine learning model training |
| MLflow | Experiment tracking and artifact management |
| GitHub Actions | Continuous Integration / CI pipeline |
| Airflow | Workflow orchestration (DAG structure) |
| Pandas | Data processing |
| Joblib | Model serialization |

# Dataset

The project uses the **Iris dataset**, a well-known dataset for classification tasks.

Features:

- sepal length
- sepal width
- petal length
- petal width

Target:

- iris species classification
# Pipeline Components

## 1. Data Preprocessing

File: `preprocess.py`

This script prepares the dataset for model training by:

- Loading the raw dataset
- Cleaning and formatting the data
- Saving the processed dataset

Output:
data/processed/iris_processed.csv
## 2. Model Training

File: `train.py`

This script trains a **Logistic Regression model** using the processed dataset.

The script performs the following tasks:

- Splits data into training and testing sets
- Trains the model
- Calculates evaluation metrics
- Saves the trained model

Model output:
models/model.pkl
Metrics generated:
- Accuracy
- F1 Score
## 3. MLflow Experiment Tracking

MLflow is used to track:

- Training parameters
- Model performance metrics
- Model artifacts

Tracked items include:

Parameters:

C
max_iter
test_size
random_state
Metrics:
accuracy
f1_macro
Artifacts:
model.pkl
metrics.json
MLflow experiment name:
ids568_milestone3
## 4. Model Validation (Quality Gate)
File: `model_validation.py`

This script checks whether the trained model satisfies minimum performance thresholds.

Quality thresholds:
Accuracy ≥ 0.90
F1 Score ≥ 0.80
If the model fails these thresholds, the pipeline fails automatically.
## 5. Continuous Integration (GitHub Actions)

File:
.github/workflows/train_and_validate.yml

The CI pipeline automatically executes the following steps when code is pushed:

1. Install dependencies
2. Run data preprocessing
3. Train the machine learning model
4. Run validation checks
5. Upload metrics as artifacts

This ensures that the model pipeline remains reproducible and functional.
## 6. Airflow DAG

File:
dags/training_pipeline.py
The Airflow DAG represents the ML workflow as a sequence of tasks:
Preprocess → Train → Validate
This demonstrates how the ML pipeline can be orchestrated in a production environment.

# How to Run the Project Locally

Install dependencies:
pip install -r requirements.txt
Run preprocessing:
python preprocess.py
Train the model:
python train.py
Validate the model:
python model_validation.py
# CI/CD Pipeline

The CI pipeline automatically executes the training pipeline whenever new code is pushed to the repository.

GitHub Actions performs:

- dependency installation
- preprocessing
- training
- validation
- artifact upload

This ensures automated model verification.
# Key Outcomes

This project demonstrates the implementation of an **end-to-end machine learning operations pipeline**, integrating:
- experiment tracking
- automated validation
- artifact management
- CI/CD workflows
- reproducible training pipelines